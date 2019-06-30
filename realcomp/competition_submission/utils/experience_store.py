import os
import sqlite3
import zlib

import numpy as np
import io

from competition_submission.consts import JOINT_POSITIONS, TOUCH_SENSORS, RETINA, GOAL
from competition_submission.utils.helper_functions import mse

CREATE_TABLE = """ CREATE TABLE IF NOT EXISTS past_experiences (
                                        id integer PRIMARY KEY,
                                        observation_number integer,
                                        novelty_score double,
                                        %s array,
                                        %s array,
                                        %s array,
                                        %s array,
                                        %s array,
                                        %s array,
                                        %s array,
                                        %s array
                                        ); 
            """ % ("initial_%s" % JOINT_POSITIONS,
                   "initial_%s" % TOUCH_SENSORS,
                   "initial_%s" % RETINA,
                   "action",
                   "result_%s" % JOINT_POSITIONS,
                   "result_%s" % TOUCH_SENSORS,
                   "result_%s" % RETINA,
                   GOAL)

CREATE_INDEX = """CREATE INDEX novelty_score_idx on past_experiences (novelty_score);"""

REPLACE_ROW_IN_TABLE = "replace into past_experiences (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) values (?,?,?,?,?,?,?,?,?,?,?)" % (
    "id",
    "observation_number",
    "novelty_score",
    "initial_%s" % JOINT_POSITIONS,
    "initial_%s" % TOUCH_SENSORS,
    "initial_%s" % RETINA,
    "action",
    "result_%s" % JOINT_POSITIONS,
    "result_%s" % TOUCH_SENSORS,
    "result_%s" % RETINA,
    GOAL
)

GET_NOVELTY_SCORES_FOR_MEMORIES = """SELECT novelty_score FROM past_experiences ORDER BY id"""

GET_MEMORY_RETINA_VALUE_BY_ID = "SELECT result_%s FROM past_experiences WHERE id = ?" % RETINA

UPDATE_NOVELTY_SCORE_FOR_MEMORY = "UPDATE past_experiences SET novelty_score = ? WHERE id = ?"


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(zlib.compress(out.read()))


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    out = io.BytesIO(zlib.decompress(out.read()))
    return np.load(out)


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)


class ExperienceStore:
    def __init__(self, db_file_name, memory_size):
        if os.path.exists(db_file_name):
            os.remove(db_file_name)
        self.conn = sqlite3.connect(db_file_name, detect_types=1)
        self.cursor = self.conn.cursor()
        self.cursor.execute(CREATE_TABLE)
        self.cursor.execute(CREATE_INDEX)
        self.observation_number = 0
        self.memory_size = memory_size
        self.novelty_decay = 0.5
        self.image_total = None

    def insert_observation(self, previous_observation, current_observation, goal, action):
        normalized_image = current_observation[RETINA].astype(np.float)/255
        if self.image_total is None:
            self.image_total = np.zeros_like(current_observation[RETINA], dtype=np.float)
        self.image_total += normalized_image
        self.observation_number += 1
        mse_score = mse(self.image_total/self.observation_number, normalized_image)
        self.cursor.execute(REPLACE_ROW_IN_TABLE,
                            (
                                self.observation_number % self.memory_size,
                                self.observation_number,
                                mse_score,
                                np.asarray(previous_observation[JOINT_POSITIONS]),
                                np.asarray(previous_observation[TOUCH_SENSORS]),
                                np.asarray(previous_observation[RETINA]),
                                np.asarray(action),
                                np.asarray(current_observation[JOINT_POSITIONS]),
                                np.asarray(current_observation[TOUCH_SENSORS]),
                                np.asarray(current_observation[RETINA]),
                                np.asarray(goal)
                            ))
        self.conn.commit()

    def select_new_goal(self):
        self.cursor.execute(GET_NOVELTY_SCORES_FOR_MEMORIES)
        mse_scores = self.cursor.fetchall()
        normalized_mse_scores = np.asarray(mse_scores)/np.sum(mse_scores)
        # remove the extra dimension caused by queries being returned as a list of tuples
        normalized_mse_scores = np.squeeze(normalized_mse_scores,1)
        selected_memory_id = np.random.choice(min(self.observation_number, self.memory_size), p=normalized_mse_scores)
        self.cursor.execute(GET_MEMORY_RETINA_VALUE_BY_ID, (selected_memory_id,))
        new_goal = self.cursor.fetchall()[0][0]
        modified_novelty_score_for_selected_memory = mse(self.image_total/self.observation_number,
                                                         new_goal.astype(np.float)/255)
        self.cursor.execute(UPDATE_NOVELTY_SCORE_FOR_MEMORY,
                            (modified_novelty_score_for_selected_memory, selected_memory_id))
        self.conn.commit()
        return new_goal
