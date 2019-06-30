import os
import sqlite3
import zlib

import numpy as np
import io

from competition_submission.consts import JOINT_POSITIONS, TOUCH_SENSORS, RETINA, GOAL

CREATE_TABLE = """ CREATE TABLE IF NOT EXISTS past_experiences (
                                        id integer PRIMARY KEY,
                                        observation_number integer,
                                        %s array,
                                        %s array,
                                        %s array,
                                        %s array,
                                        %s array,
                                        %s array,
                                        %s array,
                                        %s array
                                        ); """ % ("initial_%s" % JOINT_POSITIONS,
                                                  "initial_%s" % TOUCH_SENSORS,
                                                  "initial_%s" % RETINA,
                                                  "action",
                                                  "result_%s" % JOINT_POSITIONS,
                                                  "result_%s" % TOUCH_SENSORS,
                                                  "result_%s" % RETINA,
                                                  GOAL)

REPLACE_ROW_IN_TABLE = "replace into past_experiences (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) values (?,?,?,?,?,?,?,?,?,?)" % (
    "id",
    "observation_number",
    "initial_%s" % JOINT_POSITIONS,
    "initial_%s" % TOUCH_SENSORS,
    "initial_%s" % RETINA,
    "action",
    "result_%s" % JOINT_POSITIONS,
    "result_%s" % TOUCH_SENSORS,
    "result_%s" % RETINA,
    GOAL
)


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
        self.conn = sqlite3.connect(db_file_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute(CREATE_TABLE)
        self.observation_number = 0
        self.memory_size = memory_size

    def insert_observation(self, previous_observation, current_observation, goal, action):
        self.cursor.execute(REPLACE_ROW_IN_TABLE,
                            (
                                self.observation_number % self.memory_size,
                                self.observation_number,
                                np.asarray(previous_observation[JOINT_POSITIONS]),
                                np.asarray(previous_observation[TOUCH_SENSORS]),
                                np.asarray(previous_observation[RETINA]),
                                np.asarray(action),
                                np.asarray(current_observation[JOINT_POSITIONS]),
                                np.asarray(current_observation[TOUCH_SENSORS]),
                                np.asarray(current_observation[RETINA]),
                                goal
                            ))
        self.conn.commit()
        self.observation_number += 1
