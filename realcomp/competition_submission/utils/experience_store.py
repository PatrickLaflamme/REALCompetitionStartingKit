import os
import sqlite3
import numpy as np
import io

from competition_submission.my_controller import JOINT_POSITIONS, TOUCH_SENSORS, RETINA, GOAL

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

compressor = 'zlib'  # zlib, bz2


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    # zlib uses similar disk size that Matlab v5 .mat files
    # bz2 compress 4 times zlib, but storing process is 20 times slower.
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read().encode(compressor))  # zlib, bz2


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    out = io.BytesIO(out.read().decode(compressor))
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
                                previous_observation[JOINT_POSITIONS],
                                previous_observation[TOUCH_SENSORS],
                                previous_observation[RETINA],
                                action,
                                current_observation[JOINT_POSITIONS],
                                current_observation[TOUCH_SENSORS],
                                current_observation[RETINA],
                                goal
                            ))
