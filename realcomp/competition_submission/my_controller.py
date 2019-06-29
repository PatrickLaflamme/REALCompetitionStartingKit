import numpy as np
import os

from competition_submission.utils.experience_store import ExperienceStore

DB_FILE = os.path.join(os.getcwd(), "controller-memory.db")

JOINT_POSITIONS = "joint_positions"
TOUCH_SENSORS = "touch_sensors"
RETINA = "retina"
GOAL = "goal"


class ControllerWrapper:
    """
    A controller wrapper that handles the setting of goals if one is present. The actual controller logic will live elsewhere
    """
    def __init__(self, action_space):
        """
        initializes the ControllerWrapper.
        :param action_space:  ndarray - the action space of the environment
        """
        self.action_space = action_space
        self.action = np.zeros(action_space.shape[0])
        self.conn = None
        self.cursor = None
        self.previous_state = None
        self.experience_store = None
        self.goal = None
        self.experience_store_initialized = False

    def step(self, observation, reward, done):
        if self.previous_state is not None and self.experience_store_initialized:
            self.experience_store.insert_observation()
        if self._is_testing_step(observation[GOAL]):
            return self._perform_action(observation, reward, done)
        return self._perform_training_step(observation, reward, done)

    def _perform_action(self, observation, reward, done):
        proposed_action = self.action + 0.1*np.pi*np.random.randn(self.action_space.shape[0])
        self.action = np.maximum(np.minimum(proposed_action, self.action_space.high), self.action_space.low)
        self.previous_state = observation
        return self.action

    def _perform_training_step(self, observation, reward, done):
        if not self.replay_actions_db_initialized:
            self._initialize_experience_store()
        action = self._perform_action(observation, reward, done)
        return action

    def _is_testing_step(self, goal):
        """
                  :param: ndarray - the goal retinal image passed in on the observation object.
                  : return:  boolean - Whether or not the step is a testing step
                  """
        return np.max(goal) == 0 and np.min(goal) == 0

    def _initialize_experience_store(self):
        """
        initializes the database where the memories will be stored for memory replay
        """
        self.experience_store = ExperienceStore(DB_FILE)
        self.experience_store_initialized = True


MyController = ControllerWrapper

