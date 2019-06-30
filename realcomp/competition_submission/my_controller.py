import numpy as np
import os

from competition_submission.consts import GOAL, RETINA, DB_FILE, MAX_MEMORY_SIZE, GOAL_THRESHOLD
from competition_submission.utils.experience_store import ExperienceStore
from competition_submission.utils.helper_functions import mse


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
        self.goal = None
        self.steps_on_current_goal = 0
        self.steps_per_goal = 10
        self.experience_store = None
        self.experience_store_initialized = False

    def step(self, observation, reward, done):
        """
        Method called by the task runner
        :param observation: observation object returned by env.step(action)
        :param reward: reward returned by env.step(action)
        :param done: done object returned by env.step(action)
        :return: None
        """
        if not self.experience_store_initialized:
            self._initialize_experience_store()
        if self._is_testing_step(observation[GOAL]):
            self._save_memory(observation, True)
            return self._choose_action(observation, reward, done)
        self._save_memory(observation, False)
        self.steps_on_current_goal += 1
        return self._perform_training_step(observation, reward, done)

    def _save_memory(self, observation, is_testing_step):
        """
        utility function to save an observation to the memory database
        :param observation: observation object returned by env.set(action)
        :param is_testing_step: whether the env is in its extrinsic or intrinsic phase
        :return: None
        """
        if self.previous_state is not None and self.experience_store_initialized and self.goal is not None:
            self.experience_store.insert_observation(self.previous_state,
                                                     observation,
                                                     self.goal,
                                                     self.action)
        if is_testing_step:
            self.goal = observation[GOAL]
        elif self.goal is None:
            self.goal = observation[RETINA]
        elif self.steps_on_current_goal >= self.steps_per_goal or self._state_is_close_to_goal(observation):
            self.goal = self.experience_store.select_new_goal()

    def _state_is_close_to_goal(self, observation):
        """
        determines if the current state is close to the goal set intrinsically
        :param observation: observation object returned by env.set(action)
        :return: None
        """
        return mse(self.goal, observation[RETINA]) < GOAL_THRESHOLD

    def _choose_action(self, observation, reward, done):
        """
        calculate the action to perform
        :param observation: observation object returned by env.set(action)
        :param reward: reward object returned by env.set(action)
        :param done: done object returned by env.set(action)
        :return: list describing the action to perform
        """
        proposed_action = self.action + 0.1*np.pi*np.random.randn(self.action_space.shape[0])
        self.action = np.maximum(np.minimum(proposed_action, self.action_space.high), self.action_space.low)
        self.previous_state = observation
        return self.action

    def _perform_training_step(self, observation, reward, done):
        """
        performs a step during the intrinsic phase of the simulation
        :param observation: observation object returned by env.set(action)
        :param reward: reward object returned by env.set(action)
        :param done: done object returned by env.set(action)
        :return: list describing the action to perform
        """
        action = self._choose_action(observation, reward, done)
        return action

    def _is_testing_step(self, goal):
        """
        :param: ndarray - the goal retinal image passed in on the observation object.
        :return:  boolean - Whether or not the step is a testing step
        """
        return np.max(goal) != 0 and np.min(goal) != 0

    def _initialize_experience_store(self):
        """
        initializes the database where the memories will be stored for memory replay
        """
        self.experience_store = ExperienceStore(DB_FILE, MAX_MEMORY_SIZE)
        self.experience_store_initialized = True


MyController = ControllerWrapper

