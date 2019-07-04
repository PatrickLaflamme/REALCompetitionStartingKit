import numpy as np

from competition_submission.consts import JOINT_POSITIONS, TOUCH_SENSORS, RETINA, GOAL
from competition_submission.utils.helper_functions import mse, initialize_array

INITIAL_PREFIX = "initial_%s"
RESULT_PREFIX = "result_%s"
OBSERVATION_NUMBER = "observation_number" 
NOVELTY_SCORE = "novelty_score"
INITIAL_JOINT_POSITIONS = INITIAL_PREFIX % JOINT_POSITIONS
INITIAL_TOUCH_SENSORS = INITIAL_PREFIX % TOUCH_SENSORS
INITIAL_RETINA = INITIAL_PREFIX % RETINA
ACTION = "action"
RESULT_JOINT_POSITIONS = RESULT_PREFIX % JOINT_POSITIONS
RESULT_TOUCH_SENSORS = RESULT_PREFIX % TOUCH_SENSORS
RESULT_RETINA = RESULT_PREFIX % RETINA


class Experience:
    def __init__(self,
                 observation_number,
                 novelty_score,
                 initial_joint_positions,
                 initial_touch_sensors,
                 initial_retina,
                 action,
                 result_joint_positions,
                 result_touch_sensors,
                 result_retina,
                 goal,
                 reward_strategy):
        assert isinstance(goal, Goal)
        self.observation_number = observation_number
        self.novelty_score = novelty_score
        self.initial_joint_positions = initial_joint_positions
        self.initial_touch_sensors = initial_touch_sensors
        self.initial_retina = initial_retina
        self.action = action
        self.result_joint_positions = result_joint_positions
        self.result_touch_sensors = result_touch_sensors
        self.result_retina = result_retina
        self.goal = goal
        self.reward = reward_strategy(self.result_retina, self.goal.retina)


class ExperienceBatch:
    def __init__(self, batch_size=None, experience_list=None):
        assert batch_size is not None or experience_list is not None
        if batch_size is not None and experience_list is not None:
            assert len(experience_list) == batch_size
        elif batch_size is None:
            batch_size = len(experience_list)

        self.observation_numbers = initialize_array(batch_size)
        self.novelty_scores = initialize_array(batch_size)
        self.initial_joint_positions = initialize_array(batch_size)
        self.initial_touch_sensors = initialize_array(batch_size)
        self.initial_retinas = initialize_array(batch_size)
        self.actions = initialize_array(batch_size)
        self.result_joint_positions = initialize_array(batch_size)
        self.result_touch_sensors = initialize_array(batch_size)
        self.result_retinas = initialize_array(batch_size)
        self.goal_retinas = initialize_array(batch_size)
        self.goal_joint_positions = initialize_array(batch_size)
        self.goal_touch_sensors = initialize_array(batch_size)
        self.rewards = initialize_array(batch_size)
        self.batch_size = batch_size
        self.populated_size = 0

        if experience_list is not None:
            for i in range(batch_size):
                self.add_experience_to_batch(experience_list[i])

    def add_experience_to_batch(self, experience):
        assert isinstance(experience, Experience)
        self.populated_size += 1
        try:
            self.observation_numbers[self.populated_size] = experience.observation_number
            self.novelty_scores[self.populated_size] = experience.novelty_score
            self.initial_joint_positions[self.populated_size] = experience.initial_joint_positions
            self.initial_touch_sensors[self.populated_size] = experience.initial_touch_sensors
            self.initial_retinas[self.populated_size] = experience.initial_retina
            self.actions[self.populated_size] = experience.action
            self.result_joint_positions[self.populated_size] = experience.result_joint_positions
            self.result_touch_sensors[self.populated_size] = experience.result_touch_sensors
            self.result_retinas[self.populated_size] = experience.result_retina
            self.goal_joint_positions[self.populated_size] = experience.goal.joint_positions
            self.goal_retinas[self.populated_size] = experience.goal.retina
            self.goal_touch_sensors[self.populated_size] = experience.goal.touch_sensors
        except Exception:
            self.populated_size -= 1


class Goal:
    def __init__(self, retina, joint_positions, touch_sensors):
        self.retina = retina
        self.joint_positions = joint_positions
        self.touch_sensors = touch_sensors

    @staticmethod
    def from_experience(experience):
        assert isinstance(experience, Experience)
        return Goal(experience.result_retina, experience.result_joint_positions, experience.result_touch_sensors)


class ExperienceStore:
    def __init__(self, memory_size):
        self.observation_number = 0
        self.memory_size = memory_size
        self.memory_store = [None] * self.memory_size
        self.novelty_decay = 0.5
        self.image_total = None

    def insert_observation(self, previous_observation, current_observation, goal, action):
        normalized_image = current_observation[RETINA].astype(np.float)/255
        if self.image_total is None:
            self.image_total = np.zeros_like(current_observation[RETINA], dtype=np.float)
        self.image_total += normalized_image
        mse_score = mse(self.image_total/(self.observation_number + 1), normalized_image)
        self.memory_store[self.observation_number % self.memory_size] = Experience(
            observation_number=self.observation_number,
            novelty_score=mse_score,
            initial_joint_positions=np.asarray(previous_observation[JOINT_POSITIONS]),
            initial_touch_sensors=np.asarray(previous_observation[TOUCH_SENSORS]),
            initial_retina=np.asarray(previous_observation[RETINA]),
            action=np.asarray(action),
            result_joint_positions=np.asarray(current_observation[JOINT_POSITIONS]),
            result_touch_sensors=np.asarray(current_observation[TOUCH_SENSORS]),
            result_retina=np.asarray(current_observation[RETINA]),
            goal=goal,
            reward_strategy=mse
        )
        self.observation_number += 1

    def select_new_goal(self):

        mse_scores = [memory.novelty_score for memory in self.memory_store if memory is not None]
        normalized_mse_scores = np.asarray(mse_scores)/np.sum(mse_scores)
        selected_memory_id = np.random.choice(min(self.observation_number, self.memory_size), p=normalized_mse_scores)
        new_goal = Goal.from_experience(self.memory_store[selected_memory_id])
        modified_novelty_score_for_selected_memory = mse(self.image_total/self.observation_number,
                                                         new_goal.retina.astype(np.float)/255)
        self.memory_store[selected_memory_id].novelty_score = modified_novelty_score_for_selected_memory
        return new_goal

    def get_memory_replay_batch(self, batch_size):
        chosen_ids = np.random\
            .choice(min(self.observation_number, self.memory_size), size=batch_size-1)
        chosen_ids = np.concatenate((chosen_ids, [(self.observation_number - 1) % self.memory_size]), axis=0)
        batch = ExperienceBatch(batch_size)
        for chosen_id in chosen_ids:
            batch.add_experience_to_batch(self.memory_store[chosen_id])
        return batch
