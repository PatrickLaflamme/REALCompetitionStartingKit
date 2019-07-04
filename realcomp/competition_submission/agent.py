import tensorflow as tf
from tensorflow.python.keras.models import load_model

from competition_submission.consts import RETINA, JOINT_POSITIONS, TOUCH_SENSORS
from competition_submission.utils.experience_store import ExperienceBatch


class DeepQAgent:
    def __init__(self, model):
        self.model = model

    def choose_action(self, observation, goal):
        inputs = [
            observation[RETINA],
            goal,
            observation[JOINT_POSITIONS],
            observation[TOUCH_SENSORS]
        ]
        output = self.model.predict(inputs)
        return output[-1]

    def training_step(self, experience_batch):
        assert isinstance(experience_batch, ExperienceBatch)
        inputs = [
            experience_batch.initial_joint_positions,
            experience_batch.initial_touch_sensors,
            experience_batch.initial_retinas,
            experience_batch.goal_retinas
        ]
        outputs = [
            experience_batch.rewards
        ]
        return self.model.train_on_batch(x=inputs, y=outputs, reset_metrics=False)

    def save_agent(self, filename):
        self.model.save(filename)

    @staticmethod
    def load_agent(filename):
        return DeepQAgent(load_model(filename))


if __name__ == '__main__':
    print(tf.keras.__version__)
