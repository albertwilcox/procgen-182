import tensorflow as tf
from abc import ABC


class Agent(ABC):

    def __init__(self, file):
        self.policy = tf.keras.models.load_model(file, compile=False)

    def get_action(self, observation):
        raise NotImplementedError("Please use a subclass, QAgent or PPOAgent")

class PPOAgent(Agent):

    def get_action(self, observation):
        pass