from env.splendor_env import SplendorEnv
from utils.valid_action_utils import compute_valid_actions_from_obs
import numpy as np

class PPOAgent:
    def __init__(self, model):
        self.model = model

    def act(self, observation):
        action_mask = compute_valid_actions_from_obs(observation)
        action, _states = self.model.predict(observation, deterministic=True, action_masks=action_mask)
        return action

