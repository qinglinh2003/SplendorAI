import numpy as np
from utils.valid_action_utils import compute_valid_actions_from_obs


class RandomAgent:

    def act(self, observation):
        valid_actions = compute_valid_actions_from_obs(observation)
        valid_indices = np.where(valid_actions)[0]

        return np.random.choice(valid_indices)

