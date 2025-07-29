import numpy as np
from utils.valid_action_utils import compute_valid_actions_from_obs, enumerate_all_token_actions


class RuleBasedAgent:
    def __init__(self):
        self.all_token_actions = enumerate_all_token_actions()

    def act(self, observation):
        valid_actions = compute_valid_actions_from_obs(observation)
        valid_indices = np.where(valid_actions)[0]

        if len(valid_indices) == 0:
            raise ValueError("No valid actions available!")


        num_token_actions = 1680
        num_buy_cards = 12
        num_reserve_cards = 15
        num_buy_reserved = 3

        buy_card_start = num_token_actions
        buy_card_end = buy_card_start + num_buy_cards

        reserve_card_start = buy_card_end
        reserve_card_end = reserve_card_start + num_reserve_cards

        buy_reserved_start = reserve_card_end
        buy_reserved_end = buy_reserved_start + num_buy_reserved


        self_tokens = observation["self_tokens"]
        self_purchased_cards = observation["self_purchased_cards"]
        self_score = observation["self_score"][0]
        board_cards = observation["board_cards"]


        available_buy_cards = [action for action in valid_indices if buy_card_start <= action < buy_card_end]
        if available_buy_cards:
            best_card = max(available_buy_cards, key=lambda a: board_cards[a - buy_card_start][1])
            return best_card

        available_token_actions = [action for action in valid_indices if action < num_token_actions]
        if available_token_actions:
            best_token_action = max(available_token_actions, key=lambda a: self._token_value(a, self_tokens))
            return best_token_action


        available_reserve_cards = [action for action in valid_indices if reserve_card_start <= action < reserve_card_end]
        if available_reserve_cards:
            best_reserve = max(available_reserve_cards, key=lambda a: board_cards[a - reserve_card_start][1])
            return best_reserve


        available_buy_reserved = [action for action in valid_indices if buy_reserved_start <= action < buy_reserved_end]
        if available_buy_reserved:
            return np.random.choice(available_buy_reserved)


        return np.random.choice(valid_indices)

    def _token_value(self, action, self_tokens):
        get, discard, final = self.all_token_actions[action]
        final_tokens = self_tokens[:5] + get - discard


        total_tokens = final_tokens.sum()
        overflow_penalty = -5 if total_tokens > 10 else 0


        scarcity_bonus = np.dot(1 / (self_tokens[:5] + 1), get)


        token_gain_value = get.sum() - discard.sum()

        token_value = scarcity_bonus + token_gain_value + overflow_penalty
        return token_value