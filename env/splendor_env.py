import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from entity.board import Board
from entity.player import Player
from utils.valid_action_utils import enumerate_all_token_actions, is_valid_action, can_buy_card


def get_opponent_agent():
    agents = [RuleBasedAgent()]
    #TODO: Generate a random opponent agent every time calling reset
    return random.choice(agents)


class SplendorEnv(gym.Env):
    """
    Environment for Splendor. Supports training against a single opponent

    """
    NUM_BUY_CARDS = 12 # 12 cards to buy from the market
    NUM_RESERVE_BOARD_CARDS = 12 # 12 cards to reserve from the market
    NUM_RESERVE_DECK_CARDS = 3 # 3 tiers of deck from which a hidden card can be reserved
    NUM_BUY_RESERVED = 3 # 3 reserved cards to buy at maximum

    def __init__(self, max_turn=200, num_players=2):
        super().__init__()
        self.max_turn = max_turn
        self.num_players = num_players
        self.num_nobles = 5 - (4 - num_players) # 3 nobles for 2-player games, 4 nobles for 3, 5 nobles for 4
        self.all_token_actions = enumerate_all_token_actions()

        self.board = Board(self.num_players)
        self.players = [Player(self.board) for _ in range(self.num_players)]
        self.current_player_index:int = 0
        self.turn_count:int = 0
        self.opponent_agent = None # reset the agent to play against in `reset()`
        self.agent_player_index = 0 # which player index is the agent, either 0 or 1 in 2-player mode
        self.is_last_round = False # whenever a player hits 15 points, the game will last for one more round until player 0 gets to take turn
        self._terminated = False
        self._truncated = False

        # 1680 actions + 12 cards to buy + 15 cards to reserve + 3 cards to buy from reserve
        self.action_space = spaces.Discrete(len(self.all_token_actions) + self.NUM_BUY_CARDS + self.NUM_RESERVE_BOARD_CARDS + self.NUM_RESERVE_DECK_CARDS + self.NUM_BUY_RESERVED)

        self.observation_space = spaces.Dict({
            "self_tokens": spaces.Box(low=0, high=10, shape=(6,), dtype=np.int32),
            "self_purchased_cards": spaces.Box(low=0, high=10, shape=(5,), dtype=np.int32),
            "self_reserved_cards": spaces.Box(low=0, high=12, shape=(3, 8), dtype=np.int32),
            "self_score": spaces.Box(low=0, high=100, shape=(1, ), dtype=np.int32),

            "opponents_tokens": spaces.Box(low=0, high=10, shape=((self.num_players - 1) * 6,), dtype=np.int32),
            "opponents_purchased_cards": spaces.Box(low=0, high=10, shape=((self.num_players - 1) * 5,), dtype=np.int32),
            "opponents_reserved_cards": spaces.Box(low=0, high=1, shape=((self.num_players - 1) * 3, ), dtype=np.int32), # 0 -> no card, 1 -> has card
            "opponents_scores": spaces.Box(low=0, high=100, shape=(self.num_players - 1,), dtype=np.int32),

            "board_tokens": spaces.Box(low=0, high=10, shape=(6,), dtype=np.int32),
            "board_cards": spaces.Box(low=0, high=40, shape=(12, 8), dtype=np.int32),
            "nobles": spaces.Box(low=0, high=5, shape=(self.num_nobles, 5), dtype=np.int32),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = Board(self.num_players)
        self.players = [Player(self.board) for _ in range(self.num_players)]
        self.current_player_index = 0
        self.agent_player_index = random.randint(0, self.num_players - 1)
        self.turn_count = 0
        self.is_last_round = False
        self._terminated = False
        self._truncated = False

        self.opponent_agent = get_opponent_agent()

        return self.get_obs(), {}

    def action_mask(self):
        # Initialize all actions as invalid
        player = self.players[self.current_player_index]
        mask = np.zeros(self.action_space.n, dtype=bool)

        token_mask = np.array([is_valid_action(action[0], action[1], action[2], player.tokenList[:6], self.board.tokens[:5]) for action in self.all_token_actions])

        buy_card_mask = np.zeros(self.NUM_BUY_CARDS, dtype=bool)
        for i in range(self.NUM_BUY_CARDS):
            row, col = divmod(i, 4)
            card = self.board.market[row][col]
            buy_card_mask[i] = can_buy_card(player, card) # can_buy_card returns false if `card` is None

        reserve_board_card_mask = np.zeros(self.NUM_RESERVE_BOARD_CARDS, dtype=bool)
        if len(player.reserved_cards) < 3 and np.sum(player.tokenList) < 10: # forbids reserving cards when doing so results in a returning token action
            for i in range(self.NUM_RESERVE_BOARD_CARDS):
                row, col = divmod(i, 4)
                card = self.board.market[row][col]
                reserve_board_card_mask[i] = card is not None # A card on the board can be reserved if it is not None

        reserve_deck_card_mask = np.zeros(self.NUM_RESERVE_DECK_CARDS, dtype=bool)
        if len(player.reserved_cards) < 3 and np.sum(player.tokenList) < 10:
            for tier in range(self.NUM_RESERVE_DECK_CARDS):
                if self.board.counters[tier] < self.board.TIER_SIZE[tier]:
                    reserve_deck_card_mask[tier] = True
                else:
                    reserve_deck_card_mask[tier] = False

        buy_reserve_mask = np.zeros(self.NUM_BUY_RESERVED, dtype=bool)
        for i in range(len(player.reserved_cards)):
            reserved_card = player.reserved_cards[i]
            buy_reserve_mask[i] = can_buy_card(player, reserved_card)

        mask[:len(self.all_token_actions)] = token_mask
        mask[len(self.all_token_actions): len(self.all_token_actions) + self.NUM_BUY_CARDS] = buy_card_mask
        start_idx = len(self.all_token_actions) + self.NUM_BUY_CARDS
        mask[start_idx: start_idx + self.NUM_RESERVE_BOARD_CARDS] = reserve_board_card_mask
        start_idx += self.NUM_RESERVE_BOARD_CARDS
        mask[start_idx: start_idx + self.NUM_RESERVE_DECK_CARDS] = reserve_deck_card_mask
        start_idx += self.NUM_RESERVE_DECK_CARDS
        mask[start_idx: start_idx + self.NUM_BUY_RESERVED] = buy_reserve_mask

        return mask

    def decode_action(self, action):
        num_token_actions = len(self.all_token_actions)

        buy_card_start = num_token_actions
        buy_card_end = buy_card_start + self.NUM_BUY_CARDS

        reserve_board_card_start = buy_card_end
        reserve_board_card_end = reserve_board_card_start+ self.NUM_RESERVE_BOARD_CARDS

        reserve_deck_card_start = reserve_board_card_end
        reserve_deck_card_end = reserve_deck_card_start + self.NUM_RESERVE_DECK_CARDS

        buy_reserve_start = reserve_deck_card_end
        buy_reserve_end = buy_reserve_start + self.NUM_BUY_RESERVED

        if 0 <= action < num_token_actions :
            return ("take_token",
                    action)

        elif buy_card_start <= action < buy_card_end:
            return ("buy_card",
                    action - buy_card_start)

        elif reserve_board_card_start <= action < reserve_board_card_end:
            return ("reserve_board_card",
                    action - reserve_board_card_start)

        elif reserve_deck_card_start <= action < reserve_deck_card_end:
            return ("reserve_deck_card",
                    action - reserve_deck_card_start)

        elif buy_reserve_start <= action < buy_reserve_end :
            return ("buy_reserved_card",
                    action - buy_reserve_start)
        else:
            raise ValueError(f"Invalid action: {action}")

    def take_action(self, player:Player, action:int)->None:
        action_type, action_param = self.decode_action(action)

        if action_type == "take_token":
            token_action_id = action_param
            take, discard, final = self.all_token_actions[token_action_id]
            player.take_tokens(take, discard)

        elif action_type == "buy_card":
            buy_card_idx = action_param
            tier, index = divmod(buy_card_idx, 4)
            player.buy_board_card(tier, index)

        elif action_type == "reserve_board_card":
            reserve_board_card_idx = action_param
            tier, index = divmod(reserve_board_card_idx, 4)
            player.reserve_board_card(tier, index)

        elif action_type == "reserve_deck_card":
            reserve_deck_card_idx = action_param
            player.reserve_deck_card(reserve_deck_card_idx)

        elif action_type == "buy_reserved_card":
            buy_reserve_card_idx = action_param
            player.buy_reserved_card(buy_reserve_card_idx)

        else:
            raise ValueError(f"Invalid action: {action}")

    def check_game_terminates(self)->bool:
        """
        :return: A boolean indicating if the game is terminated.
        """
        return self.is_last_round and self.current_player_index == 0

    def check_agent_wins(self)->bool:
        """
        :return: A boolean indicating if the agent wins.
        """
        final_scores = [p.score for p in self.players]
        return self.players[self.agent_player_index].score == max(final_scores)

    def step(self, action):
        action = int(action)
        info = {}
        reward = 0.0
        done = False

        if self.check_game_terminates():
            if self.check_agent_wins():
                reward = 15.0
            else:
                reward = -5.0

            done = True

            return self.get_obs(), reward, done, False, info

        last_obs = self.get_obs()

        player = self.players[self.agent_player_index]
        self.take_action(player, action)

        if player.score >= 15:
            self.is_last_round = True

        self.current_player_index = (self.current_player_index + 1) % self.num_players
        self.turn_count += 1

        if self.check_game_terminates():
            if self.check_agent_wins():
                reward = 15.0
            else:
                reward = -5.0

            done = True

            return self.get_obs(), reward, done, False, info

        opponent_obs = self.get_obs()
        opponent_action = self.opponent_agent.act(opponent_obs)
        opponent = self.players[(self.agent_player_index + 1) % self.num_players]
        self.take_action(opponent, opponent_action)
        if opponent.score >= 15:
            self.is_last_round = True

        self.current_player_index = (self.current_player_index + 1) % self.num_players
        self.turn_count += 1

        reward += self.calculate_intermediate_reward(player, last_obs)
        reward -= self.calculate_intermediate_punishment(player, action)
        return self.get_obs(), reward, done, False, info

    def calculate_intermediate_reward(self, player: Player, last_obs: dict) -> float:
        reward = 0.0

        # Get previous and current scores
        opponent = self.players[(self.current_player_index + 1) % self.num_players]

        # Card advantage
        own_card_count = sum(player.purchased_cards)
        opponent_card_count = sum(opponent.purchased_cards)
        reward += 0.05 * (own_card_count - opponent_card_count)

        # Noble progress advantage
        own_progress = sum(
            np.minimum(player.purchased_cards, noble.cost).sum() / noble.cost.sum()
            for noble in self.board.nobles
        )
        opponent_progress = sum(
            np.minimum(opponent.purchased_cards, noble.cost).sum() / noble.cost.sum()
            for noble in self.board.nobles
        )
        reward += 0.2 * (own_progress - opponent_progress)

        # Score advantage
        reward += 0.5 * (player.score - opponent.score)

        return reward

    def calculate_intermediate_punishment(self, player: Player, action: int)->float:
        punishment = 0.0
        action_type, action_param = self.decode_action(action)
        if action_type == "take_token":
            take, discard, final = self.all_token_actions[action_param]
            if np.sum(discard) > 0:
                punishment += 0.5

        return punishment

    def get_obs(self):
        current_player = self.players[self.current_player_index]

        self_tokens = np.array(current_player.tokenList, dtype=np.int32)
        self_purchased_cards = np.array(current_player.purchased_cards, dtype=np.int32)
        self_reserved_cards = np.zeros((3, 8), dtype=np.int32)
        self_score = np.array([current_player.score], dtype=np.int32)

        for j, card in enumerate(current_player.reserved_cards[:3]):
            self_reserved_cards[j] = np.array([card.tier, card.value, card.card_type] + list(card.cost), dtype=np.int32)

        opponents_tokens = []
        opponents_purchased_cards = []
        opponents_reserved_cards = []
        opponents_scores = []

        for i, player in enumerate(self.players):
            if i == self.current_player_index:
                continue

            opponents_tokens.extend(player.tokenList)
            opponents_purchased_cards.extend(player.purchased_cards)

            opponent_reserved = np.zeros((3, 1), dtype=np.int32)
            for j, card in enumerate(player.reserved_cards[:3]):
                opponent_reserved[j] = card is not None

            opponents_reserved_cards.extend(opponent_reserved.flatten())
            opponents_scores.append(player.score)

        opponents_tokens = np.array(opponents_tokens, dtype=np.int32)
        opponents_purchased_cards = np.array(opponents_purchased_cards, dtype=np.int32)
        opponents_reserved_cards = np.array(opponents_reserved_cards, dtype=np.int32)
        opponents_scores = np.array(opponents_scores, dtype=np.int32)

        board_cards_array = np.zeros((12, 8), dtype=np.int32)
        faceup_cards = [c for tier in self.board.market for c in tier if c is not None]

        for i, card in enumerate(faceup_cards[:12]):
            board_cards_array[i] = np.array([card.tier, card.value, card.card_type] + list(card.cost), dtype=np.int32)

        nobles_array = np.zeros((self.num_nobles, 5), dtype=np.int32)
        for i, noble in enumerate(self.board.nobles[:self.num_nobles]):
            nobles_array[i] = np.array(list(noble.cost), dtype=np.int32)

        return {
            "self_tokens": self_tokens,
            "self_purchased_cards": self_purchased_cards,
            "self_reserved_cards": self_reserved_cards,
            "self_score": self_score,

            "opponents_tokens": opponents_tokens,
            "opponents_purchased_cards": opponents_purchased_cards,
            "opponents_reserved_cards": opponents_reserved_cards,
            "opponents_scores": opponents_scores,

            "board_tokens": np.array(self.board.tokens, dtype=np.int32),
            "board_cards": board_cards_array,
            "nobles": nobles_array,
        }

    def render(self, mode="human"):
        print(f"===Agent Index: {self.agent_player_index}    Opponent Index: {(self.agent_player_index+1) % self.num_players}===")
        print(f"=== Current Player: {self.current_player_index} (Turn {self.turn_count}) ===")

        print("Board Tokens (Green/White/Blue/Black/Red/Gold):", self.board.tokens)

        for tier_idx, tier_cards in enumerate(self.board.market, start=1):
            print(f"--- Tier {tier_idx} ---")
            for idx, card in enumerate(tier_cards):
                if card is None:
                    print(f"   [Slot {idx}] Empty")
                else:
                    print(f"   [Slot {idx}] Value={card.value}, Type={card.card_type}, Cost={card.cost}")

        for pid, player in enumerate(self.players):
            print(f"--- Player {pid} ---")
            print("  Tokens:", player.tokenList, "(Green/White/Blue/Black/Red/Gold)")
            print("  Purchased Cards:", player.purchased_cards, "(Discount count for G/W/B/K/R)")
            print("  Score:", player.score)
            if player.reserved_cards:
                print("  Reserved Cards:")
                for r_idx, c in enumerate(player.reserved_cards):
                    print(f"    #{r_idx}: Tier={c.tier}, Value={c.value}, Type={c.card_type}, Cost={c.cost}")
            else:
                print("  Reserved Cards: None")
            if player.nobleList:
                print("  Nobles Owned:", len(player.nobleList))
            else:
                print("  Nobles Owned: None")

        if self.board.nobles:
            print("--- Nobles on Board ---")
            for idx, noble in enumerate(self.board.nobles):
                cost_str = " / ".join(f"{color}x{noble.cost[i]}" for i, color in enumerate(["G", "W", "B", "K", "R"]))
                print(f"   Noble {idx}: Requirements -> {cost_str}")
        else:
            print("No nobles left on the board.")

        print("\n--- Decks Remaining ---")
        total_points = {1: 0, 2: 0, 3: 0}
        for tier_idx in range(1, 4):
            deck_size = len(self.board.cards[tier_idx - 1])
            used = self.board.counters[tier_idx - 1]
            remaining = deck_size - used
            print(f" Tier {tier_idx}: {remaining} cards left in the deck.")

            total_points[tier_idx] = sum(card.value for card in self.board.cards[tier_idx - 1][used:])
            print(f" Tier {tier_idx} Remaining Total Points: {total_points[tier_idx]}")

        print("============================================\n")