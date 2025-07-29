import itertools
import numpy as np
from entity.player import Player
from entity.card import Card

import numpy as np
import itertools


def enumerate_take_actions():
    """
    Enumerate all possible combinations of tokens that a player can take.

    In Splendor, players can take gems in different ways:
    1. Take one gem of a single color.
    2. Take one gem each of two different colors.
    3. Take two gems of the same color (if available).
    4. Take one gem each of three different colors.

    The function generates all these possible take actions and returns them as a list of NumPy arrays,
    where each array represents an action with five slots (one for each gem color).
    """

    take_actions = []

    # Case 1: Taking only one gem of a single color
    for i in range(5):
        action = np.zeros(5, dtype=int)  # Initialize action array with zeros
        action[i] = 1  # Assign one gem to the corresponding color
        take_actions.append(action)

    # Case 2: Taking one gem each of two different colors
    for combo in itertools.combinations(range(5), 2):  # Get all 2-color combinations
        action = np.zeros(5, dtype=int)
        for i in combo:
            action[i] = 1  # Assign one gem to each selected color
        take_actions.append(action)

    # Case 3: Taking two gems of the same color
    for i in range(5):
        action = np.zeros(5, dtype=int)
        action[i] = 2  # Assign two gems to the selected color
        take_actions.append(action)

    # Case 4: Taking one gem each of three different colors
    for combo in itertools.combinations(range(5), 3):  # Get all 3-color combinations
        action = np.zeros(5, dtype=int)
        for i in combo:
            action[i] = 1  # Assign one gem to each selected color
        take_actions.append(action)

    return take_actions


def enumerate_discard_combinations_for_k(k):
    results = []

    def rec(i, current, remaining):
        if i == 5:
            if remaining == 0:
                results.append(np.array(current))
            return
        for x in range(remaining + 1):
            rec(i + 1, current + [x], remaining - x)

    rec(0, [], k)
    return results


def enumerate_all_token_actions():
    all_actions = []
    take_actions = enumerate_take_actions()
    discard_actions = []
    for k in range(4):
        discard_actions.extend(enumerate_discard_combinations_for_k(k))

    for take in take_actions:
        for discard in discard_actions:
            final = take - discard
            all_actions.append((take, discard, final))
    return all_actions


def get_token_action_mask(
        all_token_actions,
        baseline: np.ndarray,
        supply: np.ndarray
):
    """
    Rule：
      1) final cannot be negative
      2) if take[color] == 2, then supply[color] >= 4
      3) if final.sum() <= 10, then discard.sum() == 0
      4) if final.sum() > 10, then discard.sum() == final.sum() - 10
    """
    mask = np.zeros(len(all_token_actions), dtype=int)

    for i, (take, discard, final) in enumerate(all_token_actions):
        if is_valid_action(take, discard, final, baseline, supply):
            mask[i] = 1

    return mask


def is_valid_action(
        take: np.ndarray,
        discard: np.ndarray,
        final: np.ndarray,
        baseline: np.ndarray,
        supply: np.ndarray
) -> bool:
    """
    Invalid Cases:
    1.Taking gems when there's not enough on the board
    2.Not taking 3 different gems/2 of a same type when there's enough on the board
    3.Returning gems that results in negative net total
    4.Returning gems when gems at hand is not more than 10
    5.Returning not enough gems when gems at hand is more than 10
    :param take: an array of numbers of tokens to take from each type
    :param discard: an array of numbers of tokens to return from each type
    :param final: take - discard
    :param baseline: tokenList of the player
    :param supply: board.tokens
    :return: a boolean indicating if the action is valid
    """
    # # Case 1: Taking gems when there's not enough on the board
    if not check_supply(take, supply):
        return False

    # # Case 2: Returning gems that results in negative net total
    padded_final = np.append(final, 0)
    net_total = baseline + padded_final
    if np.any(net_total < 0):
        return False

    # # Case 3: Returning gems when gems at hand is not more than 10
    if np.sum(net_total) < 10 and np.sum(discard) > 0:
        return False

    # # Case 4: Returning not enough gems when gems at hand is more than 10
    padded_take = np.append(take, 0)
    if np.sum(baseline + padded_take ) > 10:
        required_discard = np.sum(baseline + padded_take ) - 10
        if np.sum(discard) != required_discard:
            return False

    # Case 5: Taking not enough gems when there's enough on the board
    if np.sum(supply > 0) < 3:
        # Only two valid actions:
        # 1. Take one gem from each remaining type
        # 2. Take two gems from any remaining type
        num_remaining_gems = np.count_nonzero(supply)
        if not (np.count_nonzero(take) == num_remaining_gems or np.any(take == 2)):
            return False

    return True


def check_supply(take: np.ndarray, supply: np.ndarray) -> bool:
    for color in range(5):
        if take[color] == 2 and supply[color] < 4:
            return False
        elif take[color] == 1 and supply[color] < 1:
            return False
    return True


def can_buy_card(player: Player, card: Card) -> bool:
    """
    Checks if the player has enough resources (tokens + discounts) to buy a card.
    This function does NOT modify player state.

    :param player: The Player object.
    :param card: The Card object to check.
    :return: True if the player can afford the card, otherwise False.
    """
    if card is None:
        return False

    cost = card.cost.copy()
    discount = player.purchased_cards  # Discounts from owned cards
    net_cost = np.maximum(cost - discount, 0)  # Minimum required tokens after discounts

    gold_available = player.tokenList[5]  # Number of gold tokens available

    for color in range(5):  # Check each color
        needed = net_cost[color]
        have = player.tokenList[color]
        if have < needed:
            gold_needed = needed - have
            if gold_needed > gold_available:
                return False  # Not enough tokens to buy this card
            gold_available -= gold_needed  # Use gold tokens if necessary

    return True  # The player can afford this card


def can_buy_card_from_array(player_tokens: np.ndarray, player_discounts: np.ndarray, card_cost: np.ndarray) -> bool:
    """
    Checks if a player can buy a card given token counts, discounts, and card cost.

    :param player_tokens: np.array of shape (6,), representing the player's token counts.
                          Index 0-4: normal tokens (G, W, B, K, R), Index 5: gold tokens.
    :param player_discounts: np.array of shape (5,), representing the player's discounts from purchased cards.
    :param card_cost: np.array of shape (5,), representing the cost of the card in different colors.
    :return: True if the player can afford the card, otherwise False.
    """

    assert player_tokens.shape == (6,), f"Expected player_tokens shape (6,), got {player_tokens.shape}"
    assert player_discounts.shape == (5,), f"Expected player_discounts shape (5,), got {player_discounts.shape}"
    assert card_cost.shape == (5,), f"Expected card_cost shape (5,), got {card_cost.shape}"


    net_cost = np.maximum(card_cost - player_discounts, 0)  # shape (5,)


    gold_available = player_tokens[5]  # shape (1,)


    for color in range(5):
        needed = net_cost[color]
        have = player_tokens[color]

        if have < needed:
            gold_needed = needed - have
            if gold_needed > gold_available:
                return False
            gold_available -= gold_needed

    return True


def compute_valid_actions_from_obs(obs):
    all_token_actions = enumerate_all_token_actions()
    NUM_BUY_CARDS = 12
    NUM_RESERVE_CARDS = 15
    NUM_BUY_RESERVED = 3

    total_actions = len(all_token_actions) + NUM_BUY_CARDS + NUM_RESERVE_CARDS + NUM_BUY_RESERVED
    mask = np.zeros(total_actions, dtype=bool)

    self_tokens = obs["self_tokens"]        # shape (6,)
    board_tokens = obs["board_tokens"]      # shape (6,)

    token_mask = np.array([is_valid_action(action[0], action[1], action[2],
                                           self_tokens[:6], board_tokens[:5]) for
                           action in all_token_actions])

    buy_card_mask = np.zeros(NUM_BUY_CARDS, dtype=bool)
    board_cards = obs["board_cards"]
    self_purchased_cards = obs["self_purchased_cards"]
    for i in range(NUM_BUY_CARDS):
        cost = board_cards[i][3:]
        buy_card_mask[i] = can_buy_card_from_array(self_tokens, self_purchased_cards, cost)

    reserve_card_mask = np.zeros(NUM_RESERVE_CARDS, dtype=bool)
    for i in range(NUM_RESERVE_CARDS):
        if i < NUM_RESERVE_CARDS - 3:
            #TODO:Add logic to check what should happen for empty cards
            reserve_card_mask[i] = False
        else:
            #TODO:need to check remaining check for reserving hidden cards
            reserve_card_mask[i] = False

    buy_reserve_mask = np.zeros(NUM_BUY_RESERVED, dtype=bool)
    self_reserved_cards = obs["self_reserved_cards"]
    for i in range(len(self_reserved_cards)):
        cost = self_reserved_cards[i][3:]
        buy_reserve_mask[i] = can_buy_card_from_array(self_tokens, self_purchased_cards, cost)

    mask[:len(all_token_actions)] = token_mask
    mask[len(all_token_actions): len(all_token_actions) + NUM_BUY_CARDS] = buy_card_mask
    start_idx = len(all_token_actions) + NUM_BUY_CARDS
    mask[start_idx: start_idx + NUM_RESERVE_CARDS] = reserve_card_mask
    start_idx += NUM_RESERVE_CARDS
    mask[start_idx: start_idx + NUM_BUY_RESERVED] = buy_reserve_mask

    return mask


if __name__ == '__main__':
    for action in enumerate_all_token_actions():
        if is_valid_action(action[0], action[1], action[2], np.array([2, 2, 2, 2, 2, 0]), np.array([0, 0, 4, 0, 2])):
            print(action)