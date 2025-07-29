import numpy as np
from entity.card import Card


class Player:
    MAX_TOKENS = 10
    MAX_RESERVED_CARDS =3

    def __init__(self, board):
        self.board = board
        self.tokenList = np.zeros(6, dtype=int) #[Green, White, Blue, Black, Red, Gold]
        self.purchased_cards = np.zeros(5, dtype=int) #[Green, White, Blue, Black, Red]
        self.nobleList = []
        self.reserved_cards = []
        self.score = 0

    def take_tokens(self, take:np.ndarray, discard:np.ndarray)->bool:
        """
        Action to take tokens. Discard tokens to MAX_TOKENS if tokens at hand exceeds MAX_TOKENS.
        :param take:
            An array representing the tokens of each type of gem to take.
            A valid choice can be either 3 different types of gems or 2 of the same type (if number of tokens + card of this type >= 4).
        :param discard:
            An array representing the tokens of each type of gem to discard.
        :return:A boolean indicating whether the action is successful.
        """

        if np.sum(take) > 3:
            return False
        if np.any(take > 2):
            return False
        if np.any(take > self.board.tokens[:5]):
            return False
        if np.any(take > 1):
            index = np.where(take > 1)[0][0]
            if self.board.tokens[index] < 4:
                return False

        self.tokenList[:5] += take
        self.board.tokens[:5] -= take

        if np.sum(self.tokenList[:5]) > self.MAX_TOKENS:
            return self.return_tokens(discard)

        return True

    def return_tokens(self, tokens:np.ndarray)->bool:
        """
        Action to return tokens.
        :param tokens: an array representing the tokens of each type of gem to return.
        :return: A boolean indicating whether the action is successful.
        """
        # Can return at most 3 gems
        if np.sum(tokens) > 3:
            return False
        # Cannot return if there's not enough gems of that type
        if np.any(tokens > self.tokenList[:5]):
            return False

        self.tokenList[:5] -= tokens
        self.board.tokens[:5] += tokens

        return True

    def buy_board_card(self,tier:int, index:int)->bool:
        """
          Action to buy card from board.
          :param tier: tier of the card to buy
          :param index: index of the card to buy
          :return: A boolean value indicating if the action is successful.
        """
        card = self.board.market[tier][index]
        if self.buy_card(card):
            self.board.update_cards(tier, index)
            return True
        else:
            return False

    def buy_card(self, card:Card)->bool:
        """
        Action to buy card.
        :param card: card to buy
        :return: A boolean value indicating if the action is successful.
        """

        if card is None: # Unsuccessful if the card is empty
            return False

        cost = card.cost.copy()
        # Each card at hand reduces the cost of its type by 1
        discount = self.purchased_cards

        net_cost = np.maximum(cost - discount, 0)

        color_paid = np.zeros(5, dtype=int)
        gold_used = 0
        gold_have = self.tokenList[5]

        for color in range(5):
            needed = net_cost[color]
            have = self.tokenList[color]
            if have >= needed: # if having enough tokens, use them first
                color_paid[color] = needed
            else: # if not having enough tokens, try to pay with golds
                gold_needed = needed - have

                if gold_needed > gold_have: # Unsuccessful if not enough gold
                    return False
                else:
                    color_paid[color] = have # Able to pay the required amount of current type by using all gems of this type plus required gold
                    gold_used += gold_needed #track gold used
                    gold_have -= gold_needed # Update gold have

        self.board.tokens[:5] += color_paid
        self.tokenList[:5] -= color_paid

        self.board.tokens[5] += gold_used
        self.tokenList[5] -= gold_used

        color = card.card_type
        self.purchased_cards[color-1] += 1 #card_type is 1-indexing
        self.score += card.value

        self._check_noble()

        return True

    def buy_reserved_card(self, index: int) -> bool:
        """
        Attempts to buy a reserved card.
        :param index: index of the card to buy
        :return: A boolean indicating if the action is successful.
        """
        if index >= len(self.reserved_cards):
            return False

        card = self.reserved_cards[index]

        if self.buy_card(card):
            self.reserved_cards.remove(card)
            return True
        else:
            return False

    def reserve_board_card(self, tier, index) -> bool:
        """
        Attempts to reserve a reserved card.
        :param tier: tier of the card to reserve
        :param index: index of the card in the tier
        :return: a boolean indicating if the action is successful.
        """
        if len(self.reserved_cards) >= Player.MAX_RESERVED_CARDS:
            return False

        card = self.board.market[tier][index]
        if card is None:
            return False

        self.reserved_cards.append(card)
        self.board.update_cards(tier, index)

        # Reserving a card earns a gold if there is one on the board
        if self.board.tokens[5] > 0:
            self.board.tokens[5] -= 1
            self.tokenList[5] += 1

        return True

    def reserve_deck_card(self, tier) -> bool:
        """
        Attempts to reserve a hidden card from the deck.
        :param tier: tier of the card to reserve
        :return: a boolean indicating if the action is successful.
        """
        if len(self.reserved_cards) >= Player.MAX_RESERVED_CARDS:
            return False

        if self.board.counters[tier] >= self.board.TIER_SIZE[tier]:
            return False

        top_card_index = self.board.counters[tier]
        card = self.board.cards[tier][top_card_index]
        if card is None:
            return False

        self.reserved_cards.append(card)

        if self.board.tokens[5] > 0:
            self.board.tokens[5] -= 1
            self.tokenList[5] += 1

        return True


    def _check_noble(self):
        """
        Checks if the player has met the requirements to acquire any noble.
        If the player meets a noble's requirement, the noble is awarded.
        """
        nobles_to_acquire = []
        for noble in self.board.nobles:
            if np.all(self.purchased_cards >= noble.cost):
                nobles_to_acquire.append(noble)

        for noble in nobles_to_acquire:
            self.nobleList.append(noble)
            self.board.nobles.remove(noble)
            self.score += 3


