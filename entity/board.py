import numpy as np
import pandas as pd
from entity.card import Card
from entity.noble import Noble

class Board:
    """
    Splendor Board Class.

    This class represents the main game board for Splendor.
    It manages:
    - The three tiers of development cards (face-up and deck).
    - The token pool available for players to take.
    - The noble tiles that players can earn.
    - The updating and refreshing of face-up cards when a card is purchased.

    Attributes:
    -----------
    TIER_SIZE : list of int
        A list representing the number of cards in each tier deck.
        - Tier 1 (lowest tier) contains 40 cards.
        - Tier 2 (mid-tier) contains 30 cards.
        - Tier 3 (highest tier) contains 20 cards.
    """
    TIER_SIZE = [40, 30, 20]
    TOKENS = [4, 4, 4, 4, 4, 5] #[Green, White, Blue, Black, Red, Gold]
    def __init__(self, num_players=2):
        self.num_players = num_players
        self.tokens = np.array(Board.TOKENS, dtype=int)
        self.cards = [[], [], []]
        self.load_cards()
        self.nobles = []
        self.load_nobles()
        self.market = [tier[:4] for tier in self.cards]
        self.counters = [4, 4, 4] # Representing the top of each tier. Init to 4 since 4 cards for each tier on the board

    def load_cards(self)-> None:
        """
        Loads cards from CSV file.
        """
        card_df = pd.read_csv("../data/cards.csv")
        for _, row in card_df.iterrows():
            tier = int(row['tier'])
            card_type = int(row['type'])
            value = int(row['value'])
            cost = np.array([row['green'], row['white'], row['blue'], row['black'], row['red']], dtype=int)
            self.cards[tier - 1].append(Card(tier, value, card_type, cost))

    def load_nobles(self):
        """
        Loads nobles from CSV file.
        """
        num_nobles = 5 - (4 - self.num_players)
        noble_df = pd.read_csv("../data/nobles.csv").sample(n=num_nobles)
        for _, row in noble_df.iterrows():
            cost = np.array([row['green'], row['white'], row['blue'], row['black'], row['red']], dtype=int)
            self.nobles.append(Noble(cost))

    def update_cards(self, tier:int, index:int)->None:
        """
        :param tier: Face Up card tier.
        :param index: Face Up card index within a tier.
        """
        # If run out of cards in current board
        if self.counters[tier] >= len(self.cards[tier]):
            self.market[tier][index] = None
        #Otherwise, update current board by drawing a new card and update the pointer to the top of the tier
        else:
            self.market[tier][index] = self.cards[tier][self.counters[tier]]
            self.counters[tier] += 1
