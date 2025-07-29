import numpy as np


class Card:
    def __init__(self, tier:int, value:int, card_type:int, cost:np.ndarray):
        self.tier = tier
        self.value = value
        self.card_type = card_type # the discount type
        self.cost = cost