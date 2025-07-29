import numpy as np


class Noble:
    def __init__(self, cost:np.ndarray):
        self.cost = cost #shape = (5, ), [Green, White, Blue, Black, Red]