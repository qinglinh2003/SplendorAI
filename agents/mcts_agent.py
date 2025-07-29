import numpy as np
import copy

from utils.valid_action_utils import compute_valid_actions_from_obs


class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state  # The observation (game state)
        self.parent = parent
        self.children = {}  # Dictionary {action: child_node}
        self.visits = 0
        self.wins = 0
        self.untried_actions = None  # Will be set when expanding

    def get_uct_value(self, exploration_weight=1.41):
        """
        Compute the UCT (Upper Confidence Bound) value for node selection.
        """
        if self.visits == 0:
            return float('inf')  # Encourage exploration of new nodes
        exploit = self.wins / self.visits  # Win rate
        explore = exploration_weight * np.sqrt(np.log(self.parent.visits) / self.visits)
        return exploit + explore

    def best_child(self):
        """
        Select the child node with the highest UCT value.
        """
        return max(self.children.values(), key=lambda child: child.get_uct_value())

    def is_fully_expanded(self):
        """
        Returns True if all possible actions have been tried.
        """
        return len(self.untried_actions) == 0 if self.untried_actions is not None else False

    def get_random_untried_action(self):
        """
        Pick a random untried action.
        """
        return self.untried_actions.pop() if self.untried_actions else None

    def select(self):
        """
        Traverse the tree using UCT until we reach a node that can be expanded.
        Returns the best node for expansion.
        """
        current_node = self
        while current_node.is_fully_expanded() and current_node.children:
            current_node = current_node.best_child()
        return current_node

    def expand(self, env):
        """
        Expands the current node by taking an untried action and creating a child node.
        Returns the newly created child node.
        """
        if self.untried_actions is None:
            self.untried_actions = set(np.where(compute_valid_actions_from_obs(self.state))[0])  # Get available actions

        if not self.untried_actions:
            return self  # No expansion possible

        action = self.get_random_untried_action()  # Pick a random untried action

        # Clone the environment and apply the action
        env_clone = copy.deepcopy(env)
        obs, reward, terminated, truncated, _ = env_clone.step(action)

        child_node = MCTSNode(state=obs, parent=self)
        self.children[action] = child_node
        return child_node


