from gymnasium.wrappers import FlattenObservation

from agents.ppo_agent import PPOAgent
from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent

from env.splendor_env  import SplendorEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

model = MaskablePPO.load("../agents/against_rule/checkpoints/checkpoint_step_350000.zip")

total_scores = []
total_turns = []

print("🔹 Start Splendor Simulation")
print("=" * 50)

env = SplendorEnv(num_players=2)
env = ActionMasker(env, lambda env: env.action_mask())

real_env = env.env

def play_match(env, agent1, agent2):
    obs, _ = env.reset()
    done = False
    agents = [agent1, agent2]
    turn = 0

    while not done:
        action = agents[turn].act(obs)
        obs, reward, done, info, _ = env.step(action)
        env.render()
        if done:
            break

        turn = 1 - turn

    print("Game Over. Final State:", info)

sb3_agent = PPOAgent(model)
random_agent = RandomAgent()
rule_agent = RuleBasedAgent()


play_match(real_env, sb3_agent,rule_agent)




