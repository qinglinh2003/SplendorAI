from random import Random

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from agents.ppo_agent import PPOAgent
from agents.random_agent import RandomAgent
from env.splendor_env import SplendorEnv
from agents.rule_based_agent import RuleBasedAgent


model = MaskablePPO.load("../agents/against_rule/checkpoints/checkpoint_step_400000.zip")
model2 = MaskablePPO.load("../agents/checkpoints/checkpoint_step_400000.zip")
ppo_agent = PPOAgent(model2)
num_simulations = 1

total_scores = []
total_turns = []

print("🔹 Start Splendor Simulation")
print("=" * 50)
rule_agent= RuleBasedAgent()
env = SplendorEnv(num_players=2)
env = ActionMasker(env, lambda env: env.action_mask())
real_env = env.env


for i in range(num_simulations):

    turn = 0
    player_scores = [0] * 2
    obs, _ = real_env.reset()

    while True:
        if real_env.agent_player_index == 0:
            turn += 2
            mask = real_env.action_mask()
            obs = real_env.get_obs()
            action, _ = model.predict(obs, action_masks=mask)
            agent = real_env.players[real_env.agent_player_index]
            real_env.take_action(agent, action)
            real_env.turn_count += 1
            real_env.current_player_index = (real_env.current_player_index + 1) % 2

            obs = real_env.get_obs()
            opponent_action = rule_agent.act(obs)
            opponent = real_env.players[(real_env.agent_player_index+1)%2]
            real_env.take_action(opponent, opponent_action)
            real_env.turn_count += 1
            real_env.current_player_index = (real_env.current_player_index + 1) % 2

            real_env.render()
            if agent.score >= 15 or opponent.score >= 15:
                break
        else:
            turn += 2

            obs = real_env.get_obs()
            opponent_action = rule_agent.act(obs)
            opponent = real_env.players[(real_env.agent_player_index + 1) % 2]
            real_env.take_action(opponent, opponent_action)
            real_env.turn_count += 1
            real_env.current_player_index = (real_env.current_player_index + 1) % 2

            mask = real_env.action_mask()
            obs = real_env.get_obs()
            action, _ = model.predict(obs, action_masks=mask)
            agent = real_env.players[real_env.agent_player_index]
            real_env.take_action(agent, action)
            real_env.turn_count += 1
            real_env.current_player_index = (real_env.current_player_index + 1) % 2

            real_env.render()
            if agent.score >= 15 or opponent.score >= 15:
                break


    total_turns.append(turn)

    if (i + 1) % 100 == 0:
        print(f"Completed {i + 1}/{num_simulations} simulations...")


# avg_scores = np.mean(total_scores, axis=0)
avg_turns = np.mean(total_turns)

print("🎉 Simulation Complete!")
# print(f"Average Player Scores: {avg_scores}")
print(total_scores)
print(f"Average Turns per Game: {avg_turns:.2f}")
