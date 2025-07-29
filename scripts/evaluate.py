from env.splendor_env import SplendorEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import numpy as np

model = MaskablePPO.load("../agents/against_rule/checkpoints/checkpoint_step_400000.zip")

num_simulations = 1

total_scores = []
total_turns = []

print("🔹 Start Splendor Simulation")
print("=" * 50)

for i in range(num_simulations):
    env = SplendorEnv(num_players=2)
    env = ActionMasker(env, lambda env: env.action_mask())
    real_env = env.env

    obs, _ = real_env.reset()
    done = False
    turn = 0
    player_scores = [0] * 2

    while not done:
        turn += 1
        mask = real_env.action_mask()
        action, _ = model.predict(obs, action_masks=mask)
        print(action)
        obs, reward, done, truncated, info = real_env.step(action)
        real_env.render()
        player_scores = info.get("scores", player_scores)

        if done or truncated:
            break

    total_scores.append(player_scores)
    total_turns.append(turn)

    if (i + 1) % 100 == 0:
        print(f"Completed {i + 1}/{num_simulations} simulations...")


avg_scores = np.mean(total_scores, axis=0)
avg_turns = np.mean(total_turns)

print("🎉 Simulation Complete!")
print(f"Average Player Scores: {avg_scores}")
print(f"Average Turns per Game: {avg_turns:.2f}")