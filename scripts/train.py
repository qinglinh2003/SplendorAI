from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from env.splendor_env import SplendorEnv

env = SplendorEnv()
env = ActionMasker(env, lambda env: env.action_mask())

model = MaskablePPO("MultiInputPolicy", env, verbose=1,tensorboard_log="../logs")

total_timesteps = 400000
checkpoint_interval = 50000

for step in range(1, total_timesteps // checkpoint_interval + 1):
    print(f"🚀 Start {checkpoint_interval} Step ( {step} ) ...")


    model.learn(total_timesteps=checkpoint_interval)

    model.save(f"../agents/v2.0/checkpoints/checkpoint_step_{step * checkpoint_interval}")
    print(f"✅ Save to: checkpoint_step_{step * checkpoint_interval}")

model.save("../agents/v2.0/splendor_model")