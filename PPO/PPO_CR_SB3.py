import gymnasium as gym
from stable_baselines3 import PPO

# env = gym.make('CarRacing-v2')  # v2 is the updated version
# model = PPO('CnnPolicy', env, verbose=1)
# model.learn(total_timesteps=1_000_000)
# model.save("ppo_car_racing")

model = PPO.load("ppo_car_racing")

# Create the environment for evaluation
env = gym.make('CarRacing-v2',render_mode="human")  # or 'CarRacing-v2', depending on your version

num_episodes = 10
for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        env.render()  # displays the environment window
    print(f"Episode {episode+1}: Total Reward = {total_reward}")
env.close()