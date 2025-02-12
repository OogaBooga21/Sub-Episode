import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CarRacing-v2", render_mode="human")
agent = PPO(env, config={'lr': 3e-4, 'batch_size': 128})

state, _ = env.reset()
episode_reward = 0

while True:
    action, log_prob, value = agent.act(state)
    next_state, reward, done, truncated, _ = env.step(action)
    agent.remember(state, action, reward, done, log_prob, value)
    
    state = next_state
    episode_reward += reward
    
    if done or truncated:
        agent.update()  # Manually trigger update
        print(f"Episode reward: {episode_reward}")
        episode_reward = 0
        state, _ = env.reset()