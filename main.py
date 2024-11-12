import gym
import random
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

def discretize_state(state, bins):
    discretized = []
    for i in range(len(state)):
        discretized.append(np.digitize(state[i], bins[i]))
    return tuple(discretized)

num_bins = (10, 10, 10, 10)  # Number of bins for each state variable
bins = [
    np.linspace(-2.4, 2.4, num_bins[0] + 1)[1:-1],  # Cart Position
    np.linspace(-3, 3, num_bins[1] + 1)[1:-1],       # Cart Velocity
    np.linspace(-0.418, 0.418, num_bins[2] + 1)[1:-1],  # Pole Angle
    np.linspace(-3, 3, num_bins[3] + 1)[1:-1]        # Pole Velocity At Tip
]

q_table = np.zeros(num_bins + (env.action_space.n,)) # Initialize Q-table

# Alpha is learning rate (0-1) how much new info overrides old info
alpha = 0.9
# Gamma is discount factor (0-1) how much future rewards matter
gamma = 0.95
# 1 = full exploration, 0 = no exploitation
epsilon = 1
epsilon_decay = 0.9995
min_epsilon = 0.01
episode_count = 200
max_steps_per_episode = 500
ep_rewards = []  # List to store episode rewards

for episode in range(episode_count):
    state = env.reset()
    discrete_state = discretize_state(state, bins)
    step_cnt = 0
    ep_reward = 0
    done = False

    while not done:
        env.render()
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[discrete_state])  # Exploit

        next_state, reward, done, _ = env.step(action)
        next_discrete_state = discretize_state(next_state, bins)

        if not done:
            max_future_q = np.max(q_table[next_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif step_cnt >= max_steps_per_episode: # done and reached max steps
            q_table[discrete_state + (action,)] = 0
            print("Reached max steps")

        discrete_state = next_discrete_state
        step_cnt += 1
        ep_reward += reward
        ep_rewards.append(ep_reward)

    epsilon *= epsilon_decay
    epsilon = max(min_epsilon, epsilon)

    print('Episode: {}, Step count: {}, Episode reward: {}'.format(episode, step_cnt, ep_reward))

env.close()

plt.plot(ep_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Rewards over Time')
plt.show()
