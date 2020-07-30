from unityagents import UnityEnvironment
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from agent import Agent

APPLY_OU_NOISE = True # apply QUNoise for action selection

train_mode = False

env = UnityEnvironment(file_name='../data/Tennis_Windows_x86_64/Tennis.exe')

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=train_mode)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

agent = Agent(state_size=state_size, action_size=action_size, random_seed=0)

def ddpg(n_episodes=100, max_t=500, print_every=100):
    if not train_mode:
        agent.critic_local.load_state_dict(torch.load('../checkpoint_critic.pth'))
        agent.actor_local.load_state_dict(torch.load('../checkpoint_actor.pth'))

    scores_deque = deque(maxlen=print_every)
    episodeMaxMean = 0.0
    scores = []
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=train_mode)[brain_name]
        state = env_info.vector_observations
        agent.reset()
        score = np.array([0.0]*num_agents)
        for t in range(max_t):
            action = agent.act(state, add_noise = APPLY_OU_NOISE)
            env_info = env.step(action)[brain_name]
            next_state, reward, done = env_info.vector_observations, env_info.rewards, env_info.local_done
            if train_mode:
                agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward
            if np.any(done):
                break
        scores_deque.append(np.max(score))
        scores.append(score)
        print("\repisode scores: ", score)
        print("\raverage episode score: ", np.mean(score))
        print("\rmax agents episode score: ", np.max(score))
        print('\rEpisode {}\tLast 100 average Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        if train_mode and np.mean(score) > episodeMaxMean:
            episodeMaxMean = np.mean(score)
            torch.save(agent.actor_local.state_dict(), '../checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), '../checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    return scores


scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()