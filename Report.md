# Learning Algorithm
Agent uses deep deterministic policy gradient algorithm [DDPG](https://arxiv.org/abs/1509.02971). DDPG is an actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces. [DDPG-pendulum project](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) was used as a reference algorithm implementation adopted to the "Tennis" environment, with minor modifications.

### Hyperparameters
The algorithm has the following set of hyperparameters:
| Parameter | Value |
| ----------- | ----------- |
|BUFFER_SIZE = int(1e5) | replay buffer size
|BATCH_SIZE = 256  | minibatch size
|GAMMA = 0.99  | discount factor
|TAU = 1e-3  | for soft update of target parameters
|LR_ACTOR = 1e-4  | learning rate of the actor
|LR_CRITIC = 1e-3  | learning rate of the critic
|WEIGHT_DECAY = 0  | L2 weight decay
|APPLY_OU_NOISE = True | apply QUNoise for action selection

### Model Configuration
#### Actor
| Layer | In | Out
| ----------- | ----------- |----------- |
| Linear | state_size | 400
| BatchNorm1d | 400
| ReLU | 400 | 400
| Linear | 400 | 300
| ReLU | 300 | 300
| Linear | 300 | action_size
| tanh | action_size | action_size

#### Critic
| Layer | In | Out
| ----------- | ----------- |----------- |
| Linear | state_size | 400
| ReLU | 400 | 400
| Linear | 400 + action_size | 300
| ReLU | 300 | 300
| Linear | 300 | 1

### Additional extensions
#### Experience replay buffer
Agents use shared experience replay buffer.

#### Gradient clipping
Gradient clipping technique helps to deal with the irregular loss landscape of the model.

#### Distributed distributional deterministic policy gradients
Both agents store and sample experience from the same replay buffer, share actor and critic networks.

#### Ornstein–Uhlenbeck noise
Ornstein–Uhlenbeck process applied to the algorithm for generating temporally correlated exploration.

# Plot of Rewards
Peak of the agents score is reached around the 1700th episode with the average score around 2.7 points. Maximum score is limited by the amount of timesteps in each episode (agents were trained with 1000 timesteps per episode).
![Scores:](/images/learning.PNG)
Running the environment with an already trained network shows an average score of 1.31 points over 100 episodes. As in the case from the training plot, agents' score is limited by the amount of timesteps (100 timesteps per episode in this case). As is visible on the plont agents managed to finish only one game, since in most of the cases they cooperated for reaching maximum score instead of winning the game.
![Scores:](/images/performance.png)

# Ideas for Future Work

### Make training faster by tuning hyperparameters. Comparing performance to other algorithms, like PPO. Adding prioritized experience replay.
### Training competitive agents instead of cooperative, so agents are getting score for winning a game instead of passing the ball on the other side.
