# Learning Algorithm
Agent uses deep deterministic policy gradient algorithm [DDPG](https://arxiv.org/abs/1509.02971). DDPG is an actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces. [DDPG-pendulum project](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum) was used as a reference algorithm implementation adopted to the "Reacher" environment, with minor modifications.

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
|APPLY_OU_NOISE = False | apply QUNoise for action selection

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
Since the environment contains 20 simultaneos agents it was reasonable to utilize D4PG for speeding up the agents training process by sharing experience and actor/critic network weights. Probably it's the most significant addition to the algorithm since the agents showed a stable and significant improvement in the average learning score after applying D4PG. Due to the fact that every agent starts in a random environment's state, the learning process tends to generalize very well for different tasks like reaching the sphere that moves clockwise. counterclockwise or stays almost stationary. Which seems to be a problem for a single agent learning process since the environment dynamics changes only after completing the full episode.           
#### Ornstein–Uhlenbeck noise
Ornstein–Uhlenbeck process can be easily applied to the algorithm for generating temporally correlated exploration, but it's disabled by default since it doesn't add much to the agent's learning process.

# Plot of Rewards
Average agents score stabilizes after the 80th episode reaching the average reward around 36 points. 
![Scores:](/images/learning.PNG)
Running the environment with an already trained network shows an average score around 38.5 points over 200 episodes.
![Scores:](/images/agents_performance.png)
# Ideas for Future Work

### Comparing performance to other algorithms, like PPO. Adding prioritized experience replay.
### Testing the performance of the algorithm in a "reacher" Reacher environment, for example with an arm containing four joints 
