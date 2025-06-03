import numpy as np
import random

# Define the environment
class GridWorld:
    def __init__(self):
        self.grid_size = 4  # 4x4 grid
        self.goal_state = (3, 3)  # Goal position
        self.start_state = (0, 0)  # Starting position
        self.current_state = self.start_state
        self.actions = ['up', 'down', 'left', 'right']  # Possible actions
    
    def reset(self):
        self.current_state = self.start_state
        return self.current_state
    
    def step(self, action):
        row, col = self.current_state
        
        # Update state based on action
        if action == 'up':
            next_state = (max(row - 1, 0), col)
        elif action == 'down':
            next_state = (min(row + 1, self.grid_size - 1), col)
        elif action == 'left':
            next_state = (row, max(col - 1, 0))
        elif action == 'right':
            next_state = (row, min(col + 1, self.grid_size - 1))
        
        # Check if the new state is out of bounds or invalid (in this case, we clip it, but we penalize out-of-bound moves)
        reward = -0.1  # Small penalty for each move to encourage shorter paths
        done = False
        
        if next_state == self.goal_state:
            reward = 1.0  # Reward for reaching the goal
            done = True
        elif next_state == self.current_state:  # If the move didn't change the state (e.g., hit a wall)
            reward = -0.1  # Penalty for invalid move
        
        self.current_state = next_state
        return next_state, reward, done

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, max_exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay_rate=0.01):
        self.q_table = np.zeros((num_states[0], num_states[1], num_actions))  # Q-table: (rows, columns, actions)
        self.learning_rate = learning_rate  # Alpha: how much to learn from new information
        self.discount_factor = discount_factor  # Gamma: how much to discount future rewards
        self.exploration_rate = exploration_rate  # Epsilon: for epsilon-greedy policy
        self.max_exploration_rate = max_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate  # Decay rate for epsilon
    
    def select_action(self, state):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(0, 3)  # Explore: random action (mapped to 'up', 'down', 'left', 'right')
        else:
            return np.argmax(self.q_table[state[0], state[1], :])  # Exploit: best action based on Q-table
    
    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1], :])
        current_q = self.q_table[state[0], state[1], action]
        new_q = reward + self.discount_factor * self.q_table[next_state[0], next_state[1], best_next_action]
        self.q_table[state[0], state[1], action] += self.learning_rate * (new_q - current_q)
    
    def decay_exploration(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate - self.exploration_decay_rate)

# Main training loop
def train_agent(num_episodes=1000):
    env = GridWorld()
    agent = QLearningAgent(num_states=(4, 4), num_actions=4)  # 4 rows, 4 columns, 4 actions
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action_idx = agent.select_action(state)  # Get action index
            next_state, reward, done = env.step(env.actions[action_idx])  # Take action in environment
            agent.update_q_table(state, action_idx, reward, next_state)  # Update Q-table
            state = next_state
            total_reward += reward
        
        agent.decay_exploration()  # Decay exploration rate after each episode
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Episode: {episode + 1}, Total Reward: {total_reward:.2f}, Exploration Rate: {agent.exploration_rate:.2f}")
    
    return agent

# Test the trained agent
def test_agent(agent, env, num_test_episodes=10):
    total_rewards = []
    for episode in range(num_test_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action_idx = np.argmax(agent.q_table[state[0], state[1], :])  # Always choose the best action (greedy)
            next_state, reward, done = env.step(env.actions[action_idx])
            state = next_state
            total_reward += reward
        total_rewards.append(total_reward)
        print(f"Test Episode {episode + 1}: Total Reward: {total_reward:.2f}")
    
    avg_reward = np.mean(total_rewards)
    print(f"Average Reward over {num_test_episodes} episodes: {avg_reward:.2f}")


# Running the training 
if __name__ == "main":
    trained_agent = train_agent(num_episodes = 1000) # Train for 1000
    env = GridWorld() # load the tilemap grid
    test_agent(trained_agent, env) #test the agent 