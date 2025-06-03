import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

# REINFORCE Agent
class ReinforceAgent:
    def __init__(self, input_size, hidden_size, output_size, gamma=0.99, lr=0.01):
        self.policy = PolicyNetwork(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma  # Discount factor for future rewards

    def select_action(self, state):
        state = np.array(torch.FloatTensor(state))
        #state = torch.tensor(np.array(state))
        probs = self.policy(state)
        action = torch.multinomial(probs, 1).item()  # Sample an action based on probabilities
        log_prob = torch.log(probs[action])
        return action, log_prob

    def update_policy(self, rewards, log_probs):
        # Compute discounted rewards
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.FloatTensor(discounted_rewards)

        # Normalize rewards (optional, helps with training stability)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Compute policy loss
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)  # Negative because we are doing gradient ascent
        policy_loss = torch.stack(policy_loss).sum()

        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

def train_reinforce():
    env = gym.make('CartPole-v1')
    agent = ReinforceAgent(input_size=4, hidden_size=128, output_size=2, gamma=0.99, lr=0.01)
    num_episodes = 1000
    print_every = 100

    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        done = False

        # Collect trajectory for one episode
        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        # Update policy using the collected trajectory
        agent.update_policy(rewards, log_probs)

        # Print progress
        if (episode + 1) % print_every == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {sum(rewards):.2f}")

    env.close()

if __name__ == "__main__":
    train_reinforce()