import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Tic-Tac-Toe Environment
class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros(9, dtype=int)  # 0: empty, 1: X, 2: O
        self.done = False
        self.winner = None

    def reset(self):
        self.board = np.zeros(9, dtype=int)
        self.done = False
        self.winner = None
        return self.board.copy()

    def check_winner(self):
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        for combo in win_combinations:
            if all(self.board[combo] == 1):
                return 1  # X wins
            if all(self.board[combo] == 2):
                return 2  # O wins
        if 0 not in self.board:
            return 0  # Draw
        return None  # Game not finished

    def step(self, action, player):        
        if self.board[action] != 0 or self.done:
            return self.board.copy(), -2, True  # Invalid move penalty
        self.board[action] = player
        winner = self.check_winner()
        if winner is not None:
            self.done = True
            self.winner = winner
            if winner == 1:
                return self.board.copy(), 1, True  # X wins
            elif winner == 2:
                return self.board.copy(), -1, True  # O wins
            else:
                return self.board.copy(), 0, True  # Draw
        return self.board.copy(), 0, False  # Game continues

    def get_valid_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size=9, hidden_size=128, output_size=9):
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
    def __init__(self, input_size=9, hidden_size=128, output_size=9, gamma=0.99, lr=0.01):
        self.policy = PolicyNetwork(input_size, hidden_size, output_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

    def select_action(self, state, valid_actions):
        state = torch.FloatTensor(state)
        probs = self.policy(state)
        # Mask invalid actions
        mask = torch.zeros(9)
        for a in valid_actions:
            mask[a] = 1
        masked_probs = probs * mask
        # Normalize probabilities for valid actions
        if masked_probs.sum() > 0:
            masked_probs = masked_probs / masked_probs.sum()
        else:
            masked_probs = torch.ones(9) / len(valid_actions)
        action = torch.multinomial(masked_probs, 1).item()
        log_prob = torch.log(probs[action])
        return action, log_prob

    def update_policy(self, rewards, log_probs):
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        # Normalize rewards (optional, for stability)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

# Random Opponent
def random_opponent_action(valid_actions):
    return np.random.choice(valid_actions)

# Training Loop
def train_reinforce_tictactoe():
    env = TicTacToeEnv()
    agent = ReinforceAgent(input_size=9, hidden_size=128, output_size=9, gamma=1.0, lr=0.05)
    num_episodes = 100000
    print_every = 1000
    wins, losses, draws = 0, 0, 0

    for episode in range(num_episodes):
        state = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            # Agent's turn (X)
            valid_actions = env.get_valid_actions()
            action, log_prob = agent.select_action(state, valid_actions)
            next_state, reward, done = env.step(action, player=1)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

            if not done:
                # Opponent's turn (O, random)
                valid_actions = env.get_valid_actions()
                action = random_opponent_action(valid_actions)
                state, reward, done = env.step(action, player=2)
                rewards[-1] += reward  # Update reward for the agent's last action

        # Update policy after the episode
        agent.update_policy(rewards, log_probs)

        # Track results
        if env.winner == 1:
            wins += 1
        elif env.winner == 2:
            losses += 1
        else:
            draws += 1

        # Print progress
        if (episode + 1) % print_every == 0:
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
            print(f"Win Rate: {wins / print_every:.2f}")
            wins, losses, draws = 0, 0, 0

    return agent

# Function to play a game with the trained agent
def play_game(agent):
    env = TicTacToeEnv()
    state = env.reset()
    done = False
    print("Initial Board (0: empty, 1: X, 2: O):")
    print(state.reshape(3, 3))

    while not done:
        # Human or random opponent's turn (O)
        valid_actions = env.get_valid_actions()
        #action = random_opponent_action(valid_actions)  # Replace with human input if desired
        x = input("select a place ")
        action = int(x)
        state, reward, done = env.step(action, player=2)
        print(f"Opponent (O) plays at position {action}:")
        print(state.reshape(3, 3))

        # Agent's turn (X)
        valid_actions = env.get_valid_actions()
        action, _ = agent.select_action(state, valid_actions)
        state, reward, done = env.step(action, player=1)
        print(f"Agent (X) plays at position {action}:")
        print(state.reshape(3, 3))

        if done:
            if reward == 1:
                print("Agent (X) wins!")
                restart_game()
            elif reward == -1:
                print("Opponent (O) wins!")
                restart_game()
            else:
                print("It's a draw!")
                restart_game()           
            break

        if done:
            if reward == -1:
                print("Opponent (O) wins!")
                restart_game()
            elif reward == 1:
                print("Agent (X) wins!")
                restart_game()
            else:
                print("It's a draw!")
                restart_game()              

if __name__ == "__main__":
    trained_agent = train_reinforce_tictactoe()
    print("\nPlaying a game with the trained agent:")
    play_game(trained_agent)

def restart_game():
    qa = input("do you want to restart the game? \n yes(y) or no(n) \n")
    if(qa == "y"):
        env = TicTacToeEnv()
        state = env.reset()
        done = False
        print("Initial Board (0: empty, 1: X, 2: O):")
        print(state.reshape(3, 3))
    else:   
        print("tahnk you for playing") 

