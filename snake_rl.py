import os
import csv
import pygame
import numpy as np
import random
from collections import deque
import collections

file_path = r"C:\Users\Fauzi\Desktop\RL_qLearning\output_result.csv" 

# Constants for the game
GRID_SIZE = 20
GRID_WIDTH = 30
GRID_HEIGHT = 20
CELL_SIZE = 20
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class SnakeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Reinforcement Learning pakai game snake - next: try run guna PS emulator")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = RIGHT
        self.food = self.place_food()
        self.score = 0
        self.done = False
        return self.get_state()

    def place_food(self):
        while True:
            food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if food not in self.snake:
                return food

    def get_state(self):
        head = self.snake[0]
        food_dx = self.food[0] - head[0]
        food_dy = self.food[1] - head[1]
        return (food_dx, food_dy, self.direction)

    def step(self, action):
        # Action: 0=Continue, 1=Turn Left, 2=Turn Right
        if action == 1:
            if self.direction == UP: self.direction = LEFT
            elif self.direction == LEFT: self.direction = DOWN
            elif self.direction == DOWN: self.direction = RIGHT
            elif self.direction == RIGHT: self.direction = UP
        elif action == 2:
            if self.direction == UP: self.direction = RIGHT
            elif self.direction == RIGHT: self.direction = DOWN
            elif self.direction == DOWN: self.direction = LEFT
            elif self.direction == LEFT: self.direction = UP

        # Move snake
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        # Check for collisions with walls or self
        if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or
            new_head[1] < 0 or new_head[1] >= GRID_HEIGHT or
            new_head in self.snake):
            reward = -10
            self.done = True
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                self.score += 1
                reward = 10
                self.food = self.place_food()
            else:
                self.snake.pop()
                reward = 0

        return self.get_state(), reward, self.done

    def render(self):
        self.screen.fill(BLACK)
        for segment in self.snake:
            pygame.draw.rect(self.screen, GREEN, (segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(self.screen, RED, (self.food[0] * CELL_SIZE, self.food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.display.flip()
        self.clock.tick(10)

class QLearningAgent:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.q_table = {}
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_space)
        return self.get_best_action(state)

    def get_best_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_space)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_space)

        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def main():
    game = SnakeGame()
    agent = QLearningAgent(state_space=None, action_space=3)  # 3 actions: Continue, Left, Right
    episodes = 100000

    for episode in range(episodes):
        state = game.reset()
        total_reward = 0

        while not game.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = agent.get_action(state)
            next_state, reward, done = game.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            game.render()

        with open(file_path, 'w', newline='') as file:
            raw_data = f"Episode: {episode + 1}/{episodes}, Score: {game.score}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}"
            epi_data = [f"Episode: {episode + 1}/{episodes}", f"Score: {game.score}", f"Total Reward: {total_reward}", f"Epsilon: {agent.epsilon:.3f}" ]

            print(epi_data)

            file.write(str(raw_data))
            file.close()

    pygame.quit()

if __name__ == "__main__":
    main()