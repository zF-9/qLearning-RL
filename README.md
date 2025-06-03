# Q-Learning : Reinforcement Learning (ML)
visualization and simulation of reinforcement learning algorithm with Python.

## Reinforcement Learning ##
A machine learning ecosystem where an agent learns by interacting with the environment to obtain the optimal strategy for achieveing the goal. implement usupervied machine learning, where ingesting and proccessing data i a pre-requisite. 
reinforcement learning doe not require data berforehand. instead, it learns from the environment and reward system to make better decision. 

 ## Project Requirements ##
    - Python 3.X.X 
    - PyTorch 
    - Pygame (for environment and agent; option: run on a resourced heavy game)

  ### Install PyGame (Linux/Windows)
  ```
    pip install pygame
  ```

  ### Install PyTorch (Linux/Windows)
  ```
    Install pip
    Install CUDA, if your machine has a CUDA-enabled GPU. [CUDA](https://developer.nvidia.com/cuda-gpus)
  ```

## Q-Learning ##
Q-Learning is a model-free, value-based, off-policy algortihm that will find the best series of actions based on the agent's current state, which were then represented by how valuable the action is in maximizing future rewards. 
in contrast to the model-based algorithm, model-free algorthim learns the consequences of their actions through the experience. 

### RL Algorithm
<p style="align:center"><img src="https://github.com/zF-9/qLearning-RL/blob/b376184624fcb448305434cfb8b87485ad5216f6/img/qLearning-pokemon.png"></p>
 
## terminology in Q-Learning ##
1. Environment : the world or the system in which the agent interact.
2. Agent : the protagonist of the environement. 
3. States : the current postion of the agent in the environment.
4. Actions : step taken by the agent in particular state.
5. Rewards : each action taken, the agent will recieve reward or penalty.
6. Episodes : the end of the stages, where agent can no longer take new actions (regardless if the agent completes or failed the objective).
7. Q-table : structured table in which sets of state and actions were to be kept for future iteration.
8. Temporal Difference : use to estimate the expected value (optimal Q-value) by using the current state & action and comparing it to the previous state & actions.  

### Demo ###
**simulation #1**
 <img src="https://github.com/zF-9/qLearning-RL/blob/b376184624fcb448305434cfb8b87485ad5216f6/img/RL-ML-QL.gif"> 
 
**simulation #2**
<img src="https://github.com/zF-9/qLearning-RL/blob/b376184624fcb448305434cfb8b87485ad5216f6/img/RL-Q.gif">

  ### Prospect
  ```
    - Adjusting the model hyperparameters (number of episodes; learning rate; evaluation seed of the environment; epsilon probabilty; decay rate).
    - Visualing the Exploration-Exploitation Trade-off.
    - experimenting with automating website navigation using Q-Learning.
  ```


