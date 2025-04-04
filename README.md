# Sub-Episode Training
A Learning optimization for Q-value based Reinforcement Learning algorithms which aims to help the agent learn "tough" situations faster.
This repository supports experiments in two game environments:

- **Super Mario Bros** (`gym-super-mario-bros`)
- **Car Racing** (OpenAI Gym)

## Installation

Clone the repository and install the requirements:


#### Clone the repository
```bash
git clone https://github.com/yourusername/qlearning-rl-optimisation.git
cd qlearning-rl-optimisation
```
```bash
# (Optional but recommended) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate      # On Windows, use: venv\Scripts\activate
```

#### Install required packages (if this fails, go back, and use a venv)
```bash
pip install -r requirements.txt
```
---

We tested the code using Python 3.10 & 3.11 and the libraries from requirements.txt (note that updating some libraries might make the program not work). The code should run on most computers with a GPU and CUDA (or ROCm) installed. (Should work on CPU as well, but expect 3-10x slowdown)
## Arguments
```bash
usage: entry.py [-h] --game {Car_Racing,Super_Mario} --version {base,experimental} --training_algorithm {DQN,Double_DQN,Dueling_DQN}
                [--train_episodes TRAIN_EPISODES] [--gamma GAMMA] [--batch_size BATCH_SIZE] [--memory_size MEMORY_SIZE]
                [--epsilon EPSILON] [--epsilon_decay EPSILON_DECAY] [--epsilon_min EPSILON_MIN] [--learning_rate LEARNING_RATE]
                [--save_name SAVE_NAME] [--skip_frames SKIP_FRAMES] [--mini_render MINI_RENDER] [--max_steps MAX_STEPS]
                [--update_freq UPDATE_FREQ] [--death_memory_size DEATH_MEMORY_SIZE] [--death_batch_size DEATH_BATCH_SIZE]
                [--death_steps DEATH_STEPS] [--death_tries DEATH_TRIES] [--death_start_weights DEATH_START_WEIGHTS]
                [--death_end_weights DEATH_END_WEIGHTS] [--death_weights_ep_transition DEATH_WEIGHTS_EP_TRANSITION]
                [--continue_training CONTINUE_TRAINING] [--test_agent TEST_AGENT] [--test_epsilon TEST_EPSILON]
                [--test_episodes TEST_EPISODES]
The following arguments are required: --game, --version, --training_algorithm
```
## How to use:
### Training
Once you have the repo, the easiest way to start training is to navigate to your folder, open a cmd/terminal and run: 
```bash
python3 entry.py --game Super_Mario --version base --training_algorithm DQN
```
more parameters: Base DQN on Super Mario, 1000 episodes, and saving with the name mario_dqn_base
```bash
python entry.py --game=Super_Mario --version=base --training_algorithm=DQN --train_episodes=1000 --save_name=mario_dqn_base
```

experimental (difficult situation replay): Double DQN with experimental 'death' mechanics
```bash
python entry.py --game=Car_Racing --version=experimental --training_algorithm=Double_DQN \
--train_episodes=800 \
--death_memory_size=1000 \
--death_batch_size=64 \
--death_steps=20 \
--death_tries=3 \
--death_weights_ep_transition=300 \
--save_name=car_exp_double_dqn
```
---
### Continue Training
Resume training from a saved model in `TrainModelResults/`:
```bash
python entry.py --game=Car_Racing --version=experimental --training_algorithm=Dueling_DQN \
--continue_training=car_dueling_checkpoint --train_episodes=200
```
---
### Testing
To evaluate a trained model:
```bash
python entry.py --game=Car_Racing --version=base --training_algorithm=DQN \
--test_agent=car_dqn_run1 --test_episodes=10
```
With stochastic exploration:
```bash
python entry.py --game=Super_Mario --version=base --training_algorithm=Double_DQN \
--test_agent=mario_double_dqn --test_episodes=5 --test_epsilon=0.1
```
---

## Hyperparameters & Defaults
### Hyperparameters
- Common Parameters:
```
[--train_episodes TRAIN_EPISODES] - Nr of episodes the agent should train for (keep in mind the algo has an early stopping mechanism, if neihter the average nor the highscore improve)
[--gamma GAMMA] - Discount factor.
[--batch_size BATCH_SIZE] - Nr of samples in a batch when training,
[--memory_size MEMORY_SIZE] - Size of memory of the agent (this is how many scenes the agent will remember)
[--epsilon EPSILON] - Starting epsilon
[--epsilon_decay EPSILON_DECAY] - Nr of episodes in which the epsilon will go from the initial value to the final value.
[--epsilon_min EPSILON_MIN] - Final epsilon value
[--learning_rate LEARNING_RATE] - Learning rate of the neural network used by the agent.
[--save_name SAVE_NAME] - Name of the current agent.
[--skip_frames SKIP_FRAMES] - How many stacked frames can the agent see.
[--mini_render MINI_RENDER] - A small window which shows how the agent is acting in the environment.
[--max_steps MAX_STEPS] - Nr of maximum steps the agent can take in an environment.
[--test_episodes TEST_EPISODES] - Nr of episodes use for testing of the agent. (used at the end of training or in sepparate testing)
```
- Only for DoubleDQN:
```
[--update_freq UPDATE_FREQ] - Nr of steps between each offline network updates.
```
- Death Parameters (difficult situations)(only used in experimental runs):
```
[--death_memory_size DEATH_MEMORY_SIZE] - Size of memory in which agent will keep it's "difficult" scenes.
[--death_batch_size DEATH_BATCH_SIZE] - Nr of samples form the death memory used for each batch of training.
[--death_steps DEATH_STEPS] - Nr of steps that define a "difficult" scene.
[--death_tries DEATH_TRIES] - How many times will the agent attempt the same "difficult scene" with alternative actions.
[--death_start_weights DEATH_START_WEIGHTS] - The starting weights of the alternative actions taken by the agent. 
[--death_end_weights DEATH_END_WEIGHTS] - The end weights of the alternative actions taken by the agent.
[--death_weights_ep_transition DEATH_WEIGHTS_EP_TRANSITION] - Nr of episodes in which the weights will shift from the starting weights to the end weights.
```
- Continue training & testing (no defaults only used if specified):
```
[--continue_training CONTINUE_TRAINING] - Loads an agent from the TrainModelResults folder by this name to continue training the agent.
[--test_agent TEST_AGENT] - Loads an agent from the TrainModelResults folder by this name to test the agent.
[--test_epsilon TEST_EPSILON] - Epsilon used for random actions taken in the tests.
```
###Defaults
- Car Racing:
```
[--train_episodes TRAIN_EPISODES] - 4000
[--gamma GAMMA] - 0.85
[--batch_size BATCH_SIZE] - 512 
[--memory_size MEMORY_SIZE] - 8192
[--epsilon EPSILON] - 1
[--epsilon_decay EPSILON_DECAY] - 1000
[--epsilon_min EPSILON_MIN] - 0
[--learning_rate LEARNING_RATE] - 0.0002
[--save_name SAVE_NAME] - Car_Racing_VERSION_TRAININGALGORITHM_TIME_DATE
[--skip_frames SKIP_FRAMES] - 4
[--mini_render MINI_RENDER] - True
[--max_steps MAX_STEPS] - 250
[--update_freq UPDATE_FREQ] - 1000
[--death_memory_size DEATH_MEMORY_SIZE] -  1024
[--death_batch_size DEATH_BATCH_SIZE] - 64
[--death_steps DEATH_STEPS] - 25
[--death_tries DEATH_TRIES] - 1
[--death_start_weights DEATH_START_WEIGHTS] - [0.4,0.4,0.2]
[--death_end_weights DEATH_END_WEIGHTS] - [0.7,0.2,0.1]
[--death_weights_ep_transition DEATH_WEIGHTS_EP_TRANSITION] - 1000 
```
- Super Mario:
```
[--train_episodes TRAIN_EPISODES] - 10000
[--gamma GAMMA] - 0.9
[--batch_size BATCH_SIZE] - 32
[--memory_size MEMORY_SIZE] - 30000
[--epsilon EPSILON] - 1
[--epsilon_decay EPSILON_DECAY] - 500
[--epsilon_min EPSILON_MIN] - 0
[--learning_rate LEARNING_RATE] - 0.0001
[--save_name SAVE_NAME] - Car_Racing_VERSION_TRAININGALGORITHM_TIME_DATE
[--skip_frames SKIP_FRAMES] - 4
[--mini_render MINI_RENDER] - True
[--max_steps MAX_STEPS] - 90000
[--update_freq UPDATE_FREQ] - 1000
[--death_memory_size DEATH_MEMORY_SIZE] -  8000
[--death_batch_size DEATH_BATCH_SIZE] - 4
[--death_steps DEATH_STEPS] - 25
[--death_tries DEATH_TRIES] - 1
[--death_start_weights DEATH_START_WEIGHTS] - [0.4,0.4,0.2]
[--death_end_weights DEATH_END_WEIGHTS] - [0.7,0.2,0.1]
[--death_weights_ep_transition DEATH_WEIGHTS_EP_TRANSITION] - 650
```

## Output, Logging and Saving

###Output
When run, the program will output some important information to the console: 
- Name of the agent for future reference:
  ```
  ex: Agent:  Super_Mario_experimental_Double_DQN_20250404_182758
  ```
- Current parameters:
  ```
  ex: Parameters:  {'gamma': 0.9, 'batch_size': 32, 'memory_size': 30000, 'epsilon': 1.0, 'epsilon_decay': 500, 'epsilon_min': 0.0, 'update_freq': 1000, 'learning_rate': 0.0001, 'save_name': 'Super_Mario_experimental_Double_DQN_20250404_182758', 'training_algorithm': 'Double_DQN'}
  ```
- Current death parameters (if used):
  ```
  ex: Death Parameters:  {'death_memory_size': 8000, 'death_batch_size': 4, 'death_steps': 25, 'death_tries': 1, 'death_start_weights': [0.4, 0.4, 0.2], 'death_end_weights': [0.7, 0.2, 0.1], 'death_weights_ep_transition': 650}
  ```
- Architecture + device it's running on (cuda for GPU (regardless if using CUDA or ROCm), or cpu):
  ```
  ex:
  DQN(
  (convolutions): Sequential(
    (0): Conv2d(4, 32, kernel_size=(8, 8), stride=(4, 4))
    (1): ReLU()
    (2): Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
    (3): ReLU()
    (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
    (5): ReLU()
  )
  (dnn): Sequential(
    (0): Linear(in_features=3136, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=7, bias=True)
  )
  )
  Running on  cuda
  ```
- Replay memory progress-bar, this will need to fill up once when the program is started:
  ```
  ex: Filling Replay Memory:  [##################################################] 100.00%  Done.
  ```
- Training logs:
  ```
  ex: 
  ------------------------------
  Episode Progress:   0/10000
  Highscore:          -99999
  10-episode Avg:     1457.00
  1000-episode Avg:   1457.00
  Step Avg:           327.00
  Loss Avg:           18.8848
  Epsilon:            0.9980
  Time:               18:30:51
  ------------------------------
  ```
###Logging
This program uses tensorboard for logging, and all logs will be found in the "TrainLogs" folder (will be automatically generated if it doesn't already exist). To view the logs; just open a terminal in the project's folder and run:
```bash
tensorboard --logdir=TrainLogs
```
This should return a link, which when clicked will open a tensorboard page, where you can see all runs, even the ones in progress.

---
