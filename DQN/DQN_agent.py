import torch
from torch import nn
import numpy as np
import random
import time
from DQN_net import DQN
from DQN_mem import Replay_Memory
from torch.utils.tensorboard import SummaryWriter

class RL_Agent:
    def __init__(self, env, mem_size, gamma, epsilon, epsilon_decay, epsilon_end, batch):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = env
        (stack_frames, self.img_s_h, self.img_s_w) = self.env.reset().shape
        output_layer_size = self.env.action_space.n
        
        self.network = DQN(stack_frames, self.img_s_h, output_layer_size)
        self.network.to(self.device)
        self.network.train()
        
        print(self.network)
        print("Running on ", self.device)
        
        self.best = DQN(stack_frames, self.img_s_h, output_layer_size)
        self.best.load_state_dict(self.network.state_dict())
        
        self.mem_size = mem_size
        self.replay_memory = Replay_Memory(mem_size)
        
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        
        self.gamma = gamma
        self.batch_size = batch
        
        self.episode_count = 0 # will need to fix the env
        self.step_count = 0
        
        
        self.highscore = -99999
        self.writer = SummaryWriter('logs/DQN') #Logging
        
    def save_model(self, network, filename):
        torch.save(network.state_dict(), filename)
        
    def load_model(self, network, filename):
        network.load_state_dict(torch.load(filename))
        
    def update_epsilon(self):
        self.epsilon = max(self.epsilon - (1 - self.epsilon_end)/self.epsilon_decay, self.epsilon_end)
        
    def fill_memory(self):
        state_count = 0
        while state_count < self.mem_size:
            state = self.env.reset()
            terminal = False
            while not terminal: 
                action = self.env.random_action()
                next_state, reward, terminal = self.env.step(action)
                
                self.replay_memory.store_memory(state, action, reward, next_state, terminal)
                state = next_state
                state_count += 1
                
                if state_count>=self.mem_size:
                    terminal = True
                    
                percent = round(100 * state_count /  self.mem_size, 2)
                filled_length = int(50 * state_count //  self.mem_size)
                bar = f'[{filled_length * "#"}{"-" * (50 - filled_length)}]'
                print(f'{"Filling Replay Memory: "} {bar} {percent:.2f}% {" Done."}', end="\r")
                if state_count ==  self.mem_size:
                    print()
                    
    def pick_action(self, state):
        if random.random() < self.epsilon:
            return self.env.random_action()
        else:
            
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                q_values = self.network.forward(state)
                action = q_values.argmax().item()
            return action
    
    def learn(self):
        
        states, actions, rewards, next_states,terminals = self.replay_memory.random_memory_batch(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        terminals = torch.tensor(terminals, dtype=torch.int64).to(self.device)
        
        rewards = rewards.unsqueeze(1)
        terminals = terminals.unsqueeze(1)
        actions = actions.unsqueeze(1)
        
        predicted_q_values = self.network.forward(states)
        predicted_q_values = predicted_q_values.gather(1, actions)
        
        next_q_values = self.network.forward(next_states).detach()
        next_q_values = torch.max(next_q_values, dim=1, keepdim=True).values
        
        
        target_q_values = rewards + self.gamma * next_q_values * (1 - terminals)
        
        self.network.optimizer.zero_grad()
        loss= nn.functional.mse_loss(predicted_q_values, target_q_values)
        loss.backward()
        self.network.optimizer.step()
        
        return loss
    
    def train(self, train_episodes):
        print("Training...")
        self.fill_memory()
        
        reward_history = []
        loss_history = []
        
        for episode_count in range(train_episodes):
            state = self.env.reset()
            terminal = False
            episode_reward = 0
            
            while not terminal:
                action = self.pick_action(state)
                next_state, reward, terminal = self.env.step(action)
                
                self.replay_memory.store_memory(state, action, reward, next_state, terminal)
                state = next_state
                episode_reward += reward
                self.step_count += 1
            
            self.episode_count += 1
            loss_copy = self.learn()
        
    
            loss_history.append(loss_copy.item())
            reward_history.append(episode_reward)
            current_avg_score = np.mean(reward_history[-10:]) 
            current_avg_loss = np.mean(loss_history[-10:])
            
            # print(f'ep:{episode_count}, HS: {int(self.highscore)}, BA:{int(current_avg_score)}, LA:{current_avg_loss}, E:{self.epsilon}')   
            self.update_epsilon()
            
            if(episode_count % 10 == 0): # Update user
                print(f'ep:{episode_count}, HS: {int(self.highscore)}, BA:{int(current_avg_score)}, LA:{current_avg_loss}, E:{self.epsilon}')              
                print([int(reward) for reward in reward_history[-10:]], time.localtime().tm_hour,time.localtime().tm_min,time.localtime().tm_sec)
                # Inside RL_Agent.train() method, after updating reward_history, loss_history, etc.
                self.writer.add_scalar('Training/Average Reward', current_avg_score, episode_count)
                self.writer.add_scalar('Training/Average Loss', current_avg_loss, episode_count)
                self.writer.add_scalar('Training/Epsilon', self.epsilon, episode_count)
                self.save_model(self.best,"best.pt") #save current model
            
            if episode_reward >= self.highscore: # Save ?
                self.highscore = episode_reward
                self.best.load_state_dict(self.network.state_dict()) #save for later
                
            
        self.save_model(self.best,"best.pt") #actual save    
         
    def test(self, test_episodes):
        self.epsilon = 0
        for episode_count in range(test_episodes):
            state=self.env.reset()
            done=False
            episode_reward = 0
            
            while not done: #should be fixed? 
                action = self.pick_action(state)
                new_state, reward, done = self.env.step(action)
                episode_reward += reward
                state = new_state

            print('ep:{}, ep score: {}'.format(episode_count,episode_reward))
            
    def unit_test(self):
        print("Unit Test")
        #////////////////////////////////////////// forward //////////////////////////////////////////
        # state = self.env.reset()
        # # states = torch.tensor(state, dtype=torch.float32).to(self.device)
        
        # self.fill_memory()
        # states, _,_,_,_ = self.replay_memory.random_memory_batch(5)
        # states = torch.tensor(states, dtype=torch.float32).to(self.device)


        # print(self.network.forward(states))
        
        #////////////////////////////////////////// learn //////////////////////////////////////////
        
        # self.fill_memory()
        # loss=self.learn()
        # print(loss)
        
        #////////////////////////////////////////// train //////////////////////////////////////////
        self.train(10)
        
        print("Unit Test Passed")