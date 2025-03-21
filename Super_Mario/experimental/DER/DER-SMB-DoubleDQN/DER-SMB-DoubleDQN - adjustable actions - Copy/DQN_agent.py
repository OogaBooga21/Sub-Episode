import torch
from torch import nn
import numpy as np
import random
import time
from DQN_net import DQN
from DQN_mem import Replay_Memory
from torch.utils.tensorboard import SummaryWriter

class RL_Agent:
    def __init__(self, env, mem_size, gamma, epsilon, epsilon_decay, epsilon_end, batch, step_update, death_parameters):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = env
        (stack_frames, self.img_s_h, self.img_s_w) = self.env.reset().shape
        output_layer_size = self.env.action_space.n
        
        self.online_network = DQN(stack_frames, self.img_s_h, output_layer_size)
        self.online_network.to(self.device)
        self.online_network.train()
        
        self.target_network = DQN(stack_frames, self.img_s_h, output_layer_size)
        self.target_network.to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        # self.online_network.train()
        
        print(self.online_network)
        print("Running on ", self.device)
        
        print(self.target_network)
        print("Running on ", self.device)
        
        self.best = DQN(stack_frames, self.img_s_h, output_layer_size)
        self.best.load_state_dict(self.online_network.state_dict())
        
        self.mem_size = mem_size
        self.replay_memory = Replay_Memory(mem_size)
        
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        
        self.gamma = gamma
        self.batch_size = batch
        
        self.episode_count = 0 # will need to fix the env
        self.step_update = step_update
        self.step_count = 0
        
        # DEATH PARAMETERS: 
        self.death_memory_size = death_parameters["death_memory_size"]
        self.death_steps = death_parameters["death_steps"]
        self.death_tries = death_parameters["death_tries"]
        self.death_epsilon = death_parameters["death_epsilon"]
        self.death_batch_size = death_parameters["death_batch_size"]
        
        self.death_memory = Replay_Memory(self.death_memory_size)
        #self.action_list=[]
        
        
        
        
        
        self.highscore = -99999
        self.writer = SummaryWriter('logs/DoubleDQN') #Logging
        
    def save_model(self, online_network, filename):
        torch.save(online_network.state_dict(), filename)
        
    def load_model(self, online_network, filename):
        online_network.load_state_dict(torch.load(filename))
        
    def update_epsilon(self):
        self.epsilon -= (1 - self.epsilon_end) / self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_end)  # Ensure lower bound

        
    def fill_memory(self):
        state_count = 0
        while state_count < self.mem_size:
            state = self.env.reset()
            terminal = False
            while not terminal: 
                action = self.env.random_action()
                next_state, reward, terminal = self.env.step(action)
                
                self.replay_memory.store_memory(state, action, reward, next_state, terminal)
                self.death_memory.store_memory(state, action, reward, next_state, terminal) ################################################################################################
                
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
                q_values = self.online_network.forward(state)
                action = q_values.argmax().item()
            return action
        
    def pick_alt_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q_values = self.online_network.forward(state)
    
            if q_values.dim() == 2:
                q_values = q_values.squeeze(0)  # Shape: [num_actions]
    
            topk = torch.topk(q_values, k=3)
            top_actions = topk.indices.cpu().tolist()  # Now a flat list [a1, a2, a3]
    
            progress = min(self.episode_count / 2500, 1.0)  # 2500 episodes to reach full exploration
            weight1 = 0.4 + (0.80 - 0.4) * progress
            weight2 = 0.4 + (0.19 - 0.4) * progress
            weight3 = 0.2 + (0.01 - 0.2) * progress
            
            weights = [weight1, weight2, weight3]
            chosen_action = random.choices(top_actions, weights=weights, k=1)[0]

        return chosen_action
    
    def learn(self):
        
        states_n, actions_n, rewards_n, next_states_n,terminals_n = self.replay_memory.random_memory_batch(self.batch_size)
        states_d, actions_d, rewards_d, next_states_d,terminals_d = self.death_memory.random_memory_batch(self.death_batch_size)
        
        
        states_n = torch.tensor(states_n, dtype=torch.float32).to(self.device)
        actions_n = torch.tensor(actions_n, dtype=torch.int64).to(self.device)
        rewards_n = torch.tensor(rewards_n, dtype=torch.float32).to(self.device)
        next_states_n = torch.tensor(next_states_n, dtype=torch.float32).to(self.device)
        terminals_n = torch.tensor(terminals_n, dtype=torch.int64).to(self.device)
        
        states_d = torch.tensor(states_d, dtype=torch.float32).to(self.device)
        actions_d = torch.tensor(actions_d, dtype=torch.int64).to(self.device)
        rewards_d = torch.tensor(rewards_d, dtype=torch.float32).to(self.device)
        next_states_d = torch.tensor(next_states_d, dtype=torch.float32).to(self.device)
        terminals_d = torch.tensor(terminals_d, dtype=torch.int64).to(self.device)
        
        states = torch.cat([states_n, states_d], dim=0)
        actions = torch.cat([actions_n, actions_d], dim=0)
        rewards = torch.cat([rewards_n, rewards_d], dim=0)
        next_states = torch.cat([next_states_n, next_states_d], dim=0)
        terminals = torch.cat([terminals_n, terminals_d], dim=0)
        
        perm = torch.randperm(states.size(0))
        states = states[perm]
        actions = actions[perm]
        rewards = rewards[perm]
        next_states = next_states[perm]
        terminals = terminals[perm]
        
        rewards = rewards.unsqueeze(1)
        terminals = terminals.unsqueeze(1)
        actions = actions.unsqueeze(1)
        
        predicted_q_values = self.online_network.forward(states)
        predicted_q_values = predicted_q_values.gather(1, actions)
        
        next_actions = self.online_network.forward(next_states).detach().argmax(dim=1, keepdim=True)
        target_next_q_values = self.target_network.forward(next_states).detach().gather(1, next_actions)
        # next_q_values = torch.max(next_q_values, dim=1, keepdim=True).values
        
        
        target_q_values = rewards + self.gamma * target_next_q_values * (1 - terminals)
        
        self.online_network.optimizer.zero_grad()
        loss= nn.functional.mse_loss(predicted_q_values, target_q_values)
        # loss = nn.functional.smooth_l1_loss(predicted_q_values, target_q_values)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=2.5)
        self.online_network.optimizer.step()
        
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
            
            action_list=[]#########################################################################################################################################
            while not terminal:
                action = self.pick_action(state)
                next_state, reward, terminal = self.env.step(action)
                
                action_list.append(action) ########################################################################################################################3
                
                self.replay_memory.store_memory(state, action, reward, next_state, terminal)
                state = next_state
                episode_reward += reward
                self.step_count += 1
                if self.step_count % self.step_update == 0:
                    self.target_network.load_state_dict(self.online_network.state_dict()) #update once every x steps
                
                if self.step_count % 4 == 0:
                    loss_copy = self.learn()
                    # _ = self.learn(self.death_memory,self.death_batch_size) ################################################################################################################
            
            self.episode_count += 1
            if len(action_list) > self.death_steps:
                self.replay_death(action_list) ##############################
            
            
            # loss_copy = self.learn()
        
    
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
                self.save_model(self.best,"bestDDQN.pt") #save current model
            
            if episode_reward >= self.highscore: # Save ?
                self.highscore = episode_reward
                self.best.load_state_dict(self.online_network.state_dict()) #save for later
                
            
        self.save_model(self.best,"bestDDQN.pt") #actual save    
    
    def replay_death(self,action_list):   #DEATH REPLAY ###############################################################################################################
        for _ in range(self.death_tries):
            state = self.env.reset()
            for a_i in range(len(action_list) - self.death_steps):
                state, _, _ = self.env.step(action_list[a_i])
                
            
            for new_step in range(2*self.death_steps):
                action = self.pick_alt_action(state)
                next_state, reward, terminal = self.env.step(action)
                
                if reward < 0:
                    reward = reward / 3 #less penalty for death
                    
                if new_step > self.death_steps and reward > 0:
                    reward = reward * 1.5 #more reward for surviving
                
                self.death_memory.store_memory(state, action, reward, next_state, terminal)
                state = next_state
                if terminal:
                    break

         
    def test(self, test_episodes):
        self.epsilon = self.epsilon_end
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


        # print(self.online_network.forward(states))
        
        #////////////////////////////////////////// learn //////////////////////////////////////////
        
        # self.fill_memory()
        # loss=self.learn()
        # print(loss)
        
        #////////////////////////////////////////// train //////////////////////////////////////////
        self.train(10)
        
        print("Unit Test Passed")