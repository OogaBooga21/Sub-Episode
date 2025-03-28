import torch
from torch import nn
import numpy as np
import random
import time
from DQN_net import DQN
from Dueling_DQN_net import DQN as Dueling_DQN
from DQN_mem import Replay_Memory
from torch.utils.tensorboard import SummaryWriter
import os

class RL_Agent:
    def __init__(self, env, agent_params, death_parameters):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_algorithm = agent_params["training_algorithm"]
        
        self.env = env
        (stack_frames, self.img_s_h, self.img_s_w) = self.env.reset().shape
        output_layer_size = self.env.action_space.n
        
        if self.training_algorithm == "DQN":
            self.online_network = DQN(stack_frames, self.img_s_h, output_layer_size, agent_params["learning_rate"])
            self.online_network.to(self.device)
            self.online_network.train()
            self.best = DQN(stack_frames, self.img_s_h, output_layer_size, agent_params["learning_rate"])
            self.best.load_state_dict(self.online_network.state_dict())
            
        if self.training_algorithm == "Double_DQN":
            self.online_network = DQN(stack_frames, self.img_s_h, output_layer_size, agent_params["learning_rate"])
            self.online_network.to(self.device)
            self.online_network.train()
            self.best = DQN(stack_frames, self.img_s_h, output_layer_size, agent_params["learning_rate"])
            self.best.load_state_dict(self.online_network.state_dict())
            
            self.target_network = DQN(stack_frames, self.img_s_h, output_layer_size, agent_params["learning_rate"])
            self.target_network.to(self.device)
            self.target_network.load_state_dict(self.online_network.state_dict())
            
        if self.training_algorithm == "Dueling_DQN":
            self.online_network = Dueling_DQN(stack_frames, self.img_s_h, output_layer_size, agent_params["learning_rate"])
            self.online_network.to(self.device)
            self.online_network.train()
            self.best = Dueling_DQN(stack_frames, self.img_s_h, output_layer_size, agent_params["learning_rate"])
            self.best.load_state_dict(self.online_network.state_dict())
        # self.online_network.train()
        self.name = agent_params["save_name"]
        
        print("Agent: ", self.name)
        print("Parameters: ", agent_params)
        print("Death Parameters: ", death_parameters)
        print(self.online_network)
        print("Running on ", self.device)
        
        if self.training_algorithm == "Double_DQN":
            print(self.target_network)
            print("Running on ", self.device)
        
        self.mem_size = agent_params["memory_size"]
        self.replay_memory = Replay_Memory(self.mem_size)
        
        self.epsilon = agent_params["epsilon"]
        self.epsilon_decay = agent_params["epsilon_decay"]
        self.epsilon_end = agent_params["epsilon_min"]
        
        self.gamma = agent_params["gamma"]
        self.batch_size = agent_params["batch_size"]
        
        self.episode_count = 0 # will need to fix the env
        self.step_update = agent_params["update_freq"]
        self.step_count = 0
        
        # DEATH PARAMETERS: 
        self.death_memory_size = death_parameters["death_memory_size"]
        self.death_memory = Replay_Memory(self.death_memory_size)
        self.death_batch_size = death_parameters["death_batch_size"]
        self.death_steps = death_parameters["death_steps"]
        self.death_tries = death_parameters["death_tries"]
        self.death_start_weights = death_parameters["death_start_weights"]
        self.death_end_weights = death_parameters["death_end_weights"]
        self.death_weights_ep_transition = death_parameters["death_weights_ep_transition"]
        # self.death_epsilon = death_parameters["death_epsilon"]
        
        self.highscore = -99999
        self.old_hisghscore = -99999
        
        current_dir = os.getcwd()
        base_model_dir = os.path.join(current_dir, "TrainModelResults")
        base_log_dir = os.path.join(current_dir, "TrainLogs")
    
        os.makedirs(base_model_dir, exist_ok=True)
        os.makedirs(base_log_dir, exist_ok=True)
    
        self.model_save_path = os.path.join(base_model_dir, self.name +".pt")
        # self.model_save_path = self.model_save_path + ".pt"
        self.log_save_path = os.path.join(base_log_dir, self.name)
        # print(self.model_save_path)
    
        # os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(self.log_save_path, exist_ok=True)
        self.writer = SummaryWriter(self.log_save_path)
        
        
    def save_model(self, online_network, filename):
        torch.save(online_network.state_dict(), filename)
        
    def load_model(self, filename):
        self.online_network.load_state_dict(torch.load(filename))
        if self.training_algorithm == "Double_DQN":
            self.target_network.load_state_dict(torch.load(filename))
        
    def update_epsilon(self):
        self.epsilon = max(self.epsilon - (1 - self.epsilon_end)/self.epsilon_decay, self.epsilon_end)

        
    def fill_memory(self):
        state_count = 0
        while state_count < self.mem_size:
            state = self.env.reset()
            terminal = False
            while not terminal:
                action = self.pick_action(state)
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
                q_values = q_values.squeeze(0)
    
            topk = torch.topk(q_values, k=3)
            top_actions = topk.indices.cpu().tolist()
    
            progress = min(self.episode_count / self.death_weights_ep_transition, 1.0)
            weight1 = self.death_start_weights[0] + (self.death_end_weights[0] - self.death_start_weights[0]) * progress
            weight2 = self.death_start_weights[1] + (self.death_end_weights[1] - self.death_start_weights[1]) * progress
            weight3 = self.death_start_weights[2] + (self.death_end_weights[2] - self.death_start_weights[2]) * progress
            
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
        
        next_q_values = self.online_network.forward(next_states).detach()
        next_q_values = torch.max(next_q_values, dim=1, keepdim=True).values
        
        
        target_q_values = rewards + self.gamma * next_q_values * (1 - terminals)
        
        self.online_network.optimizer.zero_grad()
        loss= nn.functional.mse_loss(predicted_q_values, target_q_values)
        loss.backward()
        self.online_network.optimizer.step()
        
        return loss
    
    def double_learn(self):
        
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
        step_history = []
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
                    if self.training_algorithm == "Double_DQN":
                        self.target_network.load_state_dict(self.online_network.state_dict()) #update once every x steps
                
                if self.step_count % 4 == 0:
                    if self.training_algorithm == "Double_DQN":
                        loss_copy = self.double_learn()
                    else:
                        loss_copy = self.learn()
            
            self.episode_count += 1
            
            if len(action_list) > self.death_steps:
                self.replay_death(action_list) ##############################
            
            loss_history.append(loss_copy.item())
            reward_history.append(episode_reward)
            step_history.append(self.env.step_counter)
            
            # print(f'ep:{episode_count}, HS: {int(self.highscore)}, BA:{int(current_avg_score)}, LA:{current_avg_loss}, E:{self.epsilon}')   
            self.update_epsilon()
            
            if(episode_count % 10 == 0): # Update user
                self.log(train_episodes, episode_count, reward_history, loss_history,step_history)
                self.save_model(self.best,self.model_save_path) #save current model
            
            if episode_reward >= self.highscore: # Save ?
                self.highscore = episode_reward
                self.best.load_state_dict(self.online_network.state_dict()) #save for later
                
            # Early stopping
            if episode_count % 1000 == 0 and episode_count > train_episodes/2:
                current_avg = np.mean(reward_history[-1000:])
                old_avg = np.mean(reward_history[-2000:-1000])
                if current_avg < old_avg*(1 + 0.01):
                    print("Early stopping warning, no average improvement in last 1000 episodes")
                    if self.highscore < self.old_highscore * (1 + 0.02):
                        print("Early stopping, not even highscore improved")
                        break
                self.old_highscore = self.highscore
                
            
        self.save_model(self.best,self.model_save_path) #actual save       
    
    def log(self,train_episodes, episode_count, reward_history, loss_history, step_history):
        avg_reward_10 = np.mean(reward_history[-10:]) if len(reward_history) >= 10 else np.mean(reward_history)
        avg_reward_1000 = np.mean(reward_history[-1000:]) if len(reward_history) >= 1000 else np.mean(reward_history)
        avg_loss = np.mean(loss_history[-10:]) if len(loss_history) >= 10 else np.mean(loss_history)
        avg_steps = np.mean(step_history[-10:]) if len(step_history) >= 10 else np.mean(step_history)
    
        current_time = time.localtime()
        time_str = f"{current_time.tm_hour:02d}:{current_time.tm_min:02d}:{current_time.tm_sec:02d}"
    
        separator = "-" * 30
        log_message = (
            f"{separator}\n"
            f"Episode Progress:   {episode_count}/{train_episodes}\n"
            f"Highscore:          {int(self.highscore)}\n"
            f"10-episode Avg:     {avg_reward_10:.2f}\n"
            f"1000-episode Avg:   {avg_reward_1000:.2f}\n"
            f"Step Avg:           {avg_steps:.2f}\n"
            f"Loss Avg:           {avg_loss:.4f}\n"
            f"Epsilon:            {self.epsilon:.4f}\n"
            f"Time:               {time_str}\n"
            f"{separator}\n"
    )
        print(log_message)
    
        self.writer.add_scalar('Training/10-Episode Average Reward', avg_reward_10, episode_count)
        self.writer.add_scalar('Training/1000-Episode Average Reward', avg_reward_1000, episode_count)
        self.writer.add_scalar('Training/Step Average', avg_steps, episode_count)
        self.writer.add_scalar('Training/Average Loss', avg_loss, episode_count)
        self.writer.add_scalar('Training/Epsilon', self.epsilon, episode_count)
        self.writer.add_scalar('Training/Highscore', self.highscore, episode_count)
    
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