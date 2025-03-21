import cv2
import time
from collections import deque
import numpy as np
import cv2
from PIL import Image
import pygame

class gym_Env_Wrapper:
    
    window_size = (150,150)
    screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
    pygame.display.set_caption("Car_racing")
    
    def __init__(self,env,mini_render, stack_frames, rescale_factor,max_steps):
        #need rescaled sisez\
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.env = env
        self.mini_render = mini_render
        self.rescale_factor = rescale_factor
        self.initial_skip = 50 #this env has some useless frames at the begining
        self.step_counter = 0
        self.max_steps = max_steps
        self.stack_frames = stack_frames
        self.frame_stack = deque(maxlen=stack_frames)
        
        
        dummy_state = self.env.reset()
        img_height = dummy_state.shape[0]
        img_width = dummy_state.shape[1]
        self.img_s_h = int(self.rescale_factor * img_height)
        self.img_s_w = int(self.rescale_factor * img_width)
        
        
    def preprocess_state(self,state):
        gray_image = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
        if self.rescale_factor != 1.0:
            gray_image = cv2.resize(gray_image, (self.img_s_h, self.img_s_w), interpolation=cv2.INTER_AREA)
        return gray_image / 255.0
    
    def reset(self):
        s = self.env.reset()
        
        for _ in range (self.initial_skip):
            s,_,_,_ = self.env.step(0) #no action
            
        s=self.preprocess_state(s)
        
        for _ in range(self.stack_frames):
            self.frame_stack.append(s)
            
        self.episode_start_time = time.time()
        self.episode_reward = 0
        self.step_counter = 0
        
        return np.stack(self.frame_stack,axis=0)

    def step(self, action):
        stack_reward = 0
        terminal = 0
        for _ in range(self.stack_frames):
            s,r,terminal,info = self.env.step(action)    
            stack_reward+=r
            
            s = self.preprocess_state(s)
            self.frame_stack.append(s)
            # ====================Stop Conditions========================
            if (terminal == True) or (self.step_counter >= self.max_steps):
                terminal = 1
                break
        if(self.mini_render == True):    
            self.display_image(self.frame_stack[0])
        self.step_counter +=1
        
        return np.stack(self.frame_stack,axis=0), stack_reward, terminal
    
    def random_action(self):
        return self.env.action_space.sample()
    
    # def env_state_shape(self):
    #     return self.frame_stack.shape()
    
    # def action_space(self):
    #     return self.env.action_space
    
    def display_image(self, image_data): 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            
        image_data = (image_data * 255).astype(np.uint8)
        image_surface = pygame.surfarray.make_surface(np.stack([image_data]*3, axis=-1))
        image_surface = pygame.transform.scale(image_surface, self.window_size)
        self.screen.fill((0, 0, 0))
        self.screen.blit(image_surface, (0, 0))
        pygame.display.flip()