import cv2
import time
from collections import deque
import numpy as np
import cv2
from PIL import Image
import pygame

class gym_Env_Wrapper:
    # stopping_time = 0
    # # stopping_reward = 0
    # stopping_steps = 0
    
    # window_size = (150,150)
    # screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
    # pygame.display.set_caption("Grayscale Image")
    
    def __init__(self,env,mini_render,initial_skip_frames, skip_frames, stack_frames, rescale_factor,stopping_bad_steps,stopping_time,stopping_steps):
        #need rescaled sisez
    def preprocess_state(self,state):
        gray_image = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY) / 255.0
        if self.rescale_factor != 1.0:
            gray_image = cv2.resize(gray_image, (self.img_s_h, self.img_s_w), interpolation=cv2.INTER_AREA)
        return gray_image
    
    def reset(self):
        s, _ = self.env.reset()
        
        for _ in range (self.initial_skip):
            s,_,_,_,_ = self.env.step(0) #no action
            
        s=self.preprocess_state(s)
        
        for _ in range(self.stack_frames):
            self.frame_stack.append(s)
            
        self.episode_start_time = time.time()
        self.episode_reward = 0
        self.episode_steps = 0
        self.bad_step_counter = 0
        
        return self.frame_stack

    def step(self, action):
        stack_reward = 0
        for _ in range(self.skip_frames):
            s,r,terminal,truncated,info = self.env.step(action)    
            stack_reward+=r
            
            s = self.preprocess_state(s)
            self.frame_stack.append(s)
                
            # ====================Stop Conditions========================
            if terminal == True:
                break
    
    def random_action(self):
    
    def current_frame_stack(self):
    
    def env_state_shape(self):
    
    def action_space(self):
    
    def action_space_size(self):
    
    def display_image(self, image_data):