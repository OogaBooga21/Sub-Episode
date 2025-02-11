import retro
import pygame
import numpy as np

# env = retro.make(game="SuperMarioBros-Nes")
env = retro.make('SuperMarioKart-Snes', 'ChocoIsland', render_mode="rgb_array")
env.reset()

pygame.init()

window_size = (256,224)
screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
pygame.display.set_caption('image')
 
# paint screen one time
pygame.display.flip()


for i in range(10000):
    obs,_,_,_,_ = env.step([1,0,0,0,0,0,0,0,0,0,0,0])
    surface = pygame.surfarray.make_surface(obs)
    
    surface = pygame.transform.rotate(surface, -90)

    screen.blit(surface, (0, 0))
    pygame.display.flip()

    env.render()