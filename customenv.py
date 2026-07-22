from typing import Optional
import numpy as np
import gymnasium as gym
import pygame
from gymnasium.envs.registration import register
import random 

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, render_mode=None, size: int = 5):
        self.window_size = 512
        self.size = size
        self.num_agents = 2
        self._agent_locations = np.array([[-1, -1], [-1, -1]], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)
        self._vertiport_locations = np.array([[-1, -1],[-1, -1],[-1, -1],[-1, -1]], dtype=np.int32)
        self._passenger_path = np.array([[[-1, -1], [-1, -1]], [[-1, -1], [-1, -1]]], dtype=np.int32)
        self.cost = [0, 0]
        self.revenue = [0, 0]
        self.profit = [0, 0]
        self.queue_length = 1000
        self._passenger_cap = 2
        self.observation_space = gym.spaces.Dict(
            {
                "agent_locations": gym.spaces.Box(0, size - 1, shape=(2, 2), dtype=int),   
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "vertiport_locations": gym.spaces.Box(0, size - 1, shape=(4,2), dtype=int),
                "passenger_path": gym.spaces.Box(0, size - 1, shape=(self.queue_length, 2, 2), dtype=int)
            }
        )

        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([0, 1]),   
            1: np.array([-1, 0]),  
            2: np.array([0, -1]),  
            3: np.array([1, 0]),
            }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.unique_vis_1 = []
        self.unique_vis_2 = []
        self.flags = []
        self.randintlist = []

    def _action_to_state(self, location):
        return((location[0] * self.size) + location[1])
    def _state_to_action(self, state):
        return([(int(state/self.size)), (state%self.size)])


    def _get_obs(self):
        return np.concatenate([
    np.array([
        self._action_to_state(self._agent_locations[0]),
        self._action_to_state(self._agent_locations[1]),
    ]),
    np.array([
        self._action_to_state(v)
        for v in self._vertiport_locations
    ]),
    np.array([
        self._action_to_state(x)
        for path in self._passenger_path
        for x in path
        ])
])
    def _get_info(self):

        return {
            "distance": 0
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.cost = [0, 0]
        self.revenue = [0, 0]
        self.distance_carried = [0, 0]
        self.pickedup1 = [0, 0]
        self.pickedup2 = [0, 0]
        self.droppedoff1 = [0, 0]
        self.droppedoff2 = [0, 0]
        self.randintlist = []
        self.count = [0, 0]
        self.vertlocs = [4, 12, 16, 24]
        while(len(self.randintlist) < 4):
            x = random.randint(0,3)
            if x not in self.randintlist:
                self.randintlist.append(x)
        self._passenger_path = np.array([[self._state_to_action(self.vertlocs[self.randintlist[0]]), self._state_to_action(self.vertlocs[self.randintlist[1]])], [self._state_to_action(self.vertlocs[self.randintlist[2]]), self._state_to_action(self.vertlocs[self.randintlist[3]])]], dtype=int)
        self._agent_locations = np.array([self._state_to_action(0), self._state_to_action(6)], dtype=int)
        self._vertiport_locations = np.array([self._state_to_action(4), self._state_to_action(12), self._state_to_action(16), self._state_to_action(24)], dtype=int)
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _demand_generation(self):
        flip = random.randint(0, 10)
        randval1 = random.randint(0, 3)
        randval2 = 0
        while randval2 == randval1:
            randval2 = random.randint(0, 3)
        if flip == 1:
            self._passenger_path = np.vstack((self._passenger_path, np.array([[self._state_to_action(self.vertlocs[randval1]), self._state_to_action(self.vertlocs[randval2])]])))
            self.pickedup1.append(0)
            self.droppedoff1.append(0)
            self.pickedup2.append(0)
            self.droppedoff2.append(0)


    def step(self, action):
        self._demand_generation()
        rewards = [-0.1, -0.1]
        direction = [0, 0]
        direction[0] = self._action_to_direction[action[0]]
        self._agent_locations[0] = np.clip(
            self._agent_locations[0] + direction[0], 0, self.size - 1
            )
        direction[1] = self._action_to_direction[action[1]]
        self._agent_locations[1] = np.clip(
            self._agent_locations[1] + direction[1], 0, self.size - 1
            )
            
        #for i in range(4):
        terminated = False
#        for i in range(4):
#            if np.array_equal(self._vertiport_locations[i], self._agent_locations[0]):
#                if(self._action_to_state(self._vertiport_locations[i]) not in self.unique_vis_1):
#                    self.count[0] += 1
#                    #print(self._action_to_state(self._vertiport_locations[i]))
#                    self.unique_vis_1.append(self._action_to_state(self._vertiport_locations[i]))
#                    rewards[0] += 1                   
#
#        for i in range(4):
#            if np.array_equal(self._vertiport_locations[i], self._agent_locations[1]):
#                if(self._action_to_state(self._vertiport_locations[i]) not in self.unique_vis_2):
#                    self.count[1] += 1
#                    #print(self._action_to_state(self._vertiport_locations[i]))
#                    self.unique_vis_2.append(self._action_to_state(self._vertiport_locations[i]))

        for i in range(len(self._passenger_path)): 
            if np.array_equal(self._agent_locations[0], self._passenger_path[i][0]) and (self.pickedup1[i] == 0) and (self.pickedup2[i] == 0) and (self.droppedoff1[i] == 0) and (self.droppedoff2[i] == 0) and (sum(self.pickedup1) < 3):
                self.pickedup1[i] = 1
        for i in range(len(self._passenger_path)): 
            if self.pickedup1[i] == 1 and np.array_equal(self._agent_locations[0], self._passenger_path[i][1]):
                self.droppedoff1[i] = 1
                self.pickedup1[i] = 0
                rewards[0] += 1
            

        for i in range(len(self._passenger_path)): 
            if np.array_equal(self._agent_locations[1], self._passenger_path[i][0]) and (self.pickedup2[i] == 0) and (self.pickedup1[i] == 0) and (self.droppedoff1[i] == 0) and (self.droppedoff2[i] == 0) and (sum(self.pickedup2) < 3):
                self.pickedup2[i] = 1
        for i in range(len(self._passenger_path)): 
            if self.pickedup2[i] == 1 and np.array_equal(self._agent_locations[1], self._passenger_path[i][1]):
                self.droppedoff2[i] = 1
                self.pickedup2[i] = 0
                rewards[1] += 1
        
        self.count[0] = sum(self.droppedoff1)
        self.count[1] = sum(self.droppedoff2)
        if sum(self.count) >= 3:
            print(self.count)
            terminated = True

        truncated = False
        
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, rewards, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        # Convert [row, col] to pygame (x, y) by reversing the coordinates
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * (self._vertiport_locations[0])[::-1],
                (pix_square_size, pix_square_size),
            ),
        )
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * (self._vertiport_locations[1])[::-1],
                (pix_square_size, pix_square_size),
            ),
        )
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * (self._vertiport_locations[2])[::-1],
                (pix_square_size, pix_square_size),
            ),
        )
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * (self._vertiport_locations[3])[::-1],
                (pix_square_size, pix_square_size),
            ),
        )
        pygame.draw.rect(
            canvas,
            (255, 255, 0),
            pygame.Rect(
                pix_square_size * (self._passenger_path[0][0])[::-1],
                (pix_square_size, pix_square_size),
            ),
        )
        pygame.draw.rect(
            canvas,
            (255, 0, 255),
            pygame.Rect(
                pix_square_size * (self._passenger_path[0][1])[::-1],
                (pix_square_size, pix_square_size),
            ),
        )
        pygame.draw.rect(
            canvas,
            (255, 255, 0),
            pygame.Rect(
                pix_square_size * (self._passenger_path[1][0])[::-1],
                (pix_square_size, pix_square_size),
            ),
        )
        pygame.draw.rect(
            canvas,
            (255, 0, 255),
            pygame.Rect(
                pix_square_size * (self._passenger_path[1][1])[::-1],
                (pix_square_size, pix_square_size),
            ),
        )

        
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_locations[0][::-1] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        pygame.draw.circle(
            canvas,
            (0, 255, 0),
            (self._agent_locations[1][::-1] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

