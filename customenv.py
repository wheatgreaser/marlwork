from typing import Optional
import numpy as np
import gymnasium as gym
import pygame
from gymnasium.envs.registration import register

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, render_mode=None, size: int = 5):
        self.window_size = 512
        self.size = size
        self.num_agents = 2
        self._agent_locations = np.array([[-1, -1], [-1, -1]], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)
        self._vertiport_locations = np.array([[-1, -1],[-1, -1],[-1, -1],[-1, -1]], dtype=np.int32)
        self._passenger_path = np.array([[-1, -1], [-1, -1]], dtype=np.int32)

        self.observation_space = gym.spaces.Dict(
            {
                "agent_locations": gym.spaces.Box(0, size - 1, shape=(2, 1), dtype=int),   
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "vertiport_locations": gym.spaces.Box(0, size - 1, shape=(4,1), dtype=int),
                "passenger_path": gym.spaces.Box(0, size - 1, shape=(2, 1), dtype=int)
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
        self.count = [0, 0]


    def _action_to_state(self, location):
        return((location[0] * self.size) + location[1])
    def _state_to_action(self, state):
        return([(int(state/self.size)), (state%self.size)])

    def _get_obs(self):
        #return {"agent": self._agent_location, "target": self._target_location}
        return {"agent": [self._action_to_state(self._agent_locations[0]), self._action_to_state(self._agent_locations[1])], "vertiport_locations": [self._action_to_state(self._vertiport_locations[0]), self._action_to_state(self._vertiport_locations[1]), self._action_to_state(self._vertiport_locations[2]), self._action_to_state(self._vertiport_locations[3]) ], "passenger_origin": self._action_to_state(self._passenger_path[0]), "passenger_destination": self._action_to_state(self._passenger_path[1])}

    def _get_info(self):

        return {
            "distance": 0
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.count = [0, 0, 0, 0]
        self.unique_vis_1 = []
        self.unique_vis_2 = []
        self._passenger_path = np.array([self._state_to_action(4), self._state_to_action(24)], dtype=int)
        self._agent_locations = np.array([self._state_to_action(0), self._state_to_action(6)], dtype=int)
        self._vertiport_locations = np.array([self._state_to_action(4), self._state_to_action(11), self._state_to_action(14), self._state_to_action(24)], dtype=int)

        self._target_location = np.array(self._state_to_action(10), dtype= int)
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        rewards = [0, 0]
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
#                    rewards[1] += 1                    
        
        if np.array_equal(self._agent_locations[0], self._passenger_path[0]):
            self.count[0] = 1
            rewards[0] += 0.5
        if self.count[0] == 1 and np.array_equal(self._agent_locations[0], self._passenger_path[1]):
            self.count[1] = 1
            rewards[0] += 1
        else:
            rewards[0] = -0.1
        if np.array_equal(self._agent_locations[1], self._passenger_path[0]):
            self.count[2] = 1
            rewards[1] += 0.5
        if self.count[2] == 1 and np.array_equal(self._agent_locations[1], self._passenger_path[1]):
            self.count[3] = 1
            rewards[1] += 1
        else:
            rewards[1] = -0.1

        if self.count[1] >= 1:
            terminated = True
        if self.count[3] >= 1:
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
                pix_square_size * (self._passenger_path[0])[::-1],
                (pix_square_size, pix_square_size),
            ),
        )
        pygame.draw.rect(
            canvas,
            (255, 0, 255),
            pygame.Rect(
                pix_square_size * (self._passenger_path[1])[::-1],
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

