from __future__ import annotations


import math
import operator
from functools import reduce
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import logger, spaces
from gymnasium.core import ActionWrapper, ObservationWrapper, ObsType, Wrapper

from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
from minigrid.core.world_object import Goal

class RGBImgObsWrapper(ObservationWrapper):
    def __init__(self, env, agent_view_size, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size
        self.agent_view_size = agent_view_size  # add agent_view_size

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size * tile_size, self.agent_view_size * tile_size, 3),
            dtype="uint8",
        )

        # update observation space
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

        # get RGB color map from RGBImgObsWrapper
        self.color_map = {
            0: np.array([255, 0, 0]),
            1: np.array([0, 255, 0]),
            2: np.array([0, 0, 255]),
            3: np.array([112, 39, 195]),
            4: np.array([255, 255, 0]),
            5: np.array([100, 100, 100]),
        }

    def observation(self, obs):
        # get the image from the observation
        img = obs['image']

        # apply the color map
        rgb_img = self.apply_color_map(img)

        return {**obs, 'image': rgb_img}

    def apply_color_map(self, img):
        # create an empty image with the same size as img
        rgb_img = np.zeros_like(img)

        # for each color in the color map
        for color, rgb in self.color_map.items():
            # find the pixels in the image that match this color
            mask = img == color

            # set these pixels to the corresponding RGB color
            rgb_img[mask] = rgb

        return rgb_img




class ViewSizeWrappe1(ObservationWrapper):
    """
    Wrapper to customize the agent field of view size.
    This cannot be used with fully observable wrappers.

    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import ViewSizeWrapper
        >>> env = gym.make("MiniGrid-LavaCrossingS11N5-v0")
        >>> obs, _ = env.reset()
        >>> obs['image'].shape
        (7, 7, 3)
        >>> env_obs = ViewSizeWrapper(env, agent_view_size=5)
        >>> obs, _ = env_obs.reset()
        >>> obs['image'].shape
        (5, 5, 3)
    """

    def __init__(self, env, agent_view_size=7):
        super().__init__(env)

        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3

        self.agent_view_size = agent_view_size

        # Compute observation space with specified view size
        new_image_space = gym.spaces.Box(
            low=0, high=255, shape=(agent_view_size, agent_view_size, 3), dtype="uint8"
        )

        # Override the environment's observation spaceexit
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )
        
    def observation(self, obs):
        env = self.unwrapped

        grid, vis_mask = env.gen_obs_grid(self.agent_view_size)

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)
        
        # Get the agent's position
        agent_pos = env.agent_pos

        return {**obs, "image": image, "agent_pos": agent_pos}  # Return the agent's position
    
    
class MyViewSizeWrapper(ObservationWrapper):
    def __init__(self, env, agent_view_size):
        super().__init__(env)

        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3

        self.agent_view_size = agent_view_size

        # Compute observation space with specified view size
        new_image_space = gym.spaces.Box(
            low=0, high=255, shape=(agent_view_size, agent_view_size, 3), dtype="uint8"
        )

        # Override the environment's observation spaceexit
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        rgb_img = self.get_frame(highlight=False, tile_size=self.tile_size)
        
        env = self.unwrapped

        full_grid_image = env.grid.encode()

        grid, vis_mask = env.gen_obs_grid(self.agent_view_size)

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        # Create a highlight mask for the agent's view area
        highlight_mask = np.zeros((env.width, env.height), dtype=bool)
        for i in range(-self.agent_view_size // 2 + 1, self.agent_view_size // 2 + 1):
            for j in range(-self.agent_view_size // 2 + 1, self.agent_view_size // 2 + 1):
                # Calculate the coordinates based on the agent's direction
                if env.agent_dir == 0:  # left
                    y, x = env.agent_pos[0] + j + 1, env.agent_pos[1] + i
                elif env.agent_dir == 1:  # down
                    y, x = env.agent_pos[0] + j, env.agent_pos[1] + i + 1
                elif env.agent_dir == 2:  # right
                    y, x = env.agent_pos[0] + j - 1, env.agent_pos[1] + i
                else:  # up
                    y, x = env.agent_pos[0] + j, env.agent_pos[1] + i - 1

                # Check if the coordinates are within the grid
                if 0 <= x < env.width and 0 <= y < env.height:
                    highlight_mask[x, y] = True

        # Divide the full grid image into 8x8 blocks
        block_size = 32
        blocks = np.array([
            rgb_img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            for i in range(env.width)
            for j in range(env.height)
        ]).reshape(env.width, env.height, block_size, block_size, 3)

        # Apply the highlight mask to the blocks
        for i in range(env.width):
            for j in range(env.height):
                if highlight_mask[i, j]:
                    blocks[i, j] = self.highlight_img(blocks[i, j])

        # Combine the blocks back into a full grid image
        env_image = np.concatenate([np.concatenate(row, axis=1) for row in blocks], axis=0)

        return {**obs, "image": image, "env_image": env_image}

    @staticmethod
    def highlight_img(img, color=(255, 255, 255), alpha=0.30):
        """
        Add highlighting to an image
        """

        # Create a highlight overlay of the specified color
        highlight_overlay = np.full(img.shape, color, dtype=np.uint8)

        # Blend the image with the highlight overlay
        blend_img = img + alpha * (highlight_overlay - img)
        blend_img = blend_img.clip(0, 255).astype(np.uint8)

        return blend_img


