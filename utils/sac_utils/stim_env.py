import sys
from machine_learning.RL.env_related import env_utils

import os
import torch
import numpy as np
import pandas as pd
import math
from math import pi
import gymnasium
import gc
from torch.linalg import vector_norm
os.environ['KMP_DUPLICATE_LIB_OK'] ='True'


class MultiFF(gymnasium.Env):
    # The MultiFirefly-Task RL environment

    def __init__(self, 
                 space
                 ):
        

        super().__init__()

        self.space = space
        self.obs_space_length = len(space)
        self.observation_space = gymnasium.spaces.Box(low=-1., high=1., shape=(self.obs_space_length,),dtype=np.float32)


        self.lower_bounds = [dim.low for dim in space]
        self.upper_bounds = [dim.high for dim in space]
        self.stim_bounds = torch.tensor([self.lower_bounds, self.upper_bounds], dtype=torch.float)

        self.objective_func_tensor = partial(obj_func_utils.objective_function, net=net, namespace=namespace, DC_monitor_E=DC_monitor_E, r_E=r_E, r_I=r_I, r_E_sels=r_E_sels,
                                        process_input_func=obj_func_utils.process_input_ibnn, process_output_func=obj_func_utils.process_output_ibnn, E_index_map=E_index_map,
                                        maximize=True)





    def step(self, action):

        # scale the params (which are sampled from -1 to 1) based on stim_bounds
        action = torch.tensor(action)
        params = (action + 1) * (self.lower_bounds - self.upper_bounds) / 2 + self.upper_bounds



        # update training points
        train_x = torch.cat([train_x, params])
        current_y = self.objective_func_tensor(params)
        current_y = current_y.reshape(1, 1)
        train_y = torch.cat([train_y, current_y])

        # Print iteration runtime
        elapsed = time.time() - start_time
        print(f'Iteration {iteration}: total runtime is {elapsed:.2f} s')
        print('Maximum value so far:', train_y.max().item())

        if iteration == 10:
            # change the result folder to the new one
            result_folder = get_new_result_subfolder()

        if iteration % 1 == 0:
            print('Saved updated_x and updated_y as updated_x.pt and updated_y.pt')
            print('Number of training points:', train_x.size(0))
            torch.save(train_x, os.path.join(result_folder, "updated_x.pt"))
            torch.save(train_y, os.path.join(result_folder, "updated_y.pt"))

        if (iteration % 10 == 0) & (iteration >= 10):
            # save a backup
            print('Saved updated_x and updated_y as updated_x_backup.pt and updated_y_backup.pt')
            torch.save(train_x, os.path.join(result_folder, "updated_x_backup.pt"))
            torch.save(train_y, os.path.join(result_folder, "updated_y_backup.pt"))

        print('================================================================================================')
        print('================================================================================================')



        # get reward and scale it
        reward = train_y.item()/1000
        self.episode_reward += reward
        if self.time >= self.episode_len * self.dt:
            self.end_episode = True

        return self.obs, reward, self.end_episode, False, {}












    def reset(self, seed=None, use_random_ff=True):
        
        """
        reset the environment

        Returns
        -------
        obs: np.array
            return an observation based on the reset environment  
        """
        print('TIME before resetting:', self.time)
        super().reset(seed=seed)


        # reset or update other variables
        self.time = 0
        self.num_targets = 0
        self.episode_reward = 0
        self.cost_for_the_current_ff = 0
        self.JUST_CROSSED_BOUNDARY = False
        self.cost_breakdown = {'dv_cost': 0, 'dw_cost': 0, 'w_cost': 0}
        self.reward_for_each_ff = []
        self.end_episode = False
        self.action = np.array([0, 0])
        self.obs = self.beliefs().numpy()
        if self.epi_num > 0:
            print("\n episode: ", self.epi_num)
        self.epi_num += 1
        self.num_ff_caught_in_episode = 0
        info = {}

        return self.obs, info



    def beliefs(self):
        # The beliefs function will be rewritten because the observation no longer has a memory component;
        # The manually added noise to the observation is also eliminated, because the LSTM network will contain noise;
        # Thus, in the environment for LSTM agents, ffxy_noisy is equivalent to ffxy.

        self._get_ff_info()

        # see if any ff is caught; if so, they will be removed from the memory so that the new ff (with a new location) will not be included in the observation
        self._check_for_num_targets()

        self._further_process_after_check_for_num_targets()

        self._get_ff_array_for_belief()

        obs = torch.flatten(self.ff_array.transpose(0, 1))

        # append action to the observation
        if self.add_action_to_obs:
            obs = torch.cat((obs, torch.tensor(self.action)), dim=0)

        self.prev_obs = self.current_obs.clone()
        self.current_obs = obs

        # if any element in obs has its absolute value greater than 1, raise an error
        if torch.any(torch.abs(obs) > 1):
            raise ValueError('The observation has an element with an absolute value greater than 1')

        return obs












    # ========================================================================================================
    # ================== The following functions are helper functions ========================================
    
    def _random_ff_positions(self, ff_index):
        """
        generate random positions for ff

        Parameters
        -------
        ff_index: array-like
            indices of fireflies whose positions will be randomly generated
           
        """
        num_alive_ff = len(ff_index)
        self.ffr[ff_index] = torch.sqrt(torch.rand(num_alive_ff)) * self.arena_radius
        self.fftheta[ff_index] = torch.rand(num_alive_ff) * 2 * pi
        self.ffx[ff_index] = torch.cos(self.fftheta[ff_index]) * self.ffr[ff_index]
        self.ffy[ff_index] = torch.sin(self.fftheta[ff_index]) * self.ffr[ff_index]
        self.ffxy = torch.stack((self.ffx, self.ffy), dim=1)
        # The following variables store the locations of all the fireflies with uncertainties
        self.ffx_noisy[ff_index] = self.ffx[ff_index].clone()
        self.ffy_noisy[ff_index] = self.ffy[ff_index].clone()
        self.ffxy_noisy = torch.stack((self.ffx_noisy, self.ffy_noisy), dim=1)

    
    def _make_observation_space(self, num_elem_per_ff):
        self.obs_space_length = self.num_obs_ff * self.num_elem_per_ff + 2 if self.add_action_to_obs else self.num_obs_ff * self.num_elem_per_ff
        self.observation_space = gymnasium.spaces.Box(low=-1., high=1., shape=(self.obs_space_length,),dtype=np.float32)


    def _get_catching_ff_reward(self):
        catching_ff_reward = self.reward_per_ff * self.num_targets
        if self.add_cost_when_catching_ff_only:
            catching_ff_reward = max(self.reward_per_ff * self.num_targets - self.cost_for_the_current_ff, 0.2 * catching_ff_reward)
        if self.distance2center_cost > 0:
            # At the earlier stage of the curriculum training, the reward gained by catching each firefly will 
            # decrease based on how far away the agent is from the center of the firefly
            total_deviated_distance = torch.sum(self.ff_distance_all[self.captured_ff_index]).item()
            catching_ff_reward = catching_ff_reward - total_deviated_distance * self.distance2center_cost        
        return catching_ff_reward


    def _check_for_num_targets(self):
        self.num_targets = 0
        # If the velocity of the current step is low enough for the action to be considered a stop
        if not self.JUST_CROSSED_BOUNDARY:
            try:
                if (abs(self.sys_vel[0]) <= self.angular_terminal_vel) & (abs(self.sys_vel[1]) <= self.linear_terminal_vel):
                # if (abs(self.sys_vel[0]) <= 0.01) & (abs(self.sys_vel[1]) <= 0.01):
                # if (abs(self.sys_vel[1]) <= 0.01):
                    self.captured_ff_index = (self.ff_distance_all <= self.reward_boundary).nonzero().reshape(-1).tolist()
                    self.num_targets = len(self.captured_ff_index)
                    if self.num_targets > 0:
                        self.catching_ff_reward = self._get_catching_ff_reward()
                        self.ff_memory_all[self.captured_ff_index] = 0
                        self.ff_time_since_start_visible[self.captured_ff_index] = 0
                        # Replace the captured ffs with ffs of new locations
                        self._random_ff_positions(self.captured_ff_index)
                        # need to call get_ff_info again to update the information of the new ffs
                        self._update_ff_info(self.captured_ff_index)
                        
            except AttributeError:
                pass # This is to prevent the error that occurs when sys_vel is not defined in the first step

    def _get_ff_array_given_indices(self, add_memory=True, add_ff_time_since_start_visible=False):
        self.ffxy_topk_noisy = self.ffxy_noisy[self.topk_indices]
        self.distance_topk_noisy = vector_norm(self.ffxy_topk_noisy - self.agentxy, dim=1)
        # cap self.distance_topk_noisy to be less than self.invisible_distance
        self.distance_topk_noisy = torch.minimum(self.distance_topk_noisy, torch.tensor(self.invisible_distance))
        self.angle_to_center_topk_noisy, angle_to_boundary_topk_noisy = env_utils.calculate_angles_to_ff_in_pytorch(self.ffxy_topk_noisy, self.agentx, self.agenty, self.agentheading, 
                                                                                            self.ff_radius, ffdistance=self.distance_topk_noisy)
        if add_memory:
            # Concatenate angles, distance, and memory
            self.ff_array = torch.stack((self.angle_to_center_topk_noisy, angle_to_boundary_topk_noisy, self.distance_topk_noisy, 
                                         self.ff_memory_all[self.topk_indices]), dim=0)
        elif add_ff_time_since_start_visible:
            self.ff_array = torch.stack((self.angle_to_center_topk_noisy, angle_to_boundary_topk_noisy, self.distance_topk_noisy,
                                         self.ff_time_since_start_visible[self.topk_indices]), dim=0)
        else:
            self.ff_array = torch.stack((self.angle_to_center_topk_noisy, angle_to_boundary_topk_noisy, self.distance_topk_noisy), dim=0)
        return self.ff_array



    def _get_ff_array_for_belief_common(self, ff_indices, add_memory, add_ff_time_since_start_visible):
        if torch.numel(ff_indices) >= self.num_obs_ff:
            self.topk_indices = env_utils._get_topk_indices(ff_indices, self.ff_distance_all, self.num_obs_ff)
            self.ff_array = self._get_ff_array_given_indices(add_memory=add_memory, add_ff_time_since_start_visible=add_ff_time_since_start_visible)
        elif torch.numel(ff_indices) == 0:
            self.topk_indices = torch.tensor([])
            self._update_variables_when_no_ff_is_in_obs()
            placeholder_ff = env_utils._get_placeholder_ff(add_memory, add_ff_time_since_start_visible, self.invisible_distance)
            self.ff_array = placeholder_ff.repeat([1, self.num_obs_ff])
        else:
            self.topk_indices = env_utils._get_sorted_indices(ff_indices, self.ff_distance_all)
            self.ff_array = self._get_ff_array_given_indices(add_memory=add_memory, add_ff_time_since_start_visible=add_ff_time_since_start_visible)
            num_needed_ff = self.num_obs_ff - torch.numel(ff_indices)
            placeholder_ff = env_utils._get_placeholder_ff(add_memory, add_ff_time_since_start_visible, self.invisible_distance)
            needed_ff = placeholder_ff.repeat([1, num_needed_ff])
            self.ff_array = torch.cat([self.ff_array.reshape([self.num_elem_per_ff, -1]), needed_ff], dim=1)

        self.ff_array_unnormalized = self.ff_array.clone()
        self.ff_array = env_utils._normalize_ff_array(self.ff_array, self.invisible_distance, self.full_memory, add_memory, add_ff_time_since_start_visible, self.visible_time_range)


    def _further_process_after_check_for_num_targets(self):
        self._update_ff_time_since_start_visible()

    def _update_ff_memory_and_uncertainty(self):
        # update memory of all fireflies
        self.ff_memory_all[self.ff_memory_all > 0] = self.ff_memory_all[self.ff_memory_all > 0] - 1
        if len(self.visible_ff_indices) > 0:
            self.ff_memory_all[self.visible_ff_indices] = self.full_memory

        # for ff whose absolute angle_to_boundary is greater than 90 degrees, make memory 0
        self.ff_memory_all[torch.abs(self.angle_to_boundary_all) > pi/2] = 0
        self.ff_memory_all[self.ff_distance_all > self.invisible_distance] = 0

        # find ffs that are in memory
        self.ff_in_memory_indices = (self.ff_memory_all > 0).nonzero().reshape(-1)

        # calculate the std of the uncertainty that will be added to the distance and angle of each firefly
        ff_uncertainty_all = np.sign(self.full_memory - self.ff_memory_all) * (self.ffxy_noise_std * self.dt) * np.sqrt(self.ff_distance_all)
        # update the positions of fireflies with uncertainties added; note that uncertainties are cummulative across steps
        self.ffx_noisy, self.ffy_noisy, self.ffxy_noisy = env_utils.update_noisy_ffxy(self.ffx_noisy, self.ffy_noisy, self.ffx, self.ffy, ff_uncertainty_all, self.visible_ff_indices)
        return

    def _update_variables_when_no_ff_is_in_obs(self):
        self.ffxy_topk_noisy = torch.tensor([])
        self.distance_topk_noisy = torch.tensor([])
        self.angle_to_center_topk_noisy = torch.tensor([])


    def _get_ff_info(self):
        try:
            self.prev_visible_ff_indices = self.visible_ff_indices.clone()
        except AttributeError:
            self.prev_visible_ff_indices = torch.tensor([])
        
        self.ff_distance_all = vector_norm(self.ffxy - self.agentxy, dim=1)
        self.angle_to_center_all, self.angle_to_boundary_all = env_utils.calculate_angles_to_ff_in_pytorch(self.ffxy, self.agentx, self.agenty, self.agentheading, 
                                                                            self.ff_radius, ffdistance=self.ff_distance_all)
        self.visible_ff_indices = env_utils.find_visible_ff(self.time, self.ff_distance_all, self.angle_to_boundary_all, self.invisible_distance, self.invisible_angle, self.ff_flash)
        return
    

    def _update_ff_info(self, ff_index):
        self.ff_distance_all[ff_index] = vector_norm(self.ffxy[ff_index] - self.agentxy, dim=1)
        self.angle_to_center_all[ff_index], self.angle_to_boundary_all[ff_index] = env_utils.calculate_angles_to_ff_in_pytorch(self.ffxy[ff_index], self.agentx, self.agenty, self.agentheading,
                                                                                                    self.ff_radius, ffdistance=self.ff_distance_all[ff_index])
        # find visible ff among the updated ff
        self.visible_ff_indices_among_updated_ff = env_utils.find_visible_ff(self.time, self.ff_distance_all[ff_index], self.angle_to_boundary_all[ff_index], self.invisible_distance, self.invisible_angle, 
                                                                             [self.ff_flash[i] for i in ff_index])
        # delete from the visible_ff_indices the indices that are in ff_index
        self.visible_ff_indices = torch.tensor([i for i in self.visible_ff_indices if i not in ff_index])
        # concatenate the visible_ff_indices_among_updated_ff to the visible_ff_indices
        self.visible_ff_indices = torch.cat((self.visible_ff_indices, self.visible_ff_indices_among_updated_ff), dim=0)
        # change dtype to int
        self.visible_ff_indices = self.visible_ff_indices.int()

        return



    def _update_ff_time_since_start_visible_base_func(self, not_visible_ff_indices):
        self.ff_time_since_start_visible += self.dt
        self.ff_time_since_start_visible[not_visible_ff_indices] = 0

        # get ff that has turned from not visible to visible
        self.newly_visible_ff = torch.tensor(list(set(self.visible_ff_indices) - set(self.prev_visible_ff_indices)), dtype=torch.int)
        self.ff_time_since_start_visible[self.newly_visible_ff] = self.dt


    def _update_ff_time_since_start_visible(self):
        self.ff_not_visible_indices = torch.tensor(list(set(range(self.num_alive_ff)) - set(self.visible_ff_indices.tolist())), dtype=torch.int)
        self._update_ff_time_since_start_visible_base_func(self.ff_not_visible_indices)
