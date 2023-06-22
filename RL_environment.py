from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess
import datatracker
import numpy as np
import random
import tf_agents

from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tensorflow.python.ops.numpy_ops import np_config

from Global_parameters import gp
from Environment_parser import Environment_parser

np_config.enable_numpy_behavior()


class HandoverEnv(py_environment.PyEnvironment):

    def __init__(self, eval1=False, eval2=False):
        self.env_parser = Environment_parser()
        self._action_spec = tf_agents.specs.BoundedArraySpec(
            shape=(2,), dtype=np.float32, minimum=0, maximum=34, name='action')
        self._observation_spec = tf_agents.specs.BoundedArraySpec(
            shape=(gp.all_sate,), dtype=np.float64, minimum=np.full((gp.all_sate, ), -1000), maximum=np.full((gp.all_sate, ), 30000), name='observation')
        self.eval_env = eval1
        self.eval_env2 = eval2
        self.NeighbourCellOffset = 5
        self.ServingCellThreshold = 30
        self.duration = 60
        self.durration_in_ms = self.duration * 1000
        self.UE_Count = 8
        self.min_speed = 70
        self.max_speed = 70
        self._state = []
        self._state.extend(np.zeros([gp.all_count*gp.ENB_Count]))
        self._state.append(self.UE_Count/gp.UE_upper_count)
        self._state.append(self.max_speed)
        self._state.append(self.min_speed)
        self._state.append(self.duration)
        self._episode_ended = False
        self.environment_called = 1
        self.episode = 0
        self.x_pos = 0
        self.y_pos = 300
        self.rho = 200

    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self.NeighbourCellOffset = 5
        self.ServingCellThreshold = 30
        
        self._state = []
        self._state.extend(np.zeros([gp.all_count*gp.ENB_Count]))
        self._state.append(self.UE_Count/gp.UE_upper_count)
        self._state.append(self.max_speed)
        self._state.append(self.min_speed)
        self._state.append(self.duration)

        self.environment_called = 1
        return ts.restart(np.array(self._state))

    def _step(self, action):
        
        if self._episode_ended:
        # The last action ended the episode. Ignore the current action and start
        # a new episode.
            return self.reset()

        # Make sure episodes don't go on forever.
        if self.environment_called >= 1:
            self._episode_ended = True
            self.environment_called = 1
        else:
            self.environment_called += 1
            
        if self.eval_env:
            self.duration = 70
            self.durration_in_ms = self.duration * 1000
            self.UE_Count = 8
            self.min_speed = 20
            self.max_speed = 20
            eval_max_speed = 20
            eval_min_speed = 20
            self.x_pos = 200
            self.y_pos = 300
            self.rho = 200
            rng = 100
            
            print('\nEvaluation Environment1')
            print('duration', self.duration, 'UE_Count', self.UE_Count)
            print('min_speed', self.min_speed, 'max_speed', self.max_speed, 'x_pos',
                    self.x_pos, 'y_pos', self.y_pos, 'rho', self.rho)
            print()
            gp.eval_actions.append(action)
            gp.eval_action__ = action
            gp.eval_throughput_to_save = []
            gp.eval_rsrq_to_save = []
            gp.eval_cell_rsrqs = []
            gp.eval_cell_throughputs = []
            
        elif self.eval_env2:
            self.duration = 80
            self.durration_in_ms = self.duration * 1000
            self.UE_Count = 8
            self.min_speed = 70
            self.max_speed = 70
            eval2_max_speed = 70
            eval2_min_speed = 70
            self.x_pos = 300
            self.y_pos = 100
            self.rho = 200
            rng = 350
            
            print('\nEvaluation Environment2')
            print('duration', self.duration, 'UE_Count', self.UE_Count)
            print('min_speed', self.min_speed, 'max_speed', self.max_speed, 'x_pos',
                    self.x_pos, 'y_pos', self.y_pos, 'rho', self.rho)
            print()
            gp.eval2_actions.append(action)
            gp.eval2_action__ = action
            gp.eval2_throughput_to_save = []
            gp.eval2_rsrq_to_save = []
            gp.eval2_cell_rsrqs = []
            gp.eval2_cell_throughputs = []
            
        else:
            rng = random.randint(0, 1000)
            if self.episode % 100 == 0:
                self.duration = random.choice(list(range(60, 91, 10)))
                gp.duration = self.duration
                self.durration_in_ms = self.duration * 1000
                self.UE_Count = random.choice([7, 8, 9])
                gp.ues = self.UE_Count
                print('duration', self.duration, 'UE_Count', self.UE_Count)
            if self.episode % 50 == 0:
                self.min_speed = random.choice([20, 40, 70])
                self.max_speed = self.min_speed
                gp.max_speed = self.min_speed
                gp.min_speed = self.min_speed
                self.x_pos = random.choice(list(range(0, 601, 100)))
                self.y_pos = random.choice(list(range(0, 601, 100)))
                self.rho = 200
                print('min_speed', self.min_speed, 'max_speed', self.max_speed, 'x_pos',
                    self.x_pos, 'y_pos', self.y_pos, 'rho', self.rho)
            gp.actions.append(action)
            gp.action__ = action
            gp.throughput_to_save = []
            gp.rsrq_to_save = []
            gp.cell_rsrqs = []
            gp.cell_throughputs = []
                
        self.NeighbourCellOffset = action[0]
        self.ServingCellThreshold = action[1]
        
        myoutput = open(gp.events_file_name, 'w')
        subprocess.run(["./simulator", "--NeighbourCellOffset="+str(self.NeighbourCellOffset), "--ServingCellThreshold="+str(self.ServingCellThreshold), 
                        "--duration="+str(self.duration), "--UE_Count="+str(self.UE_Count), "--ENB_Count="+str(gp.ENB_Count)
                    , "--x_pos="+str(self.x_pos), "--rho="+str(self.rho), "--y_pos="+str(self.y_pos)
                    , "--max_speed="+str(self.max_speed), "--min_speed="+str(self.max_speed), "--RngRun="+str(rng)]
                    , stdout=myoutput)
        
        myoutput.close() # close the file
            
        data = datatracker.Data()
        self.env_parser.parse_document(data, gp.events_file_name)
        ue_cell_connection, cell_connected_ue = self.env_parser.find_cell_ue_dicts(data.data, self.durration_in_ms)
        cell_connected_ue, cell_info = self.env_parser.Add_more_cell_info(data.data, cell_connected_ue)
        
        self._state = self.env_parser.find_state(cell_info, ue_cell_connection, self.UE_Count, self.duration,
                                self.eval_env, self.eval_env2, self.max_speed, self.min_speed)


        if self._episode_ended:
            self.episode += 1
            reward = self.env_parser.find_reward(data, cell_connected_ue, self.duration, self.UE_Count, self.eval_env, self.eval_env2)
            if self.eval_env:
                gp.eval_reward_ = reward
            elif self.eval_env2:
                gp.eval2_reward_ = reward
            else:
                gp.reward_ = reward
            return ts.termination(np.array(self._state), reward)
        else:
            return ts.transition(
            np.array(self._state), reward=0.0, discount=1.0)