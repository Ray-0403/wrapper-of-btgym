
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym import spaces
import gym

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import backtrader as bt
from btgym import BTgymDataset, BTgymBaseStrategy, BTgymEnv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from pyts.image import GADF
from btgym.research.strategy_gen_4 import DevStrat_4_11,  DevStrat_4_12
from btgym.research.strategy_gen_6.base import BaseStrategy6
from btgym.strategy.observers import Reward, Position, NormPnL

import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env



class wrapper_btgym_env(gym.Env):
    
    def __init__(self):
        
        self.MyCerebro = bt.Cerebro()
        
        self.MyCerebro.addstrategy(
               DevStrat_4_11,
               start_cash=2000,  # initial broker cash
               commission=0.0001,  # commisssion to imitate spread
               leverage=10.0,
               order_size=2000,  # fixed stake, mind leverage
               drawdown_call=10,# max % to loose, in percent of initial cash
               target_call=10,  # max % to win, same
               skip_frame=10,
               gamma=0.99,
               reward_scale=7, # gardient`s nitrox, touch with care!
               state_ext_scale = np.linspace(3e3, 1e3, num=5)
               )
        
        self.MyCerebro.addobserver(Reward)
        self.MyCerebro.addobserver(Position)
        self.MyCerebro.addobserver(NormPnL)
        
        self.MyDataset = BTgymDataset(
               filename="/Users/bluecharles/Desktop/data/DAT_ASCII_EURUSD_M1_2016.csv",
               start_weekdays=[0,1,2,3,4],
               start_00=True,
               episode_duration={'days':0, 'hours':23, 'minutes': 55},
               time_gap={'hours': 5},
               )
        
        
        self._env = BTgymEnv(
                      dataset=self.MyDataset,
                      engine=self.MyCerebro,
                      port=5555,
                      render_enabled=False,
                      verbose=0,
                      )
        time_dim = 30 
        avg_period = 20
        
        self.observation_space = spaces.Dict({
            'external': spaces.Box(low=-100, high=100, shape=(time_dim, 1, 5), dtype=np.float32),
            'internal': spaces.Box(low=-2, high=2, shape=(avg_period, 1, 6), dtype=np.float32),
            'metadata': spaces.Dict(
                {
                    'type': spaces.Box(
                        shape=(),
                        low=0,
                        high=1,
                        dtype=np.uint32
                    ),
                    'trial_num': spaces.Box(
                        shape=(),
                        low=0,
                        high=10 ** 10,
                        dtype=np.uint32
                    ),
                    'trial_type': spaces.Box(
                        shape=(),
                        low=0,
                        high=1,
                        dtype=np.uint32
                    ),
                    'sample_num': spaces.Box(
                        shape=(),
                        low=0,
                        high=10 ** 10,
                        dtype=np.uint32
                    ),
                    'first_row': spaces.Box(
                        shape=(),
                        low=0,
                        high=10 ** 10,
                        dtype=np.uint32
                    ),
                    'timestamp': spaces.Box(
                        shape=(),
                        low=0,
                        high=np.finfo(np.float64).max,
                        dtype=np.float64
                    ),
                }
            )
        })
        
        
        self.action_space = spaces.Discrete(4)
        
    def reset(self):   
        self.obs = self._env.reset()
        return self.obs
    
    def step(self, action):
        self.a = self._env.step(action)
        return self.a
    
    
   
env = wrapper_btgym_env()

num_episodes = 1

for episode in range(num_episodes):
    init_state = env.reset()
    
    while True:
        random_action = env.action_space.sample()
        obs, reward, done, info = env.step(random_action)
        print('observation space:', env.observation_space)
        print('obs:', env.step(random_action))
        print('action space:', env.action_space)
        #print('reward:', reward)
        print('action:', random_action)
        print(obs)
        if done: break
    
obs = env.reset()
print('initial observation:', obs)
env.close()
   

'''  
ray.shutdown()    
ray.init()

register_env("env", lambda kwargs: wrapper_btgym_env(**kwargs))


config = DEFAULT_CONFIG.copy()
config['num_workers'] = 3
config['num_sgd_iter'] = 30
config['sgd_minibatch_size'] = 128
config['model']['fcnet_hiddens'] = [100, 100]
config['num_cpus_per_worker'] = 0
config['vf_share_layers'] = True

agent = PPOTrainer(config, 'env')

for i in range(10):
    result = agent.train()
    print(pretty_print(result))
    
checkpoint_path = agent.save()
print(checkpoint_path)   
'''





