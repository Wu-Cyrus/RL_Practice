'''
Author: CYRUS 2241855671@qq.com
Date: 2026-10-20 11:13:03
Description: 

Copyright (c) 2026 by HITCRT, All Rights Reserved.
'''

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.grid_world import GridWorld
import random
import numpy as np

# Example usage:
if __name__ == "__main__":             
    env = GridWorld()
    state = env.reset()               
    for t in range(1000):
        env.render()
        action = random.choice(env.action_space)
        next_state, reward, done, info = env.step(action)
        print(f"Step: {t}, Action: {action}, State: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")
        # if done:
        #     break
    
    # Add policy
    policy_matrix=np.random.rand(env.num_states,len(env.action_space))                                            
    policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]  # make the sum of elements in each row to be 1

    env.add_policy(policy_matrix)

    
    # Add state values
    values = np.random.uniform(0,10,(env.num_states,))
    env.add_state_values(values)

    # Render the environment
    env.render(animation_interval=2)