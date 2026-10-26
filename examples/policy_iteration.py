'''
Author: CYRUS 2241855671@qq.com
Date: 2026-10-22 00:15:41
Description: 

Copyright (c) 2026 by HITCRT, All Rights Reserved.
'''

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.grid_world import GridWorld
import random
import numpy as np

#策略迭代   先堆一个策略执行很多次，然后再选择策略
def policy_iteration(env: GridWorld, gamma=0.9, eval_tol=1e-4):
    # 初始化:V 全 0,π 每个状态随便指个动作
    V = np.zeros(env.num_states)  # 初始化 V[s] = 0
    policy_matrix = np.zeros((env.num_states, len(env.action_space)))
    term_idx = env.target_state[0] + env.target_state[1] * env.env_size[0]  # 终点下标:V 恒为 0,不评估不改进
    # 主循环:
    #   策略评估:反复 V[s] ← r + γ·V[s'],直到收敛
    #   策略改进:贪心选 Q 最大的动作,没人变化就 break
    for t in range(300):   
            V_new = np.zeros(env.num_states)
            for s in range(env.num_states):
                # if s == term_idx:      # 终点:V 恒为 0,跳过
                #     continue
                x = s % env.env_size[0]
                y = s // env.env_size[0]
                #对于每一个状态遍历所有的action 计算出Q值 然后和policy_matrix结合 进行value的迭代
                for a in range(len(env.action_space)):
                    (next_statex, next_statey), r = env._get_next_state_and_reward((x, y), env.action_space[a])
                    V_new[s] =  policy_matrix[s][a] * (r + gamma * V[next_statex + next_statey * env.env_size[0]]) + V_new[s]     
            
            if(np.max(np.abs(V_new - V)) < eval_tol): #策略更新
                #根据Q值更新policy_matrix 对于状态遍历  计算出每一个action的Q值  然后选择最大的那个action作为policy_matrix的值
                stable = True   #这一轮改进有没有状态换动作
                for s in range(env.num_states):
                    # if s == term_idx:      # 终点没有动作可言,跳过
                    #     continue
                    x = s % env.env_size[0]
                    y = s // env.env_size[0]
                    max_action_value = -np.inf  #最优Q数值初始化为负无穷
                    best_action = None   #最大的Q序列的值
                    for a in range(len(env.action_space)):
                        (next_statex, next_statey), r = env._get_next_state_and_reward((x, y), env.action_space[a])
                        if(max_action_value < r + gamma * V_new[next_statex + next_statey * env.env_size[0]]):
                            max_action_value = r + gamma * V_new[next_statex + next_statey * env.env_size[0]]
                            best_action = a
                    if best_action != int(np.argmax(policy_matrix[s])):  # 和原动作比,变了就标记
                        stable = False
                    policy_matrix[s] = np.zeros(len(env.action_space)) #将所有的动作的概率初始化为0
                    policy_matrix[s][best_action] = 1
                if stable:
                    break   # 策略一个都没变 → 已最优,收工
            V = V_new
    return policy_matrix, V          # 返回结果给外面画图

if __name__ == "__main__":
    env = GridWorld()
    policy_matrix, values = policy_iteration(env)

    env.reset()
    env.render()
    env.add_policy(policy_matrix)
    env.add_state_values(values)

    # 按最优策略从起点走到终点(验收环节)
    done = False
    while not done:
        env.render(animation_interval=0.5)              # 每步停半秒,看清移动
        s = env.agent_state                             # 当前位置 (x, y)
        idx = s[0] + s[1] * env.env_size[0]             # 换算成扁平下标
        a = int(np.argmax(policy_matrix[idx]))          # 查策略:one-hot 行里那个 1
        next_state, reward, done, _ = env.step(env.action_space[a])   # 真走一步
    env.render(animation_interval=0.5)                  # 补一帧:星号停在终点
    input('看完按回车退出...') 
    

