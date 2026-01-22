# model free ε-greedy

import numpy as np
from tqdm import tqdm

#5x5的网格,(1,1)(1,2)(2,2)(3,1)(3,3)(4,1)#障碍物,(3，2)终点

obstacles=[(1,1),(1,2),(2,2),(3,1),(3,3),(4,1)]
target=(3,2)

#所有状态
states=[(i,j) for i in range(5) for j in range(5)]

#所有动作
actions=['up','down','left','right','stay']

#discount rate
gamma=0.9

#采样次数
num_episodes=1000

#ε
epsilon=0.5

#采样长度
max_steps=10000

np.random.seed(42)

#状态转转移函数
def state_transition(state,action):
    x,y=state
    if action=='up':
        x=max(0,x-1)
    elif action=='down':
        x=min(4,x+1)
    elif action=='left':
        y=max(0,y-1)
    elif action=='right':
        y=min(4,y+1)
    return (x,y)

#奖励函数
def reward(state,action,next_state):
    #走出边界
    if next_state==state and action!='stay':
        return -1
    #障碍物
    if next_state in obstacles:
        return -10
    #终点
    if next_state==target:
        return 1
    #普通点
    return 0

def get_action_with_epsilon_greedy(state,policy):
    
    #获得当前状态下每个动作的概率
    prob_list=[policy[state][a] for a in actions]
    
    action=np.random.choice(actions,p=prob_list)
    
    return action

def update_policy(state,Q,policy):
    
    #获得当前状态下每个动作的action_value
    q_values=[ Q[state][a] for a in actions]
    
    #找到action_value最大的动作
    a_start=actions[np.argmax(q_values)]
    
    for a in actions:
        if a==a_start:
            policy[state][a]=1-(len(actions)-1)*epsilon/len(actions)#acton_value最大的动作的概率为1- (num_actions - 1) * epsilon / num_actions
        else:
            policy[state][a]=epsilon/len(actions)#其他动作的概率为 epsilon / num_actions
    
    return policy


def model_free_epsilon_greedy():
    
    #初始化策略 每个状态下的动作概率相等
    policy={s:{a:1/len(actions) for a in actions} for s in states}
    #初始化q(s,a))
    Q={s:{a:0 for a in actions} for s in states}
    #初始化return(s,a)
    R={s:{a:0 for a in actions} for s in states}
    #初始化N(s,a)
    N={s:{a:0 for a in actions} for s in states}
    
    #开始采样
    for episode in tqdm(range(num_episodes),desc=f'Training with ε-Greedy'):
        #随机选择状态的动作
        current_state=states[np.random.choice([i for i in range(len(states))])]
        current_action=actions[np.random.choice([i for i in range(len(actions))])]
        episode_history=[]
        reward_history=[]
        
        #采样长度
        for step in range(max_steps):
            episode_history.append((current_state,current_action))
            next_state=state_transition(current_state,current_action)
            r=reward(current_state,current_action,next_state)
            next_action=get_action_with_epsilon_greedy(next_state,policy)
            reward_history.append(r)
            current_state=next_state
            current_action=next_action
            
        G=0
        for t in range(len(episode_history)-1,-1,-1):
            s_t,a_t=episode_history[t]
            r_t=reward_history[t]
            G=gamma*G + r_t
            R[s_t][a_t]+=G
            N[s_t][a_t]+=1
            #policy evaluation
            Q[s_t][a_t]=R[s_t][a_t]/N[s_t][a_t]
            #policy improvement
            policy=update_policy(s_t,Q,policy)
        
    optimal_V={s:0 for s in states}
    for s in states:
        for a in actions:
            optimal_V[s]+=Q[s][a]*policy[s][a]
    
    optimal_policy={s:None for s in states}
    for s in states:
        max_value=max(policy[s].values())
        for a in actions:
            if policy[s][a]==max_value:
                optimal_policy[s]=a
    return optimal_policy,optimal_V


def display_policy(policy, grid_size=(5,5)):
    """可视化显示策略"""
    action_symbols = {
        'up': '↑',
        'down': '↓',
        'left': '←',
        'right': '→',
        'stay': '●',
        None: '★'
    }
    
    print("\nOptimal Policy:")
    for i in range(grid_size[0]):
        row = []
        for j in range(grid_size[1]):
            state = (i, j)
            if state == target:
                row.append(f'★({action_symbols.get(policy.get(state), "?")})')  # 目标
            elif state in obstacles:
                row.append(f'■({action_symbols.get(policy.get(state), "?")})')  # 障碍物
            else:
                row.append("  "+action_symbols.get(policy.get(state), '?')+" ")
        print(' | '.join(row))
        if i < grid_size[0] - 1:
            print('-' * (grid_size[1] * 7 - 1))
            

if __name__ == "__main__":
    optimal_policy, optimal_value = model_free_epsilon_greedy()
    display_policy(optimal_policy)
    print("\nOptimal Policy:")
    for i in range(5):
        for j in range(5):
            print(f"{optimal_policy[(i,j)]:6}",end=" ")
        print()
    print("\nOptimal Value Function:")
    for i in range(5):
        for j in range(5):
            print(f"{optimal_value[(i,j)]:6.2f}",end=" ")
        print()