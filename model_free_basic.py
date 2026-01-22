#model free basic 
#在policy iteration的基础上不使用模型，采用蒙特卡洛的方法采样进行策略评估

import numpy as np

#5x5的网格,(1,1)(1,2)(2,2)(3,1)(3,3)(4,1)#障碍物,(3，2)终点

obstacles=[(1,1),(1,2),(2,2),(3,1),(3,3),(4,1)]
target=(3,2)

#所有状态
states=[(i,j) for i in range(5) for j in range(5)]

#所有动作
actions=['up','down','left','right','stay']

#discount rate
gamma=0.9

#结束阈值
threshold=1e-4

#最大采样次数
max_episodes=1 #由于是确定性策略，每次采样的结果都是一样的，采样一次即可

#最大步数
max_steps=101

#状态转移函数
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

#policy evaluation using Monte Carlo sampling
def policy_evaluation(policy):#在给定的policy下采样计算每个状态的的每种action value
    V={s:{a:0 for a in actions} for s in states}#每个状态下每个动作的action value
    for state in states:
        for action in actions:
            total_return=0#多次采样的总return
            for episode in range(max_episodes):
                current_state=state
                current_action=action
                G=0#每次采样的return
                for step in range(max_steps):
                    next_state=state_transition(current_state,current_action)
                    r=reward(current_state,current_action,next_state)
                    G+=r*(gamma**step)
                    current_state=next_state
                    current_action=policy[current_state]
                
                total_return+=G
            average_return=total_return/max_episodes
            V[state][action]=average_return #每个状态下每个action的action value        
    return V

#policy improvement
def policy_improvement(V):#根据当前状态跟新策略：使用贪心策略，即选择action value最大的策略
    policy={s:None for s in states}
    for state in states:
        action_values=[]
        for action in actions:
            action_value=V[state][action]
            action_values.append(action_value)
        policy[state]=actions[action_values.index(max(action_values))]
    return policy

#model free policy iteration
def policy_iteration():
    policy={s:actions[np.random.randint(0,5)] for s in states}#初始化随机策略
    while True:
        V=policy_evaluation(policy)
        new_policy=policy_improvement(V)
        if new_policy==policy:
            V= {s: max(a_values.values()) for s, a_values in V.items()}  # 最大的action value作为状态价值
            break
        policy=new_policy
    return policy,V


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


if __name__=="__main__":
    optimal_policy,optimal_value=policy_iteration()
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