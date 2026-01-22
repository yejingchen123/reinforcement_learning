import numpy as np

#3x3的方格，(1,2),(2,0)障碍物，(2,2)终点

obstacles=[(1,2),(2,0)]
target=(2,2)

#所有状态
states=[(i,j) for i in range(3) for j in range(3)]

#所有动作
actions=['up','down','left','right','stay']

#discount rate
gamma=0.9

#结束阈值
threshold=1e-4

#状态转移函数
def state_transition(state,action):
    x,y=state
    if action=='up':
        x=max(0,x-1)
    elif action=='down':
        x=min(2,x+1)
    elif action=='left':
        y=max(0,y-1)
    elif action=='right':
        y=min(2,y+1)
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

#policy evaluation
def plicy_evaluation(policy):#在给定的policy下根据贝尔曼期望方程计算每个状态的价值,使用迭代法
    V={s:0 for s in states}
    while True:
        V_old=V.copy()
        for state in states:
            action=policy[state]
            next_state=state_transition(state,action)
            r=reward(state,action,next_state)
            V[state]=r+gamma*V_old[next_state]#贝尔曼期望方程,为什么state value等于action value，因为是每个状态的policy是根据贪心确定的，该状态下的策略只有一个动作
        if np.linalg.norm(np.array(list(V.values()))-np.array(list(V_old.values())))<threshold:
            break
    return V

#policy improvement
def policy_improvement(V):#根据当前状态跟新策略：使用贪心策略，即选择action value最大的策略
    policy={s:None for s in states}
    for state in states:
        action_values=[]
        for action in actions:
            next_state=state_transition(state,action)
            r=reward(state,action,next_state)
            action_value=r+gamma*V[next_state]
            action_values.append(action_value)
        policy[state]=actions[action_values.index(max(action_values))]
    return policy

#policy iteration
def policy_iteration():
    policy={s:np.random.choice(actions) for s in states}#初始化随机策略
    while True:
        V=plicy_evaluation(policy)
        new_policy=policy_improvement(V)
        if new_policy==policy:#策略不变，收敛
            break
        policy=new_policy
    return policy,V

def display_policy(policy, grid_size=(3,3)):
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
            print('-' * (grid_size[1] * 4 - 1))



if __name__=='__main__':
    optimal_policy,optimal_value=policy_iteration()
    display_policy(optimal_policy)
    print("\nOptimal Policy:")
    for i in range(3):
        for j in range(3):
            print(f"State ({i},{j}): {optimal_policy[(i,j)]}")
    print("\nOptimal Value Function:")
    for i in range(3):
        for j in range(3):
            print(f"State ({i},{j}): {optimal_value[(i,j)]:.4f}")

