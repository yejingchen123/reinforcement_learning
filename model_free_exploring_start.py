# model free exploring start(every vistit)
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

#采样次数
num_episodes=10000

#采样长度
max_steps=100

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

#expoloring start with every visit
def exploring_strat():
    #初始化策略
    policy={s:np.random.choice(actions) for s in states}
    #初始化q(s,a)
    V={(s,a):0 for s in states for a in actions}
    #初始化return(s,a)
    R={(s,a):0 for s in states for a in actions}
    #初始化N(s,a)
    N={(s,a):0 for s in states for a in actions}
    
    #开始采样
    for episode in range(num_episodes):
        #随机选择状态和动作
        current_state=states[np.random.randint(len(states))]
        current_action=np.random.choice(actions)
        episode_history=[]
        reward_history=[]
        
        #采样长度
        for step in range(max_steps):
            episode_history.append((current_state,current_action))
            next_state=state_transition(current_state,current_action)
            r=reward(current_state,current_action,next_state)
            next_action=policy[next_state]
            reward_history.append(r)
            current_state=next_state
            current_action=next_action
        
        G=0
        #从后向前计算回报
        for t in range(len(episode_history)-1,-1,-1):
            state,action=episode_history[t]
            r=reward_history[t]
            G=gamma*G+r
            R[(state,action)]+=G  
            N[(state,action)]+=1
            #policy evaluation
            V[(state,action)]=R[(state,action)]/N[(state,action)]
            #policy improvement
            best_action=max(actions,key=lambda a:V[(state,a)])
            policy[state]=best_action
    
    optimal_V={s:max([V[(s,a)] for a in actions]) for s in states}
    return policy,optimal_V


    
    
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
    optimal_policy, optimal_value = exploring_strat()
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