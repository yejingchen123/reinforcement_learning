import numpy as np

#2x2的方格，(0,1)障碍物，(1,1)终点

obstacle=(0,1)
target=(1,1)

#所有状态
states=[(0,0),(0,1),(1,0),(1,1)]

#所有动作
actions=['up','down','left','right','stay']

#discount rate
gamma=0.9

#结束阈值
threshold=1e-4

#状态转移函数
def state_transation(state,action):
    x,y=state
    if action=='up':
        x=max(0,x-1)
    elif action=='down':
        x=min(1,x+1)
    elif action=='left':
        y=max(0,y-1)
    elif action=='right':
        y=min(1,y+1)
    return (x,y)

#奖励函数
def reward(state,action,next_state):
    #走出边界
    if next_state==state and action!='stay':
        return -1
    #障碍物
    if next_state==obstacle:
        return -1
    #终点
    if next_state==target:
        return 1
    #普通点
    return 0

#value iteration
def value_iteration():
    V={s:0 for s in states}
    policy={s:None for s in states}
    while True:
        V_old=V.copy()
        for state in states:
            action_values=[]
            for action in actions:
                next_state=state_transation(state,action)
                r=reward(state,action,next_state)
                action_value=r+gamma*V_old[next_state]
                action_values.append(action_value)
            V[state]=max(action_values)
            policy[state]=actions[action_values.index(V[state])]
        
        #检查收敛
        delta=np.linalg.norm(np.array(list(V.values()))-np.array(list(V_old.values())))
        if delta<threshold:
            return V,policy
V,policy=value_iteration()
print("Optimal Value state:")
for state in states:
    print(f"State {state}: {V[state]:.4f}")
print("\nOptimal Policy:")
for state in states:
    print(f"State {state}: {policy[state]}")
