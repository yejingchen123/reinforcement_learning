import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm

# --- 1. 您的环境代码 ---
class GridEnvironment:
    def __init__(self):
        # 5x5的网格,(1,1)(1,2)(2,2)(3,1)(3,3)(4,1)为障碍物,(3,2)为终点
        self.obstacles = [(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)]
        self.target = (3, 2)
        # 所有状态
        self.states = [(i, j) for i in range(5) for j in range(5)]
        # 所有动作
        self.actions = ['up', 'down', 'left', 'right', 'stay']

    def state_transition(self, state, action):
        x, y = state
        if action == 'up':
            x = max(0, x - 1)
        elif action == 'down':
            x = min(4, x + 1)
        elif action == 'left':
            y = max(0, y - 1)
        elif action == 'right':
            y = min(4, y + 1)
        return (x, y)

    def reward(self, state, action, next_state):
        # 走出边界
        if next_state == state and action != 'stay':
            return -1
        # 障碍物
        if next_state in self.obstacles:
            return -10
        # 终点
        if next_state == self.target:
            return 1
        # 普通点
        return 0

    def display_policy(self, policy_dict, grid_size=(5, 5)):
        """可视化显示策略"""
        action_symbols = {
            'up': '↑', 'down': '↓', 'left': '←', 'right': '→', 'stay': '●', None: '★'
        }
        print("\n=== Learned Optimal Policy ===")
        for i in range(grid_size[0]):
            row = []
            for j in range(grid_size[1]):
                state = (i, j)
                if state == self.target:
                    row.append(f'★({action_symbols.get(policy_dict.get(state), "?")})')
                elif state in self.obstacles:
                    row.append(f'■({action_symbols.get(policy_dict.get(state), "?")})')
                else:
                    row.append("  " + action_symbols.get(policy_dict.get(state), '?') + " ")
            print(' | '.join(row))
            if i < grid_size[0] - 1:
                print('-' * (grid_size[1] * 7 - 1))

# --- 2. 定义神经网络模型 ---

# 辅助函数：将状态(x,y)转换为One-Hot张量
# 输入(0,0) -> [1, 0, 0, ... 0] (长度25)
def state_to_tensor(state):
    index = state[0] * 5 + state[1]
    tensor = torch.zeros(25)
    tensor[index] = 1.0
    return tensor

# Actor网络 (Policy Network): 输入状态 -> 输出每个动作的概率
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        # 输入25个状态特征，输出5个动作的logits
        self.fc =nn.Sequential(
            nn.Linear(25, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        ) 
        
    def forward(self, x):
        x = self.fc(x)
        # 使用Softmax确保输出是概率分布
        return F.softmax(x, dim=-1)

# Critic网络 (Value Network): 输入状态 -> 输出状态价值 V(s)
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        # 输入25个状态特征，输出1个标量价值
        self.fc=nn.Sequential(
            nn.Linear(25, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ) 
        # nn.Linear(25, 1)
        
    def forward(self, x):
        return self.fc(x)

# --- 3. 算法实现 (Off-policy Actor-Critic) ---

def train():
    env = GridEnvironment()
    
    # 初始化 Actor 和 Critic
    actor = Actor()
    critic = Critic()
    
    # 定义优化器 (对应伪代码中的参数更新)
    # alpha_theta
    optimizer_actor = optim.Adam(actor.parameters(), lr=0.005) 
    # alpha_w
    optimizer_critic = optim.Adam(critic.parameters(), lr=0.01) 
    
    gamma = 0.9  # 折扣因子
    num_episodes = 3000 # 训练回合数

    print("Start Training...")
    
    for episode in tqdm(range(num_episodes)):
        state = env.states[np.random.choice(len(env.states))] # 每次从起点开始，或者随机起点
        steps = 100
        
        for step in range(steps):
        # while state != env.target and steps < 50: # 限制步数防止死循环
        #     steps += 1
            
            # --- 1. Behavior Policy (beta) ---
            # 伪代码: Generate a_t following beta(s_t)
            # 这里我们使用完全随机策略作为 Behavior Policy，以保证探索性
            # beta(a|s) = 0.2 (因为有5个动作，均匀分布)
            action_idx = random.randint(0, 4)
            action_str = env.actions[action_idx]
            prob_behavior = 1.0 / 5.0 
            
            # --- 2. Execute and Observe ---
            # 伪代码: observe r_{t+1}, s_{t+1}
            next_state = env.state_transition(state, action_str)
            reward = env.reward(state, action_str, next_state)
            
            # 准备数据转Tensor
            s_tensor = state_to_tensor(state)
            next_s_tensor = state_to_tensor(next_state)
            
            # --- 3. 计算 Actor (Target Policy) 的概率 ---
            # 获取当前策略 pi(a|s)
            prob_all_actions = actor(s_tensor)
            prob_target = prob_all_actions[action_idx] # pi(a_t|s_t)
            
            # --- 4. 计算 Critic (Value) ---
            val_curr = critic(s_tensor)       # v(s_t, w_t)
            val_next = critic(next_s_tensor)  # v(s_{t+1}, w_t)
            
            # --- 5. 计算 TD Error (delta) ---
            # 伪代码: delta = r + gamma * v(s_next) - v(s)
            # 注意：计算delta时，不需要对v(s)求导，也不需要对v(s_next)求导，
            # 它们在计算delta这一步仅仅是数值。
            target_value = reward + gamma * val_next.item()
            delta = target_value - val_curr.item()
            
            # --- 6. 重要性采样比率 (rho) ---
            # rho = pi(a|s) / beta(a|s)
            # .detach() 很重要，因为rho只是一个缩放系数，不应该反向传播去更新pi的分母部分
            rho = prob_target.item() / prob_behavior
            
            # --- 7. 更新 Actor ---
            # 伪代码: theta = theta + alpha * rho * delta * grad(ln(pi))
            # 在PyTorch中，我们通过最小化 Loss 来更新: theta = theta - lr * grad(Loss)
            # 为了等价，我们要构造 Loss 使得: -grad(Loss) = rho * delta * grad(ln(pi))
            # 即: Loss = - rho * delta * ln(pi)
            
            log_prob = torch.log(prob_target)
            actor_loss = -1.0 * rho * delta * log_prob
            
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()
            
            # --- 8. 更新 Critic ---
            # 伪代码: w = w + alpha * rho * delta * grad(v(s))
            # 同样构造 Loss 使得: -grad(Loss) = rho * delta * grad(v(s))
            # 即: Loss = - rho * delta * v(s)
            # 注意：这里我们重新计算 val_curr，但这回需要保留梯度（不要detach）
            
            val_curr_grad = critic(s_tensor) # 重新前向传播以保留计算图
            critic_loss = -1.0 * rho * delta * val_curr_grad
            
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()
            
            # 状态更新
            state = next_state
            
        if (episode + 1) % 500 == 0:
            print(f"Episode {episode+1}/{num_episodes} completed.")

    print("Training Finished.")
    return actor

# --- 4. 测试与可视化 ---
def test_and_display(env, actor_model):
    policy = {}
    
    # 遍历所有状态，选择概率最大的动作作为该状态的策略
    with torch.no_grad():
        for i in range(5):
            for j in range(5):
                state = (i, j)
                s_tensor = state_to_tensor(state)
                probs = actor_model(s_tensor)
                best_action_idx = torch.argmax(probs).item()
                policy[state] = env.actions[best_action_idx]
    
    env.display_policy(policy)

if __name__ == '__main__':
    trained_actor = train()
    env = GridEnvironment()
    test_and_display(env, trained_actor)