import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from environment import GridEnvironment  # 导入你的环境类

class QNetwork(nn.Module):
    """Q值神经网络，输入状态，输出所有动作的Q值"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)

class SarsaTorch:
    def __init__(self):
        self.env = GridEnvironment()
        self.gamma = 0.9    # 折扣因子
        self.epsilon = 0.1  # ε-greedy探索率
        self.num_episodes = 2000  # 增加训练轮次（神经网络需要更多训练）
        self.max_steps = 100
        self.alpha = 0.001  # 学习率（Adam优化器使用）
        
        # 状态和动作维度
        self.state_dim = 2  # (i, j)坐标
        self.action_dim = len(self.env.actions)
        self.action_to_idx = {a: i for i, a in enumerate(self.env.actions)}
        self.idx_to_action = {i: a for i, a in enumerate(self.env.actions)}
        
        # 初始化Q网络
        self.q_network = QNetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.alpha)
        
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        
    def state_to_tensor(self, state):
        """将状态转换为PyTorch张量"""
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    def get_q_values(self, state):
        """获取给定状态下所有动作的Q值"""
        state_tensor = self.state_to_tensor(state)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
        return q_values
    
    def get_action_with_epsilon_greedy(self, state):
        """ε-greedy策略选择动作"""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.actions)
        else:
            q_values = self.get_q_values(state)
            best_action_idx = np.argmax(q_values)
            return self.idx_to_action[best_action_idx]
    
    def compute_td_error(self, state, action, reward, next_state, next_action):
        """计算TD误差"""
        state_tensor = self.state_to_tensor(state)
        next_state_tensor = self.state_to_tensor(next_state)
        
        # 获取Q值
        current_q_values = self.q_network(state_tensor)
        next_q_values = self.q_network(next_state_tensor)
        
        # 获取特定动作的Q值
        action_idx = self.action_to_idx[action]
        next_action_idx = self.action_to_idx[next_action]
        
        current_q = current_q_values[0, action_idx]
        next_q = next_q_values[0, next_action_idx]
        
        # 计算TD目标和误差
        td_target = reward + self.gamma * next_q
        td_error = td_target - current_q
        
        return td_error, current_q
    
    def update_weights(self, td_error, current_q):
        """更新网络权重"""
        loss = td_error ** 2  # MSE损失
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def train(self):
        """训练Sarsa算法"""
        episode_rewards = []
        best_avg_reward = -np.inf
        patience_counter = 0
        patience_limit = 50  # 早停耐心值
        
        for episode in tqdm(range(self.num_episodes), desc="Training Sarsa (PyTorch)"):
            # 仅从合法状态开始（非障碍物、非目标）
            valid_states = [s for s in self.env.states 
                           if s not in self.env.obstacles and s != self.env.target]
            state = valid_states[np.random.choice(len(valid_states))]
            action = self.get_action_with_epsilon_greedy(state)
            
            total_reward = 0
            done = False
            
            for step in range(self.max_steps):
                # 检查是否到达目标
                if state == self.env.target:
                    done = True
                    break
                
                # 执行动作，获取下一状态和奖励
                next_state = self.env.state_transition(state, action)
                reward = self.env.reward(state, action, next_state)
                total_reward += reward
                
                # 选择下一个动作（使用ε-greedy）
                next_action = self.get_action_with_epsilon_greedy(next_state)
                
                # 计算TD误差
                td_error, current_q = self.compute_td_error(
                    state, action, reward, next_state, next_action
                )
                
                # 更新网络权重
                self.update_weights(td_error, current_q)
                
                # 转移到下一状态
                state, action = next_state, next_action
                
                # 检查是否到达目标
                if state == self.env.target:
                    done = True
                    break
            
            episode_rewards.append(total_reward)
            
            # 每100轮打印一次平均奖励
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"\nEpisode {episode+1}/{self.num_episodes}, Avg Reward: {avg_reward:.2f}")
                
                # 早停机制
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience_limit:
                    print(f"Early stopping at episode {episode+1} (no improvement for {patience_limit} episodes)")
                    break
        
        return episode_rewards
    
    def get_optimal_policy(self):
        """提取最优策略（确定性）"""
        optimal_policy = {}
        for state in self.env.states:
            if state == self.env.target:
                optimal_policy[state] = 'stay'  # 在目标处停留
                continue
                
            if state in self.env.obstacles:
                optimal_policy[state] = None  # 障碍物无策略
                continue
                
            q_values = self.get_q_values(state)
            best_action_idx = np.argmax(q_values)
            optimal_policy[state] = self.idx_to_action[best_action_idx]
        
        return optimal_policy
    
    def display_optimal_policy(self):
        """显示最优策略"""
        optimal_policy = self.get_optimal_policy()
        self.env.display_policy(optimal_policy)
    
    def save_model(self, path="sarsa_model.pth"):
        """保存模型"""
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path="sarsa_model.pth"):
        """加载模型"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == "__main__":
    # 训练模型
    sarsa_torch = SarsaTorch()
    rewards = sarsa_torch.train()
    
    # 保存模型
    sarsa_torch.save_model()
    
    # 显示最优策略
    print("\nFinal Optimal Policy:")
    sarsa_torch.display_optimal_policy()
    
    # 可选：绘制训练曲线
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Sarsa Training Rewards')
        plt.grid(True)
        plt.savefig('sarsa_training_curve.png')
        print("\nTraining curve saved as 'sarsa_training_curve.png'")
    except ImportError:
        print("\nMatplotlib not available. Skipping training curve plot.")