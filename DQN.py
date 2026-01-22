# Deep Q-learning with PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from tqdm import tqdm
import matplotlib.pyplot as plt
from environment import GridEnvironment

class QNetwork(nn.Module):
    """
    用于估计Q值的网络
    输入：状态(i,j)
    输出：该状态下每个动作的Q值
    """
    def __init__(self,state_dim,action_dim,hidden_dim=64):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,action_dim),
        )
        
    def forward(self,x):
        return self.model(x)
    


class ReplayBuffer:
    """
    经验回放缓冲区
    存储（state, action, reward, next_state，done）
    实现batch采样
    """
    def __init__(self,capacity):#capacity:容量
        self.buffer=deque(maxlen=capacity)
        
    def push(self,state,action,reward,next_state):
        #将采集的数据存入缓冲区
        self.buffer.append((state,action,reward,next_state))
    
    def sample(self,batch_size):
        #随机采用一个batch_size的数据
        batch=random.sample(self.buffer,min(len(self.buffer),batch_size))
        #batch是一个list，包含多个(state,action,reward,next_state,done)
        #[(s1,a1,r1,s1',d1),(s2,a2,r2,s2',d2),...]
        state,action,reward,next_state=zip(*batch)
        #*batch表示将batch解包,变为(s1,a1,r1,s1',d1),(s2,a2,r2,s2',d2),...
        #zip将每个元组的对应位置打包成一个元组，形成新的列表
        #[(s1,s2,...),(a1,a2,...),(r1,r2,...),(s1',s2',...),(d1,d2,...)]
        return state,action,reward,next_state
    
    def __len__(self):
        #返回缓冲区当前的大小
        return len(self.buffer)

class DQN:
    """
    实现Deep Q-learning算法
    """
    
    def __init__(self,env,state_dim=2,action_dim=5,gamma=0.9,lr=0.001,
                 replay_capacity=10000,batch_size=64,target_update_freq=100,
                 num_episodes=2000,max_steps=100):
        """
        Args:
            env: 环境对象
            state_dim: 状态维度
            action_dim: 动作维度
            gamma: 折扣因子
            lr: 学习率
            replay_capacity: 经验回放缓冲区容量
            batch_size: 训练批次大小
            target_update_freq: 目标网络更新频率（步数）
            num_episodes: 训练总回合数
            max_steps: 每回合最大步数
        """
        
        self.env=env
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.gamma=gamma
        self.lr=lr
        self.replay_capacity=replay_capacity
        self.batch_size=batch_size
        self.target_update_freq=target_update_freq
        self.num_episodes=num_episodes
        self.max_steps=max_steps
        
        #动作映射
        self.action_to_idx={a:i for i,a in enumerate(self.env.actions)}
        self.idx_to_action={i:a for i,a in enumerate(self.env.actions)}
        
        #设备
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # main network 和 target network
        self.q_network=QNetwork(state_dim,action_dim).to(self.device)
        self.target_network=QNetwork(state_dim,action_dim).to(self.device)
        #将target network的参数初始化为main network的参数
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        #优化器
        self.optimizer=optim.Adam(self.q_network.parameters(),lr=self.lr)
        
        #经验回放缓冲区
        self.replay_buffer=ReplayBuffer(replay_capacity)
        
        #训练统计
        self.episode_rewards=[]
        self.loss_history=[]
        
    # def state_to_tensor(self,state):
    #     #将状态转换为张量
    #     state_tensor=torch.tensor(
    #         state,dtype=torch.float32
    #     ).unsqueeze(0).to(self.device) #添加batch维度
    #     return state_tensor
    
    def get_action(self):
        #探索性策略，在任意状态下选择每个动作的概率相等
        return np.random.choice(self.env.actions)
    
    def compute_loss(self,batch):
        #计算一个batch的损失
        states,actions,rewards,next_states=batch
        #states:tuple ((s1x,s1y),(s2x,s2y),...)
        #actions:tuple (a1,a2,...)
        #rewards:tuple (r1,r2,...)
        
        #转化为张量
        states_tensor=torch.tensor(
            states,dtype=torch.float32
        ).to(self.device) #shape: (batch_size, state_dim)
        actions_tensor=torch.tensor(
            [self.action_to_idx[a] for a in actions],dtype=torch.long
        ).to(self.device) #shape: (batch_size,),long表示整数类型
        rewards_tensor=torch.tensor(
            rewards,dtype=torch.float32
        ).to(self.device) #shape: (batch_size,)
        next_states_tensor=torch.tensor(
            next_states,dtype=torch.float32
        ).to(self.device) #shape: (batch_size, state_dim)
        
        #当前Q值
        current_q_values=self.q_network(states_tensor)#shape: (batch_size, action_dim)
        current_q_values=current_q_values.gather(1,actions_tensor.unsqueeze(1)).squeeze(1)
        #经过network,每个状态对应5个q值，gather函数根据actions的索引取出状态对应的q值
        
        #目标Q值
        with torch.no_grad():
            next_q_values=self.target_network(next_states_tensor)#shape: (batch_size, action_dim)
            max_next_q_values=next_q_values.max(1)[0] 
            #取每个状态下最大的q值,shape: (batch_size,),取[0]是因为max返回值是(最大值,索引)
            target_q_values=rewards_tensor + self.gamma * max_next_q_values
            #shape: (batch_size,)
            
        #损失函数：均方误差
        loss=nn.MSELoss()(current_q_values,target_q_values)
        return loss
    
    def train_step(self):
        #执行一个训练步骤
        if len(self.replay_buffer) < self.batch_size:
            return None  # 如果缓冲区中的样本不足一个批次，则不进行训练
        
        batch=self.replay_buffer.sample(self.batch_size)
        loss=self.compute_loss(batch)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
    
    def update_target_network(self):
        #更新目标网络参数
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def train(self):
        #训练主循环
        for episode in tqdm(range(self.num_episodes)):
            #随机选择起始状态
            state=self.env.states[np.random.choice(len(self.env.states))]
            
            total_reward=0
            
            for step in range(self.max_steps):
                action = self.get_action()
                next_state=self.env.state_transition(state,action)
                reward=self.env.reward(state, action, next_state)
                
                #存储经验
                self.replay_buffer.push(state,action,reward,next_state)
                
                #训练一步
                self.train_step()
                
                #更新状态
                state=next_state
                total_reward+=reward
            
            #更新目标网络
            if episode % self.target_update_freq == 0:
                self.update_target_network()
            
            #记录总奖励
            self.episode_rewards.append(total_reward)
            
            # 打印每100回合的平均奖励
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
                
    
    #提取最优策略
    def get_optimal_policy(self):
        policy={}
        for state in self.env.states:
            state_tensor=torch.tensor(
                state,dtype=torch.float32
            ).unsqueeze(0).to(self.device) #shape: (1, state_dim)
            with torch.no_grad():
                q_values=self.q_network(state_tensor) #shape: (1, action_dim)
            best_action_idx=q_values.argmax().item()
            best_action=self.idx_to_action[best_action_idx]
            policy[state]=best_action
        return policy

    #显示最优策略
    def display_optimal_policy(self):
        optimal_policy = self.get_optimal_policy()
        self.env.display_policy(optimal_policy)
    
    
    #绘制图像
    def plot_statistics(self):
        #绘制奖励和损失曲线
        plt.figure(figsize=(12,5))
        
        #奖励曲线
        plt.subplot(1,2,1)
        plt.plot(self.episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode Rewards over Time')
        
        #损失曲线
        plt.subplot(1,2,2)
        plt.plot(self.loss_history)
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss over Time')
        
        plt.tight_layout()
        plt.show()
    

if __name__ == "__main__":
    env=GridEnvironment()
    config = {
        'state_dim': 2,
        'action_dim': len(env.actions),
        'gamma': 0.95,
        'lr': 0.001,
        'replay_capacity': 5000,
        'batch_size': 32,
        'target_update_freq': 50,
        'num_episodes': 2000,
        'max_steps': 100
    }
    
    dqn=DQN(env,**config)
    dqn.train()
    dqn.display_optimal_policy()
    dqn.plot_statistics()
    
                