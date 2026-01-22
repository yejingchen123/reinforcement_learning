#policy gradient with pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from environment import GridEnvironment
from matplotlib import pyplot as plt

class PolicyNetwork(nn.Module):
    #使用神经网络表示策略函数
    #输入状态(x,y),输出该状态下各动作的概率
    def __init__(self,state_size,action_size,hidden_size=64):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(state_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,action_size),
            nn.Softmax(dim=-1)#dim=-1表示对最后一个维度进行softmax,输出为每个动作的概率
        )
        
    def forward(self,x):
        return self.model(x)
    
    
class PolicyGradient:
    """
    Policy Gradient 算法
    使用神经网络逼近策略函数
    通过monte carlo方法估计每个动作的价值
    通过梯度上升法优化策略网络,更新公式：θ ← θ + α * ∇lnπ(a|s) * q(s,a)
    """
    
    def __init__(self,env,state_dim=25,action_dim=5,lr=0.001,gamma=0.9,
                 nums_episodes=2000,max_steps=100):
        
        self.env=env
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.lr=lr
        self.gamma=gamma
        self.nums_episodes=nums_episodes
        self.max_steps=max_steps
        
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.policy_network=PolicyNetwork(self.state_dim,self.action_dim).to(self.device)
        
        self.optimizer=optim.Adam(self.policy_network.parameters(),lr=self.lr)
        
        self.episode_rewards=[]
        self.loss_history=[]
    
    def state_to_tensor(self,state):
        #将单个状态转换为tensor
        x,y=state
        index=x*5+y
        one_hot=np.zeros(25)
        one_hot[index]=1
        state_tensor=torch.tensor(
            one_hot,dtype=torch.float32
        ).unsqueeze(0).to(self.device)#增加一个batch维度
        return state_tensor
    
    def get_action(self,state_tensor):
        #在给定状态下根据策略网络选择动作，返回动作索引及其对数（ln）概率
        action_probs=self.policy_network(state_tensor)#[1,5]
        action_dist=torch.distributions.Categorical(action_probs)#创建一个类别分布
        action_idx=action_dist.sample()#采样动作索引
        log_prob=action_dist.log_prob(action_idx)#获取该动作的对数概率
        return action_idx.item(),log_prob

    def generate_episode(self):
        #生成一个采样轨迹
        state_episode=[]
        action_episode=[]
        prob_episode=[]
        reward_episode=[]
        
        #随机选择初始状态
        state=self.env.states[np.random.choice(len(self.env.states))]
        for step in range(self.max_steps):
            state_tensor=self.state_to_tensor(state)
            action_idx,log_prob=self.get_action(state_tensor)
            action=self.env.actions[action_idx]
            next_state=self.env.state_transition(state,action)
            reward=self.env.reward(state,action,next_state)
            state_episode.append(state)
            action_episode.append(action_idx)
            prob_episode.append(log_prob)
            reward_episode.append(reward)
            state=next_state
        
        return state_episode,action_episode,prob_episode,reward_episode
    
    def update_policy(self,prob_episode,reward_episode):
        #计算损失
        #损失函数为 -ln(π(a(t)|s(t))) * q(s(t),a(t)) 负号是为了迎合pytorch的梯度下降
        
        #先计算q(s(t),a(t)),从后往前计算累计奖励
        
        q_value=[]
        G=0
        for reward in reversed(reward_episode):
            G=reward+self.gamma*G
            q_value.insert(0,G)
        
        #转换为tensor
        q_value=torch.tensor(q_value,dtype=torch.float32).to(self.device)
        
        #标准化q_value，提升训练稳定性
        q_value=(q_value - q_value.mean()) / (q_value.std() + 1e-9)
        
        #计算损失
        policy_loss=[]
        for log_prob , G in zip(prob_episode,q_value):
            policy_loss.append(-log_prob*G)
            
        loss=torch.cat(policy_loss).sum()# 使用cat保持求和时的完整计算图
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_history.append(loss.item())
    
    def train(self):
        #训练
        for episode in tqdm(range(self.nums_episodes),desc="Policy Gradient Training"):
            state_episode,action_episode,prob_episode,reward_episode=self.generate_episode()
            self.episode_rewards.append(sum(reward_episode))
            self.update_policy(prob_episode,reward_episode)
            
            if (episode+1) % 100 == 0:
                tqdm.write(f"Episode {episode+1} Reward: {sum(reward_episode)}")
                tqdm.write(f"Loss: {self.loss_history[-1]}")        
            
    
    
    def get_optimal_policy(self):
        #获取最优策略
        policy={}
        for state in self.env.states:
            state_tensor=self.state_to_tensor(state)
            with torch.no_grad():
                action_index=torch.argmax(self.policy_network(state_tensor)).item()
                policy[state]=self.env.actions[action_index]
        
        return policy
    
    def display_optimal_policy(self):
        optimal_policy=self.get_optimal_policy()
        self.env.display_policy(optimal_policy)
    
    def plot_statistics(self):
        #绘制统计图
        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.plot(self.episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Episode Rewards")
        
        plt.subplot(1,2,2)
        plt.plot(self.loss_history)
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Policy Loss")
        plt.show()
        

if __name__ == "__main__":
    env=GridEnvironment()
    config = {
        'state_dim': len(env.states),
        'action_dim': len(env.actions),
        'gamma': 0.9,
        'lr': 0.0001,
        'nums_episodes': 500,
        'max_steps': 500
    }
    
    policy_gradient=PolicyGradient(env,**config)
    
    policy_gradient.train()
    policy_gradient.display_optimal_policy()
    policy_gradient.plot_statistics()
