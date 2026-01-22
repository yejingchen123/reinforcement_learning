# actor-critic off policy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from environment import GridEnvironment
import matplotlib.pyplot as plt
from tqdm import tqdm

class Actor(nn.Module):
    # Actor网络,输入状态（25维one-hot向量），输出动作概率
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(25,64),
            nn.ReLU(),
            nn.Linear(64,5)
        )
    
    def forward(self,x):
        x=self.model(x)
        return nn.functional.softmax(x,dim=-1)


class Critic(nn.Module):
    # Critic网络，输入状态（25维one-hot向量），输出当前状态价值
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(25,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
    
    def forward(self,x):
        return self.model(x)

    

class Actor_Critic:
    # actor-critic off-policy based on importance sampling
    def __init__(self,env,actor,critic,gamma=0.9,actor_lr=0.001,critic_lr=0.001,
                 nums_episodes=3000,max_steps=100):
        
        self.env=env
        self.gamma=gamma
        
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor=actor.to(self.device)
        self.critic=critic.to(self.device)
        
        self.actor_optimizer=optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer=optim.Adam(self.critic.parameters(),lr=critic_lr)
        
        self.nums_episodes=nums_episodes
        self.max_steps=max_steps
        
        self.episode_rewards=[]
        self.actor_loss_history=[]
        self.critic_loss_history=[]
        
    def state_to_tensor(self,state):
        #将（i，j)转化维25维one-ont向量
        x,y=state
        index=x*5+y
        state_tensor=torch.zeros(25,dtype=torch.float32)
        state_tensor[index]=1.0
        return state_tensor
    
    def train(self):
        
        for episode in tqdm(range(self.nums_episodes),desc="Traing Actor-Critic"):
            #随机选择状态
            state=self.env.states[np.random.choice(len(self.env.states))]
            
            total_reward=0
            for step in range(self.max_steps):
                #探索性策略，随机选取动作
                action_idx=np.random.choice(len(self.env.actions))
                action=self.env.actions[action_idx]
                next_state=self.env.state_transition(state,action)
                reward=self.env.reward(state,action,next_state)
                
                #先获得状态tensor
                state_tensor=self.state_to_tensor(state)
                next_state_tensor=self.state_to_tensor(next_state)
                
                #获得状态价值
                state_value=self.critic(state_tensor)
                next_sate_value=self.critic(next_state_tensor)
                
                #获得动作概率
                action_prob=self.actor(state_tensor)[action_idx]
                
                #计算TD error r + gamma * v(s_next) - v(s)
                #TD error 不参与求导
                td_error=reward+self.gamma*next_sate_value.item()-state_value.item()
                
                #计算重要性采样权重
                # 不参与求导
                weight=action_prob.item()/0.2
                
                #计算actor_losss
                #pytoch是梯度下降，loss需要加负号
                actor_loss=-1.0*weight*td_error*torch.log(action_prob+1e-8)#上一个小数值 1e-8 避免 log(0)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.actor_loss_history.append(actor_loss.item())
                
                #计算critic_loss
                critic_loss=-1.0*weight*td_error*state_value
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                self.critic_loss_history.append(critic_loss.item())
                
                total_reward+=reward
                
                state=next_state
            
            self.episode_rewards.append(total_reward)
            if (episode+1)%100==0:
                tqdm.write(f'100轮平均奖励：{np.mean(self.episode_rewards[-100:])}')
            
    
    def show_optimal_policy(self):
        optimal_policy={}
        for state in self.env.states:
            
            with torch.no_grad():
                state_tensor=self.state_to_tensor(state)
                action_probs=self.actor(state_tensor)
                action_idx=torch.argmax(action_probs).item()
                optimal_policy[state]=self.env.actions[action_idx]
            
        self.env.display_policy(optimal_policy)
    
    def plot_statistics(self):
        plt.figure(figsize=(12,4))
        
        plt.subplot(1,3,1)
        plt.plot(self.episode_rewards)
        plt.xlabel('episode')
        plt.ylabel('total reward')
        plt.title('Total Reward per Episode')
        
        plt.subplot(1,3,2)
        plt.plot(self.actor_loss_history)
        plt.xlabel('step')
        plt.ylabel('actor loss')
        plt.title('Actor Loss per Step')
        
        plt.subplot(1,3,3)
        plt.plot(self.critic_loss_history)
        plt.xlabel('step')
        plt.ylabel('Critic Loss')
        plt.title('Critic Loss per Step')
        
        plt.tight_layout()
        plt.show()

    

if __name__=='__main__':
    env=GridEnvironment()
    actor=Actor()
    critic=Critic()
    
    config={
        'gamma':0.9,
        'actor_lr':0.001,
        'critic_lr':0.001,
        'nums_episodes':2000,
        'max_steps':100
    }
    
    agent=Actor_Critic(env,actor,critic,**config)
    agent.train()
    agent.show_optimal_policy()
    agent.plot_statistics()
    
        
    