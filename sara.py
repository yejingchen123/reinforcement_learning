# Optimal policy learning by Sarsa

import numpy as np
from environment import GridEnvironment
from tqdm import tqdm

class Sarsa:
    
    def __init__(self):
        self.env=GridEnvironment()
        self.gamma=0.9# discount rate
        self.epsilon=0.1#ε-greedy
        self.num_episodes=1000#采样次数
        self.max_steps=10000#采样长度
        self.q={}#q表
        self.policy={}#策略表
        self.alpha=0.01# learning rate
        
    def inite(self):#初始化action_vale和策略
        self.q={s:{a:0 for a in self.env.actions} for s in self.env.states}
        self.policy={s:{a:1/len(self.env.actions) for a in self.env.actions} for s in self.env.states}
    
    def get_action_with_epsilon_greedy(self,state):
        #获得当前状态下每个动作的概率
        action_probs=self.policy[state]
        actions=list(action_probs.keys())
        probs=list(action_probs.values())
        return np.random.choice(actions,p=probs)
    
    def update_policy(self,state):
        #获得当前状态下每个动作的action_value
        q_values=[self.q[state][a] for a in self.env.actions]
        
        #找到action_value最大的动作
        max_q=max(q_values)
        max_actions=[a for a in self.env.actions if self.q[state][a]==max_q]
        
        n_actions=len(self.env.actions)
        n_best=len(max_actions)
        
        #更新策略
        for a in self.env.actions:
            if a in max_actions:
                #有多个最大动作时,计算公式为（1-ε）/n_best + ε/n_actions
                self.policy[state][a]=((1 - self.epsilon) / n_best) + (self.epsilon / n_actions)
            else:
                #其他动作概率为ε/n_actions
                self.policy[state][a]=self.epsilon / n_actions
                
                
    def train(self):
        for episode in tqdm(range(self.num_episodes),desc="Training Sarsa"):
            #随机选择初始状态和动作
            state=self.env.states[np.random.choice(len(self.env.states))]
            action=self.get_action_with_epsilon_greedy(state)
            
            for step in range(self.max_steps):
                next_state=self.env.state_transition(state,action)
                reward=self.env.reward(state,action,next_state)
                next_action=self.get_action_with_epsilon_greedy(next_state)
                
                #sarsa更新公式 q(s,a)=q(s,a)+α[r+γ*q(s',a')-q(s,a)]
                td_target=reward+self.gamma*self.q[next_state][next_action]
                td_error=self.q[state][action]-td_target
                self.q[state][action]-=self.alpha*td_error
                
                
                #更新策略
                self.update_policy(state)
                
                
                state=next_state
                action=next_action
    
    def get_optimal_policy(self):
        #提取最优策略
        optimal_policy={}
        for state in self.env.states:
            action_probs=self.policy[state]
            best_action=max(action_probs,key=action_probs.get)
            optimal_policy[state]=best_action
        return optimal_policy
                
    def display_optimal_policy(self):
        #显示最优策略
        optimal_policy=self.get_optimal_policy()
        self.env.display_policy(optimal_policy)
    
    
            
    
    
    
if __name__=='__main__':
    sarsa=Sarsa()
    sarsa.inite()
    sarsa.train()
    sarsa.display_optimal_policy()