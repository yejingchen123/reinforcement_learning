# Q_learining

from sara import Sarsa
import numpy as np
from tqdm import tqdm

class QLearning(Sarsa):
    
    def __init__(self):
        super().__init__()
        #用于off_policy的策略
        self.behavior_policy={}#行为策略表
        self.optimal_policy={}#最优策略表
        
    def inite(self):#初始化action_vale和策略
        super().inite()
        self.behavior_policy={s:{a:1/len(self.env.actions) for a in self.env.actions} for s in self.env.states}
        self.optimal_policy={s:{a:0 for a in self.env.actions} for s in self.env.states}
    
    def get_action(self,state):
        actions_list=list(self.behavior_policy[state].keys())
        actions_probs=list(self.behavior_policy[state].values())
        
        return np.random.choice(actions_list,p=actions_probs)
    
    def update_optimal_policy(self, state):
        
        index=np.argmax(list(self.q[state].values()))
        action=list(self.q[state].keys())[index]
        
        for a in self.env.actions:
            if a==action:
                self.optimal_policy[state][a]=1
            else:
                self.optimal_policy[state][a]=0
    
    def train_on_policy(self):
        
        for episode in tqdm(range(self.num_episodes),desc="Q Learning on policy Trainging"):
            #随机选择初始状态和策略
            state=self.env.states[np.random.choice(len(self.env.actions))]
            action=self.get_action_with_epsilon_greedy(state)
            for step in range(self.max_steps):
                next_state=self.env.state_transition(state,action)
                reward=self.env.reward(state,action,next_state)
                
                #求max of q(s(t+1),a)
                max_q=max([self.q[next_state][a] for a in self.env.actions])
                
                #Q_learing 更新公式 q(s,a)=q(s,a)-α[q(s,a)-r+γ*max_a(q(s',a))]
                td_target=reward+self.gamma*max_q
                td_error=self.q[state][action]-td_target
                self.q[state][action]-=self.alpha*td_error
                
                self.update_policy(state)
                
                state=next_state
                action=self.get_action_with_epsilon_greedy(state)
        
    def train_off_policy(self):
        for episode in tqdm(range(self.num_episodes),desc="Q Learning off policy Training"):
            #随机选择初始状态和策略
            state=self.env.states[np.random.choice(len(self.env.actions))]
            action=self.get_action(state)
            for step in range(self.max_steps):
                next_state=self.env.state_transition(state,action)
                reward=self.env.reward(state,action,next_state)
                
                #求max of q(s(t+1),a)
                max_q=max([self.q[next_state][a] for a in self.env.actions])
                
                #Q_learing 更新公式 q(s,a)=q(s,a)-α[q(s,a)-r+γ*max_a(q(s',a))]
                td_target=reward+self.gamma*max_q
                td_error=self.q[state][action]-td_target
                self.q[state][action]-=self.alpha*td_error
                
                #更新optimal_policy
                self.update_optimal_policy(state)
                
                state=next_state
                action=self.get_action(state)
        
    def display_off_policy(self):
        off_policy={}
        for state in self.env.states:
            index=np.argmax(list(self.optimal_policy[state].values()))
            off_policy[state]=list(self.optimal_policy[state].keys())[index]
        
        self.env.display_policy(off_policy)
            
            
            
            
        

if __name__=='__main__':
    qlearning=QLearning()
    qlearning.inite()
    # qlearning.train_on_policy()
    # qlearning.display_optimal_policy()
    # qlearning.train_off_policy()
    # qlearning.display_off_policy()