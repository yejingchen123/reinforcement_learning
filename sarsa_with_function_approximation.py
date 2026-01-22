# sarsa with function approximation
#这里使用线性函数逼近来估计Q值
#输入向量为状态(i,j)和动作的one-hot编码拼接而成

from sara import Sarsa
import numpy as np
from tqdm import tqdm

class Sarsa_with_Function_Approximation(Sarsa):
    
    def __init__(self):
        super().__init__()
        #用函数逼近Q值
        self.weights=np.random.randn(len(self.env.states[0])+len(self.env.actions))
        
    def get_feature_vector(self,state,action):
        #先将状态和动作转换为特征向量
        state_vector=np.array(state)
        action_one_hot=np.zeros(len(self.env.actions))
        action_index=self.env.actions.index(action)
        action_one_hot[action_index]=1
        #拼接（i，j）和动作的one-hot编码
        feature_vector=np.concatenate((state_vector,action_one_hot))
        return feature_vector

    def get_q_value(self,state,action):
        feature_vector=self.get_feature_vector(state,action)
        q_value=np.dot(self.weights,feature_vector)
        return q_value
    
    def update_policy(self, state):
        #获得当前状态下每个动作的action_value
        q_values = [self.get_q_value(state, a) for a in self.env.actions]
        
        #找到action_value最大的动作
        max_q = max(q_values)
        max_actions = [a for a in self.env.actions if self.get_q_value(state, a) == max_q]
        
        n_actions = len(self.env.actions)
        n_best = len(max_actions)
        
        #更新策略
        for a in self.env.actions:
            if a in max_actions:
                #有多个最大动作时,计算公式为（1-ε）/n_best + ε/n_actions
                self.policy[state][a] = ((1 - self.epsilon) / n_best) + (self.epsilon / n_actions)
            else:
                #其他动作概率为ε/n_actions
                self.policy[state][a] = self.epsilon / n_actions
    
    def train(self):
        for episode in tqdm(range(self.num_episodes),desc="Training Sarsa with Function Approximation"):
            
            state=self.env.states[np.random.choice(len(self.env.states))]
            action=self.get_action_with_epsilon_greedy(state)
            
            for step in range(self.max_steps):
                next_state=self.env.state_transition(state,action)
                reward=self.env.reward(state,action,next_state)
                next_action=self.get_action_with_epsilon_greedy(next_state)
                
                #计算当前Q值和下一个Q值
                current_q=self.get_q_value(state,action)
                next_q=self.get_q_value(next_state,next_action)
                
                #参数更新
                td_target=reward+self.gamma*next_q
                td_error=td_target-current_q
                self.weights+=self.alpha*td_error*self.get_feature_vector(state,action)
                
                #更新策略
                self.update_policy(state)
                
                state=next_state
                action=next_action
                
                

if __name__=="__main__":
    sarsa_fa=Sarsa_with_Function_Approximation()
    sarsa_fa.inite()
    sarsa_fa.train()
    sarsa_fa.display_optimal_policy()