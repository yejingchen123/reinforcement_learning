# expected_sarsa

from sara import Sarsa
import numpy as np
from tqdm import tqdm

class ExpectedSarsa(Sarsa):
    
    def train(self):
        for episode in tqdm(range(self.num_episodes),desc="Training Expected Sarsa"):
            #随机选择初始状态和动作
            state=self.env.states[np.random.choice(len(self.env.states))]
            action=self.get_action_with_epsilon_greedy(state)
            
            for step in range(self.max_steps):
                next_state=self.env.state_transition(state,action)
                reward=self.env.reward(state,action,next_state)
                
                #计算下一状态下的期望Q值
                expected_q=0
                for a in self.env.actions:
                    action_prob=self.policy[next_state][a]
                    expected_q+=action_prob*self.q[next_state][a]
                
                #expected sarsa更新公式 q(s,a)=q(s,a)+α[r+γ*E[q(s',a')]-q(s,a)]
                td_target=reward+self.gamma*expected_q
                td_error=self.q[state][action]-td_target
                self.q[state][action]-=self.alpha*td_error
                
                
                #更新策略
                self.update_policy(state)
                
                
                state=next_state
                action=self.get_action_with_epsilon_greedy(state)
                
                



if __name__=="__main__":
    expected_sarsa=ExpectedSarsa()
    expected_sarsa.inite()
    expected_sarsa.train()
    expected_sarsa.display_optimal_policy()