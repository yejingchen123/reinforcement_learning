# 5x5网格环境

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
        return -0.1

    def display_policy(self,policy,grid_size=(5,5)):
        """可视化显示策略"""
        action_symbols = {
            'up': '↑',
            'down': '↓',
            'left': '←',
            'right': '→',
            'stay': '●',
            None: '★'
        }
        
        print("\nOptimal Policy:")
        for i in range(grid_size[0]):
            row = []
            for j in range(grid_size[1]):
                state = (i, j)
                if state == self.target:
                    row.append(f'★({action_symbols.get(policy.get(state), "?")})')  # 目标
                elif state in self.obstacles:
                    row.append(f'■({action_symbols.get(policy.get(state), "?")})')  # 障碍物
                else:
                    row.append("  "+action_symbols.get(policy.get(state), '?')+" ")
            print(' | '.join(row))
            if i < grid_size[0] - 1:
                print('-' * (grid_size[1] * 7 - 1))
