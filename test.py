import gymnasium as gym
import numpy as np
import time # 用于延时程序，方便渲染画面


class SarsaAgent(object):
    def __init__(self, obs_n, act_n, learning_rate=0.01, gamma=0.9, e_greed=0.1):
        self.act_n = act_n  # 动作维度，有几个动作可选
        self.lr = learning_rate  # 学习率
        self.gamma = gamma  # 后面的 Q 值对前面的影响
        self.epsilon = e_greed  # 按一定概率随机选动作
        self.Q = np.zeros((obs_n, act_n))

    # 根据输入观察值，采样输出的动作值（带 10% 的探索）
    def sample(self, obs):
        if (np.random.uniform(0, 1) < 1 - self.epsilon):  # 这里是 90% 可能性
            action = self.predict(obs)  # 执行最优动作
        else:  # 10% 的概率
            action = np.random.choice(self.act_n)  # 执行随机动作
        return action

    # 根据输入观察值，预测输出的动作值
    def predict(self, obs):
        Q_list = self.Q[obs, :]  # 获取当前状态下，作出所有动作，对应的 Q 值列表
        maxQ = np.max(Q_list)  # 求列表中的最大值
        action_list = np.where(Q_list == maxQ)[0]  # 最大 Q 值对应的动作即最优动作
        action = np.random.choice(action_list)  # 随机选择一个最优动作
        return action

    # 学习方法，也就是更新Q-table的方法
    def learn(self, obs, action, reward, next_obs, next_action, done):
        """ on-policy
            obs: 交互前的obs, s_t
            action: 本次交互选择的action, a_t
            reward: 本次动作获得的奖励r
            next_obs: 本次交互后的obs, s_t+1
            next_action: 根据当前Q表格, 针对next_obs会选择的动作, a_t+1
            done: episode是否结束
        """
        predict_Q = self.Q[obs, action]  # 交互前的状态下，选择的动作所对应 Q 值
        if (done):  # 游戏结束
            target_Q = reward  # 新的 Q 值为 reward
        else:  # 游戏没有结束
            target_Q = reward + self.gamma * self.Q[next_obs, next_action]
            # 用 reward 和 交互后状态下，选择的下一个动作对应的 Q 值，综合得到新的 Q 值
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)  # 使用 lr 做修正更新的幅度

    # 保存Q表格数据到文件
    def save(self):
        npy_file = './q_table.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')

    # 从文件中读取数据到Q表格中
    def restore(self, npy_file='./q_table.npy'):
        self.Q = np.load(npy_file)
        print(npy_file + ' loaded.')


def run_episode(env, agent, render=False):
    total_steps = 0  # 记录每个episode走了多少step
    total_reward = 0  # 记录每一局游戏的总奖励

    obs = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
    action = agent.sample(obs)  # 根据算法选择一个动作

    while True:
        next_obs, reward, done, _ = env.step(action)  # 与环境进行一个交互，执行动作
        next_action = agent.sample(next_obs)  # 根据算法选择下一个动作
        # 训练 Sarsa 算法
        agent.learn(obs, action, reward, next_obs, next_action, done)
        # obs 执行动作前的状态，action 执行的动作，得到预测的 Q0
        # reward 执行动作后的奖励，next_obs 执行动作后的状态，next_action 选择的下一个动作，得到更新的 Q0
        # done 判断游戏是否结束

        action = next_action  # 迭代新的动作
        obs = next_obs  # 存储上一个观察值，迭代新的状态
        total_reward += reward  # 累计奖励
        total_steps += 1  # 计算step数
        if render:  # 判断是否需要渲染图形显示
            env.render()  # 渲染新的一帧图形
        if done:  # 游戏结束
            break  # 跳出循环，即结束本局游戏
    return total_reward, total_steps  # 返回总的奖励和总的步数


def test_episode(env, agent):
    total_reward = 0  # 记录总的奖励
    obs = env.reset()  # 重置环境，obs 初始观察值，即初始状态
    while True:
        action = agent.predict(obs)  # greedy，每次选择最优动作
        next_obs, reward, done, _ = env.step(action)  # 交互后，获取新的状态，奖励，游戏是否结束
        total_reward += reward  # 累计奖励
        obs = next_obs  # 迭代更新状态
        time.sleep(0.5)  # 休眠，以便于我们观察渲染的图形
        env.render()  # 渲染图形显示
        if done:  # 游戏结束
            break  # 跳出循环
    return total_reward  # 返回最终累计奖励

# 使用gym创建迷宫环境，设置is_slippery为False降低环境难度
env = gym.make("FrozenLake-v0", is_slippery=False)  # 0 left, 1 down, 2 right, 3 up
# 使用 make 方法创建需要的环境

# 创建一个agent实例，输入超参数
agent = SarsaAgent(
        obs_n=env.observation_space.n, # 16 个状态代表这个环境中 4*4 一共 16 个格子
        act_n=env.action_space.n, # 4 种动作选择：0 left, 1 down, 2 right, 3 up
        learning_rate=0.1, # 学习速率
        gamma=0.9, # 下一步的影响率
        e_greed=0.1) # 随机选择概率


# 训练500个episode，打印每个episode的分数
for episode in range(500):
    ep_reward, ep_steps = run_episode(env, agent, False)
    print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))

# 全部训练结束，查看算法效果
test_reward = test_episode(env, agent)
print('test reward = %.1f' % (test_reward))
