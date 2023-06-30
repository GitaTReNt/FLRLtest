import numpy as np
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, env, model="SARSA", alpha=0.5, gamma=0.9, e_greed=1, e_decay=0.001,
                 e_greedy_avaliable=True):
        super(Agent, self).__init__()
        self.env = env
        self.action_n = env.action_space.n
        self.state_n = env.observation_space.n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = e_greed
        self.epsilon_decay = e_decay
        self.Q = np.zeros((self.state_n, self.action_n))
        self.model = model
        self.e_greedy_avaliable = e_greedy_avaliable
        self.outcome_list = []

    def sample(self, state):
        # Choose the action with the highest value in the current state
        if self.e_greedy_avaliable and np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            if np.max(self.Q[int(state)]) > 0:
                action = np.argmax(self.Q[int(state)])

            # If there's no best action (only zeros), take a random one
            else:
                action = self.env.action_space.sample()

        return action

    def train(self, state):
        action = self.sample(state)
        new_state, reward, done, _, info = self.env.step(action)
        new_action = self.sample(new_state)

        if self.model == 'q_learning':
            self.Q[int(state), action] = self.Q[int(state), action] + self.alpha * (reward + self.gamma *
                                                                                    np.max(self.Q[int(new_state)]) -
                                                                                    self.Q[int(state), action])
        elif self.model == 'SARSA':
            if done:
                self.Q[int(state), action] = self.Q[int(state), action] + self.alpha * (
                        reward - self.Q[int(state), action])
            else:
                self.Q[int(state), action] = self.Q[int(state), action] + self.alpha * (
                       reward + self.gamma * self.Q[int(new_state), new_action] - self.Q[int(state), action])
        return done, new_state, reward

    def predict(self, state):
        if np.max(self.Q[int(state)]) > 0:
            action = np.argmax(self.Q[int(state)])

            # If there's no best action (only zeros), take a random one
        else:
            action = self.env.action_space.sample()

        new_state, reward, done, _, info = self.env.step(action)
        return done, new_state, reward

    def agentReset(self):
        state, _dict = self.env.reset()
        done = False
        return state, done

    def save(self):
        npy_file = 'q_table_' + self.model + '.npy'
        np.save(npy_file, self.Q)
        print(npy_file + ' saved.')




        # 从文件中读取数据到Q表格中

    def load(self, npy_file='q_table.npy'):
        self.Q = np.load(npy_file.split('.')[0] + '_' + self.model + '.' + npy_file.split('.')[1])
        print(npy_file + ' loaded.')

    def resultPaint(self):
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams.update({'font.size': 17})
        plt.figure(figsize=(12, 5))
        plt.xlabel("Run number")
        plt.ylabel("Outcome")
        ax = plt.gca()
        ax.set_facecolor('#efeeea')
        plt.bar(range(len(self.outcome_list)), self.outcome_list, color="#0A047A", width=1.0)
        plt.show()
