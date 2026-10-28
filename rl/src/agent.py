'''
RL agents learning to prevent churn.
'''

import random
from rl.src.interfaces import Agent


class DumbAgent(Agent):
    '''
    Agent that takes a fixed action.
    '''
    def __init__(self, action):
        self.action = action

    def act(self, state):
        return self.action

    def update(self, s, a, r, s2, done):
        pass


class QLAgent(Agent):
    '''
    Q-Learning agent with epsilon-greedy and sample mean updates.
    '''
    def __init__(self, actions, gamma=1.0, epsilon=0.1, seed=49):
        self.actions = list(actions)
        self.gamma = gamma # discount factor
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.q = {} # Q-table
        self.n = {} # visit count

    def get_q(self, state, action):
        '''
        Return Q-val for (s,t); defaults to 0.
        '''
        return self.q.get((state, action), 0.0)

    def best_action(self, state):
        '''
        Return highest Q-val action; break ties randomly.
        '''
        best_q = max(self.get_q(state, a) for a in self.actions)
        best_actions = [a for a in self.actions if self.get_q(state, a) == best_q]
        return self.rng.choice(best_actions)

    def act(self, state):
        '''
        Return action, chosen with epsilon-greedy.
        '''
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions)
        return self.best_action(state)

    def update(self, s, a, r, s2, done):
        '''
        Update Q-table with sample-mean.
        '''
        if done: # terminal
            target = r
        else:
            best_next_q = max(self.get_q(s2, next_a) for next_a in self.actions) # max_a Q(s',a)
            target = r + self.gamma * best_next_q

        self.n[(s, a)] = self.n.get((s, a), 0) + 1 # visit count (N)
        old_q = self.get_q(s, a)
        self.q[(s, a)] = old_q + (target - old_q) / self.n[(s, a)] # Q(s,a) <- Q(s,a) + error * 1/N