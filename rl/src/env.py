'''
Environment modeling churn probability and customer satisfaction.
'''

import random
from rl.src.interfaces import Env

CHURN_PROB = [0.4, 0.2, 0.08, 0.02]
WAIT, CALL, DISCOUNT = "wait", "call", "discount"
PRICE, SERVICE = "price", "service"


class BasicEnv(Env):
    '''
    Basic simulation of a customer over a time period.
    '''
    def __init__(self, months=12, seed=49):
        self.months = months
        self.rng = random.Random(seed)
        self.sat = 0
        self.last = WAIT
        self.month = 0

    def reset(self):
        '''
        Setup new customer.
        '''
        self.sat = self.rng.randint(1, 3)
        self.last = WAIT
        self.month = 0
        return (self.sat, self.last)

    def actions(self, state):
        '''
        Actions for reference.
        '''
        return [WAIT, CALL, DISCOUNT]

    def step(self, action):
        '''
        A step in the environment: 
        
            act -> cost + effect -> drift -> log -> churn check.
        '''
        # action vs. cost + effect
        if action == WAIT:
            cost = 0
        elif action == CALL:
            cost = 1
            if self.rng.random() < 0.6:
                self.sat = min(3, self.sat + 1) # increase 1; cap 3
        elif action == DISCOUNT:
            cost = 4
            if self.last != DISCOUNT:
                self.sat = min(3, self.sat + 2) # increase 2; cap 3

        # random drift
        if self.rng.random() < 0.3:
            self.sat = max(0, self.sat - 1)

        # log
        self.last = action
        self.month += 1

        # churn check
        if self.rng.random() < CHURN_PROB[self.sat]:
            reward = -cost - 20 # if they leave
            done = True
        else:
            reward = 10 - cost # if they stay
            done = (self.month >= self.months)

        return (self.sat, self.last), reward, done


class RealEnv(BasicEnv):
    '''
    More realistic simulation of a customer over a time period.
    '''
    def __init__(self, months=12, seed=49):
        super().__init__(months=months, seed=seed)
        self.cust_type = PRICE
        self.disc_used = 0

    def reset(self):
        '''
        Setup new customer.
        '''
        self.sat = self.rng.randint(1, 3)
        self.last = WAIT
        self.month = 0
        self.cust_type = self.rng.choice([PRICE, SERVICE])
        self.disc_used = 0
        return (self.sat, self.last, self.cust_type, self.disc_used)

    def step(self, action):
        '''
        A step in the environment: 
        
            act -> cost + effect -> drift -> log -> churn check.
        '''
        # action vs. cost + effect
        if action == WAIT:
            cost = 0
        elif action == CALL:
            cost = 1
            # service-type like calls; price-type less so
            call_prob = 0.80 if self.cust_type == SERVICE else 0.30
            if self.last == CALL:
                call_prob *= 0.50 # half as effective if repeated
            if self.rng.random() < call_prob:
                self.sat = min(3, self.sat + 1)
        elif action == DISCOUNT:
            cost = 4
            if self.last != DISCOUNT:
                # price-type get more boost
                base_boost = 3 if self.cust_type == PRICE else 1
                # service-type get less if already used discount
                if self.cust_type == PRICE:
                    boost = max(0, int(base_boost - 0.5 * self.disc_used))
                else:
                    boost = max(0, int(base_boost - self.disc_used))
                self.sat = min(3, self.sat + boost)
            self.disc_used = min(2, self.disc_used + 1)

        # random drift
        if self.rng.random() < 0.30:
            self.sat = max(0, self.sat - 1)

        # log
        self.last = action
        self.month += 1

        # churn check
        if self.rng.random() < CHURN_PROB[self.sat]:
            reward = -cost - 20 # if they leave
            done = True
        else:
            reward = 10 - cost # if they stay
            done = (self.month >= self.months)

        state = (self.sat, self.last, self.cust_type, self.disc_used)
        return state, reward, done