'''
Extension to interface classes (jamshidi.py).
'''

from rl.src.jamshidi import (
    Env,
    Agent,
    RandomAgent,
    EpsilonGreedyBandit,
    Bandit,
    run_episode,
    train,
)


def eval_episode(env, agent):
    '''
    Episode without learning.
    '''
    s = env.reset()
    done = False
    total_reward = 0.0
    r = 0.0
    while not done:
        if hasattr(agent, "best_action"):
            a = agent.best_action(s)
        else:
            a = agent.act(s)
        s, r, done = env.step(a)
        total_reward += r
    survived = (r > 0) # churn implies r <= -20
    return total_reward, survived

def evaluate(env, agent, episodes):
    '''
    Eval results over episodes.
    '''
    results = [eval_episode(env, agent) for _ in range(episodes)]
    rewards = [r for r, _ in results]
    saved = sum(1 for _, survived in results if survived)
    return rewards, saved