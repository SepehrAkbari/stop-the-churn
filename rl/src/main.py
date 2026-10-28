'''
CLI to train and evaluate agents.
'''

import argparse
import statistics
from rl.src.interfaces import RandomAgent, train, evaluate
from rl.src.env import BasicEnv, RealEnv, WAIT, CALL, DISCOUNT
from rl.src.agent import DumbAgent, QLAgent

ACTIONS = [WAIT, CALL, DISCOUNT]


def print_policy(agent):
    '''
    Print learned policy.
    '''
    print(f"\n{'State':<32} | {'Best Action':<12} | {'Q(wait, call, disc)'}")
    print("-" * 100)
    states = sorted({s for s, _ in agent.q})
    for s in states:
        best_a = agent.best_action(s)
        q_vals = ", ".join(f"{agent.get_q(s, a):5.1f}" for a in ACTIONS)
        print(f"{str(s):<32} | {best_a:<12} | [{q_vals}]")

def parse():
    '''
    Parse CLI arguments.
    '''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", choices=["basic", "real"], default="basic", help="environment type")
    parser.add_argument("--agent", choices=["ql", "dumb", "random"], default="ql", help="agent type")
    parser.add_argument("--action", choices=ACTIONS, default=CALL, help="fixed action for dumb agent")
    parser.add_argument("--episodes", type=int, default=50000, help="training episodes for ql")
    parser.add_argument("--eval-episodes", type=int, default=5000, help="customer count for evaluation")
    parser.add_argument("--gamma", type=float, default=1.0, help="discount factor")
    parser.add_argument("--epsilon", type=float, default=0.1, help="exploration rate")
    parser.add_argument("--seed", type=int, default=49, help="random seed")
    parser.add_argument("--show-policy", action="store_true", help="print learned policy")
    return parser.parse_args()

def setup(args):
    '''
    Initialize env and agent.
    '''
    ENV = BasicEnv if args.env == "basic" else RealEnv

    if args.agent == "ql":
        AGENT = QLAgent(ACTIONS, gamma=args.gamma, epsilon=args.epsilon, seed=args.seed)
        train(ENV(seed=args.seed), AGENT, episodes=args.episodes)
        label = "Q-learning"
    elif args.agent == "dumb":
        AGENT = DumbAgent(args.action)
        label = f"Dumb ({args.action})"
    else:
        AGENT = RandomAgent(ACTIONS, seed=args.seed)
        label = "Random"

    return ENV, AGENT, label


if __name__ == "__main__":
    args = parse()
    env, agent, label = setup(args)

    rewards, saved = evaluate(env(seed=args.seed), agent, args.eval_episodes)
    avg_r = statistics.mean(rewards)

    print(f"[{args.env:<5}] {label:<16} | Avg Reward: {avg_r:5.1f} | Saved: {saved:4d}")

    if args.show_policy and args.agent == "ql":
        print_policy(agent)