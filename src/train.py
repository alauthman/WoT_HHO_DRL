import argparse
from src.environment import WoTEnv
from src.agent import DQNAgent
from src.hho import HarrisHawksOptimizer
import numpy as np
import torch

def objective(hyperparams, args):
    lr, gamma, eps_decay, hidden1, hidden2 = hyperparams
    hidden_sizes = [int(hidden1), int(hidden2)]
    env = WoTEnv(args.dataset)
    agent = DQNAgent(env.state_dim, env.action_dim, hidden_sizes,
                     lr, gamma, 1.0, 0.01, eps_decay)
    total_reward = 0
    for ep in range(args.episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition((state, action, reward, next_state, float(done)))
            agent.update()
            state = next_state
            total_reward += reward
    avg_reward = total_reward / args.episodes
    return -avg_reward

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['cic_iot', 'bot_iot'])
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--hho_iters', type=int, default=10)
    args = parser.parse_args()

    dim = 5
    lb = [1e-4, 0.90, 0.995, 32, 32]
    ub = [1e-2, 0.99, 0.999, 256, 256]
    hho = HarrisHawksOptimizer(lambda x: objective(x, args), lb, ub, dim,
                               population_size=10, max_iter=args.hho_iters)
    best_params, best_score = hho.optimize()
    print(f'Best hparams: {best_params}, Score: {-best_score}')

    lr, gamma, eps_decay, h1, h2 = best_params
    env = WoTEnv(args.dataset)
    agent = DQNAgent(env.state_dim, env.action_dim, [int(h1), int(h2)],
                     lr, gamma, 1.0, 0.01, eps_decay)
    for ep in range(args.episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition((state, action, reward, next_state, float(done)))
            agent.update()
            state = next_state
    torch.save(agent.model.state_dict(), 'best_model.pth')
