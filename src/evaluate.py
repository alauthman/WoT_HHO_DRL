import argparse
import torch
from src.environment import WoTEnv
from src.agent import DQNAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['cic_iot', 'bot_iot'])
    args = parser.parse_args()

    env = WoTEnv(args.dataset)
    state_dim, action_dim = env.state_dim, env.action_dim
    agent = DQNAgent(state_dim, action_dim, [64, 64], 1e-3, 0.99, 1.0, 0.01, 0.995)
    agent.model.load_state_dict(torch.load(args.model_path))
    agent.model.eval()

    total_reward = 0
    for ep in range(10):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

    print(f'Average reward over 10 eval episodes: {total_reward / 10}')

if __name__ == '__main__':
    main()
