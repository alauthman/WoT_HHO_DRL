# WoT Security: HHO-Optimized DQN

This repository implements the Harris Hawks Optimization (HHO) meta-learning framework to optimize Deep Q-Network (DQN) hyperparameters for Web of Things (WoT) security, as described in the paper *Enhancing Web of Things Security Using Harris Hawks Optimization with Reinforcement Learning*.

## Repository Structure

```
wot_hho_dqn/
├── README.md
├── requirements.txt
└── src/
    ├── agent.py
    ├── environment.py
    ├── utils.py
    ├── hho.py
    ├── train.py
    └── evaluate.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

- **Meta-training**:  
  `python src/train.py --dataset cic_iot --episodes 100 --hho_iters 20`

- **Evaluation**:  
  `python src/evaluate.py --model_path best_model.pth --dataset bot_iot`
