# ProcGen

Creating an agent to solve ProcGen tasks for CS 182 at Berkeley.

https://openai.com/blog/procgen-benchmark/

## To run our code:

### Training
Fruitbot:

`python3 train_dqn.py procgen:procgen-fruitbot-v0 --num_steps 600000 --seed 42069 --double_q --multistep`

CartPole:

`python3 train_dqn.py CartPole-v0 --num_steps 100000 --seed 42069 --double_q --multistep`

You might need to run `brew install ffmpeg`

### Testing:

Fruitbot:

`python3 test_model.py procgen:procgen-fruitbot-v0 --seed 42069 --num_tests 5 --load_loc [MODEL FILE LOC] --render`

Cartpole:

`python3 test_model.py CartPole-v0 --seed 42069 --num_tests 5 --load_loc [MODEL FILE LOC] --render`

### Playing:
`python3 -m procgen.interactive --env-name fruitbot --distribution-mode easy`

## To Run Starter code:

### Installation
```
conda env update --name train-procgen --file train-procgen/environment.yml
conda activate train-procgen
pip install https://github.com/openai/baselines/archive/9ee399f5b20cd70ac0a871927a6cf043b478193f.zip
```

### Running
`insert code here`

## Sources:

ProcGen Info: https://arxiv.org/pdf/1912.01588.pdf

Reference Code: https://github.com/openai/train-procgen

PPO: 
* https://openai.com/blog/openai-baselines-ppo/
* https://spinningup.openai.com/en/latest/algorithms/ppo.html

Rainbow DQN: https://arxiv.org/pdf/1710.02298.pdf

Dist DQN: 
* https://arxiv.org/pdf/1707.06887.pdf
* https://flyyufelix.github.io/2017/10/24/distributional-bellman.html

Network Architecture: https://arxiv.org/pdf/1802.01561.pdf