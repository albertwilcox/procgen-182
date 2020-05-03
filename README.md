# ProcGen

Creating an agent to solve ProcGen tasks for CS 182 at Berkeley.

https://openai.com/blog/procgen-benchmark/

## To run:

## Training
Fruitbot:

`python3 train_dqn.py procgen:procgen-fruitbot-v0 --num_steps 600000 --seed 42069 --double_q --multistep`

CartPole:

`python3 train_dqn.py CartPole-v0 --num_steps 100000 --seed 42069 --double_q --multistep`

You might need to run `brew install ffmpeg`

## Testing:

Fruitbot:

`python3 test_dqn.py procgen:procgen-fruitbot-v0 --seed 42069 --num_tests 5 --load_loc [MODEL FILE LOC] --render`

Cartpole:

`python3 test_dqn.py CartPole-v0 --seed 42069 --num_tests 5 --load_loc [MODEL FILE LOC] --render`

## Playing:
`python3 -m procgen.interactive --env-name fruitbot --distribution-mode easy`

## Sources:

ProcGen Info: https://arxiv.org/pdf/1912.01588.pdf

PPO: 
* https://openai.com/blog/openai-baselines-ppo/
* https://spinningup.openai.com/en/latest/algorithms/ppo.html

Rainbow DQN: https://arxiv.org/pdf/1710.02298.pdf

Dist DQN: 
* https://arxiv.org/pdf/1707.06887.pdf
* https://flyyufelix.github.io/2017/10/24/distributional-bellman.html