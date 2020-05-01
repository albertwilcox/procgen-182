Creating an agent to solve ProcGen tasks for CS 182 at Berkeley.

https://openai.com/blog/procgen-benchmark/

# To run:

## Training
Fruitbot:

`python3 train_dqn.py procgen:procgen-fruitbot-v0 --num_steps 600000 --seed 42069 --double_q`

CartPole:

`python3 train_dqn.py CartPole-v0 --num_steps 100000 --seed 42069 --double_q`

You might need to run `brew install ffmpeg`

## Testing:

Fruitbot:

`python3 test_dqn.py procgen:procgen-fruitbot-v0 --seed 42069 --num_tests 5 --load_loc [MODEL FILE LOC] --render`

Cartpole:

`python3 test_dqn.py CartPole-v0 --seed 42069 --num_tests 5 --load_loc [MODEL FILE LOC] --render`