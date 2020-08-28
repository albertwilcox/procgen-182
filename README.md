# ProcGen

Creating an agent to solve ProcGen tasks for CS 182 at Berkeley.

See our report at https://albertwilcox.github.io/procgen-182/

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
Training code:

`python3 starter_train.py --num_levels 100 --start_level 0`

Testing code:

`python3 starter_test.py`

To create graphs you need to go into the `starter_test.py` file and adjust the parameters the way I have at
the bottom, and then run the code.
