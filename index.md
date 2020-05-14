{% include mathjax.html %}
# CS 182 Final Project - ProcGen: Fruitbot

## Introduction

TODO: brief summary of what we did and what our results were

## Problem Statement


### Fruitbot
ProcGen is a set of RL tasks designed my OpenAI to investigate reinforcement learning agents' abilities to learn 
generalizable skills. All ProcGen environments have a variety of procedurally generated levels to train and test on.

For this project we focussed on training a model to play the ProcGen environment known as Fruitbot, 
in which the agent is a robot who can move left and right across a scrolling screen, trying to collide with fruit 
while avoiding other foods such as fried eggs and ice cream. At the same time, the agent must learn to avoid
walls by making its way through gaps in them. A demonstration of the environment, played by a human, can be found below.

{% include image.html url="https://i.imgur.com/oMhGdd8.gif" description="The Fruitbot environment" %}

### Environment Details
The observation returned by the environment is a 64x64x3 tensor encoding an image of the game. 
The action space is discrete, with 15 possible actions. The robot is only capable of moving left,
moving right, doing nothing, and shooting a key, so these actions are duplicated to fill up the
action space.

The environment gives a reward of +1 when the agent collides with a piece of fruit, but gives a reward
of -3 when it collides with a non-fruit object, such as ice cream. If the agent reaches the end of 
a level, it gets a large reward of +20.

As with most reinforcement learning tasks, our agent will be evaluated based on its average reward
on training and testing levels.

### Motivation
A large problem for reinforcement learning researchers is that agents can overfit very large datasets.
Even in the arcade learning environment, which is considered the "golden standard in RL",  overfitting 
can still be a problem. As a result, we use the ProcGen environment to train agents that can generalize
well even if the dataset is small. Below, we show training and test experiments that may increase test-time
performance.
 
## Background

### Reinforcement Learning

Reinforcement learning refers to a class of algorithms in which an agent acts in an environment,
makes observations, recieves rewards, and based on these things learns how to better act in that
environment. This feedback loop is shown in the diagram below.

{% include image.html url="https://www.researchgate.net/profile/Roohollah_Amiri/publication/323867253/figure/fig2/AS:606095550738432@1521515848671/Reinforcement-Learning-Agent-and-Environment.png" description="Source: researchgate.net" %}

While there are a wide variety of reinforcement learning algorithms, the main two that we considered are Proximal 
Policy Optimization and Deep Q Learning.

### Proximal Policy Optimization

Proximal Policy Optimization (PPO) is a policy gradient method, meaning it attempts to estimate the gradient of
the agent's reward with respect to the policy parameters. Unfortunately the reward functions used in reinforcement
learning are not differentiable so we cannot do this directly.

The main idea behind PPO is that, with each update, the divergence of the new policy from the old should
be limited. Defining 

$$\operatorname{clip}(x, a, b) = \begin{cases} a & x < a \\ b & x > b \\ x & \text{else} \end{cases}$$

and 

$$r_t(\theta) = \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta_\text{old}}(a_t | s_t)}$$ 

the loss function for PPO is 

$$L(\theta) = \mathbb{E}\left[\min \left( r_t(\theta)\hat{A}_t, \operatorname{clip}(r_t(\theta), 
1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]\$$

where the expectation is taken empirically and the $$A_t$$ terms are advantage estimates calculated based on estimates
of each state's value (done by another neural network that is trained alongside the policy network) and the
corresponding rewards received.

### Deep Q Learning

Deep Q Learning (abbreviated DQN for Deep Q Network) is a value-based reinforcement learning algorithm, meaning it attempts to learn a value function $$V$$ 
which maps a state to the expected discounted reward to go (the sum of future rewards where distant rewards lose
their weight exponentially) for a policy $$\pi$$. That is,

$$V^{\pi}(s) = \mathbb{E}_\pi[r_0 + \gamma r_1 + \gamma^2 r_2 + \ldots | s_0=s]$$

with DQN, we attempt to learn the Q function, which predicts future rewards based on the current state and action:

$$Q^{\pi}(s, a) = \mathbb{E}_\pi[r_0 + \gamma r_1 + \gamma^2 r_2 + \ldots | s_0=s, a_0 = a]$$

and act by choosing the best action based on this Q function,

$$a_{t} = \operatorname{arg\,max}_a Q(s_t, a)$$

To train a DQN, we keep an older Q function model, parametrized by $$\theta_\text{target}$$ that we update 
periodically with the values from a moving parametrization, $$\theta$$. Then, we pretend that the output
of the target Q function is completely accuracte, and formulate the loss so the moving Q function predicts
a Q value based on that target:

$$L(\theta) = \mathbb{E}_{(s_t,a_t,r_t,s_{t+1})\sim D}
\left[ 
\Big(r_t + \gamma \max_{a \in \mathcal{A}} Q_{\theta^-}(s_{t+1},a) - Q_\theta(s_t,a_t)\Big)^2
\right]$$

where $$D$$ is an experience replay buffer storing a fixed amount (usually 1,000,000) of past frames.

The DQN can be improved upon in a variety of ways, and Rainbow DQN is simply a DQN that uses all those improvements.

## Approach
We originally planned to implement a Rainbow DQN, but according to the ProcGen paper, Fruitbot
had the best results using a Proximal Policy Optimization algorithm so we decided to use that.

### Data Preparation
We wrap the environment in a baselines VecNormalize wrapper, which normalizes the observations and rewards based on a moving average.

Additionally, we tried to improve our reward with reward engineering, adding small rewards for living and penalties for
dying. However, this did not lead to any real improvement and our models performed fine without this.

Finally, because the fruitbot environment only has 4 possible actions, we tried limiting the output space to just those 
4 possible actions. However, this introduced extra complexity to our code and even caused our agent to fail to learn, 
and only saved relatively few model parameters, so we decided against this in the end.

Although some environments require multiple frames to be passed into the model to get enough information to act,
the ProcGen paper showed that this is not the case for Fruitbot, which performs just as well given only one
frame as it does given multiple. Therefore we opted against passing in multiple frames.

### Tools
For neural network building we decided to use Tensorflow. This is because at first, when we planned to use
a DQN, it was convenient to use the Tensorflow implementations we already had.

Although we did implement a PPO algorithm that solved the CartPole task, we found that it performed 
significantly worse than the OpenAI PPO2 baseline, so we decided to use that for our experiments.


### Baseline Model 

According to the ProcGen paper the IMPALA CNN model was the only one that had any real success for 
the FruitBot task, so we decided to continue using that for our experiments. The architecture
of that model is shown below. The model involves three duplicates of of the block shown, where 
each block has the number of filters explained in the diagram.
 
{% include image.html url="https://i.imgur.com/7z9RkQc.png" description="The default IMPALA model architecture" %}

The following visualization shows how the model trained on 250 training levels performed on test levels over time.

{% include image.html url="https://i.imgur.com/iUUVthe.gif" description="An agent trained on 250 training levels" %}

By the time this model reaches 50 million timesteps of training it has far surpassed human performance, consuming
as much fruit as possible, and hitting no junk food unless it has no other choice. It even opts not to pursue 
fruit that will put it in a position where it can't avoid junk food.

## Experiments and Results

Plotting note: the error bars shown in the plots below show the maximum and minimum from the trajectories
over which we averaged to make the plots.

### Number of Training Levels
We trained our baseline model for 10 million time steps each for models with 50, 100, 250 and 500 training levels.
As expected, models trained on more levels saw consistently better test level performance.

{% include image.html url="https://imgur.com/vem5u9J.png" description="The performance of our baseline models" %}

Because it seemed like 250 and 500 training levels both generalized nearly perfectly to the test data, we decided to limit the number of training levels to 100 for our future experiments. While we technically have access to as much as 100k levels, limiting the amount of training data allows us to simulate a scenario where we don't have access to such a large dataset.

### Number of Parallel Environments
The baseline model used 64 parallel environments to train, but we experimented with using only 16 parallel environments, and found it to generalize much better than the baseline:

{% include image.html url="https://imgur.com/nZxefl7.png" description="The effect of reduced number of parallel environments on model performance for the baseline IMPALA architecture." %}

This could essentially be thought of as reducing the batch size, in that each learning step will only recieve the gradients from a batch of 16 episodes, instead of 64. Since the losses in this PPO algorithm are determined by taking the mean over a batch, reducing the size of a batch increases the noise in the gradient. Noisier gradients can help prevent a network from overfitting, making it generalize better.

Additionally, with 16 parallel environments it tends to converge more quickly than with 64. This makes sense, because with 1/4 the batch size, 4x as many training updates will be performed in the same number of steps.

Since this performed so well, we did all of our experiments after this one with 16 environments instead of 64. We also trained some models at 50, 250, and 500 levels with 16 environments: 

### Model Architecture

#### Varying Size

We tried modifying the IMPALA architecture by removing one of the "duplicated" convolutional sections, as less complex networks tend to overfit less. Somewhat unintuitively, this actually made the network generalize much more poorly. This probably happened because each convolutional sequence in the Impala architecture consists of a pooling layer, and removing that pooling actually ended up tripling the number of network parameters.

However, *adding* an extra convolutional sequence actually *did* reduce the number of parameters, because of the additional pooling operation. This seemed to generalize comparably to the test set as the baseline, but because of the reduced number of parameters, it probably isn't able to learn a very complex function, evidenced by its low training accuracy. Thus, we didn't see much benefit to using this, either.

{% include image.html url="https://imgur.com/8qZVbfL.png" description="The performance of models of varying depth. 
The 'Deep' model has an extra convolutional block and the 'Shallow' model has one less." %}


#### BatchNorm
Since batchnorm can act as a regularizer, we tried applying spatial batchnorm layers before every convolutional layer (after each ReLU nonlinearity), as well as once before the fully connected layer:

{% include image.html url="https://imgur.com/dChZakH.png" description="The performance of models with and without batchnorm." %}

It seemed to perform slightly worse than the baseline, and was highly inconsistent. Thus, it didn't seem very promising in terms of improvement.

Since batch normalization also often allows one to train a neural network more quickly, we also tried increasing the learning rate by 4x and seeing if it could effectively train with fewer iterations. However, this performed terribly, not reaching an average reward of even 15, so we decided not pursue this further.

### Ensembles
We took some of the models trained previously and using them in an ensemble. Specifically, for a given model, we assemble an ensemble from it along with some of its intermediate models from training. We choose an action from this ensemble by holding a simple majority vote, with more weight given to more recent models from training. (Each subsequent past intermediate model is given 0.95x the weight of the previous model). We found this to improve generalization significantly across the board, especially for networks trained on fewer levels. 

We provided more results with ensembles in the final results, but this table compares differing numbers of intermediate networks in the ensemble, where we find an ensemble of 10 to be an improvement across the board:

$$\begin{table}[]
\begin{tabular}{|l|l|l|l}
\cline{1-3}
         & Trial 1 & Trial 2 & Average \\ \cline{1-3}
1 Pred.  & 22.53   & 22.01   & 22.27   \\ \cline{1-3}
3 Pred   & 20.68   & 21.54   & 21.11   \\ \cline{1-3}
10 Pred. & 24.05   & 23.54   & 23.80   \\ \cline{1-3}
\end{tabular}
\end{table}$$


### Final results

For our final results, we used 16 parallel environments to train IMPALA network, evaluated their average test reward in an ensemble with 10 intermediate models, and compared that to the baseline, both with and without ensembles. The different colums represent the number of training samples, while the different rows indicate multiple training runs.

16 parallel environments, no ensemble
$$\begin{table}[]
\begin{tabular}{|l|l|l|l|l|}
\hline
        & 50    & 100   & 250   & 500   \\ \hline
Trial 1 & 16.02 & 22.53 & 22.46 & 25.87 \\ \hline
Trial 2 & 18.55 & 22.01 & 22.55 & 22.69 \\ \hline
Mean    & 17.29 & 22.27 & 22.51 & 24.28 \\ \hline
\end{tabular}
\end{table}$$

Baseline (64 parallel envs), no ensemble
$$\begin{table}[]
\begin{tabular}{|l|l|l|l|l|}
\hline
        & 50    & 100   & 250   & 500   \\ \hline
Trial 1 & 13.04 & 17.87 & 22.34 & 25.03 \\ \hline
Trial 2 & 15.89 & 21.57 & 21.34 & 22.26 \\ \hline
Mean    & 14.46 & 19.72 & 21.84 & 23.65 \\ \hline
\end{tabular}
\end{table}$$


16 parallel envs, ensemble of 10
$$\begin{table}[]
\begin{tabular}{|l|l|l|l|l|}
\hline
        & 50    & 100   & 250   & 500   \\ \hline
Trial 1 & 20.20 & 24.05 & 24.40 & 25.91 \\ \hline
Trial 2 & 22.58 & 23.54 & 24.12 & 25.53 \\ \hline
Mean    & 21.39 & 23.80 & 24.26 & 25.72 \\ \hline
\end{tabular}
\end{table}$$

Baseline (64 parallel envs), ensemble of 10
$$\begin{table}[]
\begin{tabular}{|l|l|l|l|l|}
\hline
        & 50    & 100   & 250   & 500   \\ \hline
Trial 1 & 18.17 & 17.34 & 23.27 & 26.34 \\ \hline
Trial 2 & 16.34 & 23.05 & 21.27 & 25.43 \\ \hline
Mean    & 17.26 & 20.20 & 22.27 & 25.89 \\ \hline
\end{tabular}
\end{table}$$

Overall, we found that 16 parallel environments to have consistently outperformed the baseline of 64 for 50, 100, and 250 training levels, both with and without ensembles, outperforming in every single trial. However, as our other change (reducing the batch size to 16) mainly targets overfitting, and 500 training levels is enough to generalize easily, the baseline performs similarly at that point.

As for ensembles, using an ensemble of 10 performed better than the baseline in every single trial that we performed. This is especially exciting because our method of using ensembles (voting alongside earlier iterations of yourself) requires no extra training time or data. It across the board improved the test set reward, for a relatively cost-free change.

## Team Contributions

### Albert
#### Time
~ 30 Hours

#### Contributions

* Wrote the original PPO implementation
* Wrote the code to gather data from trained models
* Made all the graphs and plotting code
* Set up the GitHub site for the report and wrote a good amount of it
* Made the training visualization

### Dylan
#### Time
~ 30 Hours

#### Contributions
* Wrote the parallelized version of Q-networks, replay
* Trained all of the models
* Wrote the ensemble code
* Wrote the experiments section of the website


### Victor
#### Time
~ 4 hours

#### Contributions
* Helped run a couple of preliminary experiments with the models
* Gave a couple pointers on model performance and what to improve
* Wrote a couple of subsections of the report

## References

ProcGen Info: https://arxiv.org/pdf/1912.01588.pdf

Reference Code: https://github.com/openai/train-procgen

PPO: 
* https://arxiv.org/pdf/1707.06347.pdf
* https://openai.com/blog/openai-baselines-ppo/
* https://spinningup.openai.com/en/latest/algorithms/ppo.html

Rainbow DQN: https://arxiv.org/pdf/1710.02298.pdf

Dist DQN: 
* https://arxiv.org/pdf/1707.06887.pdf
* https://flyyufelix.github.io/2017/10/24/distributional-bellman.html

IMPALA Network Architecture: https://arxiv.org/pdf/1802.01561.pdf
