## Policy Gradients

**Policy gradients**

Regard $\tau$ as the abbreviation of $s_1, a_1\dots s_T,a_T$. Then: 
$$
p_\theta(\tau) = p_\theta(s_1,a_1\dots s_T,a_t) = p(s_1)\prod_{t=1}^T \pi_{\theta}(a_t|s_t)p(s_{t+1}|s_t,a_t) \\
\theta^{*} =\arg \max_{\theta}E_{\tau\sim p_{\theta}(\tau)} \Big[\sum_{t}  r(s_t,a_t)\Big]
$$
The expectation can be approximated by random sampling: 
$$
J(\theta) = E_{\tau\sim p_{\theta}(\tau)}\Big[\sum_{t} r(s_t,a_t)\Big] \approx \frac{1}{N}\sum_{i=1}^N\sum_t r(s_{i,t}, a_{i,t})
$$
We can differentiate the object directly: 
$$
\begin{align}
J(\theta) &= E_{\tau \sim p_{\theta}(\tau)}r(\tau) = \int p_{\theta}(\tau)r(\tau)d\tau \\
\nabla_{\theta}J(\theta) &= \nabla_{\theta}\int p_{\theta}(\tau)r(\tau)d\tau\\
&= \int \nabla_{\theta}p_{\theta}(\tau)r(\tau)d\tau\\
&= \int p_{\theta}(\tau)\big(\nabla_{\theta}\log p_{\theta}(\tau)\big)r(\tau)d\tau\\
&= E_{r\sim p_{\theta}(\tau)} \nabla_{\theta}\log p_{\theta}(\tau)r(\tau)\\
&= E_{r\sim p_{\theta}(\tau)}\Big[\Big(\sum_{t=1}^T\nabla_{\theta} \log\pi_{\theta}(a_t|s_t)\Big)\Big(\sum_{i=1}^T r(s_t,a_t)\Big)\Big]
\end{align}
$$
where $p_{\theta}\nabla_{\theta}\log p_{\theta}(\tau) = p_{\theta}\frac{\nabla_{\theta}p_{\theta}(\tau)}{p_{\theta}}=\nabla_{\theta}p_{\theta}(\tau)$ is an useful identity. We can use the same way to approximate the policy gradients: 
$$
\nabla_{\theta}J(\theta) \approx \frac{1}{N}\sum_{i=1}^N \Big(\sum_{t=1}^T \nabla_{\theta}\log\pi_{\theta}(a_{i,t}|s_{i,t})\Big)\Big(\sum_{t=1}^Tr(s_{i,t},a_{i,t})\Big)
$$
**Reducing variance**

Because policy gradient bears high variance, we provide two possible solutions to mitigate it. 

Consider the following causality: policy at time $t'$ can't affect reward at time $t$ when $t<t'$. Therefore, we only consider the future reward, instead of summing up all: 
$$
\nabla_{\theta}J(\theta) \approx \frac{1}{N}\sum_{i=1}^N \sum_{t=1}^T \nabla_{\theta}\log\pi_{\theta}(a_{i,t}|s_{i,t})\Big(\sum_{t'=t}^Tr(s_{i,t'},a_{i,t'})\Big)
$$
The second approach is to use baselines. Set $b=\frac{1}{N}\sum_{i=1}^N r(\tau)$ and then:
$$
\nabla_{\theta}J(\theta) \approx \frac{1}{N}\sum_{i=1}^N \nabla_{\theta}\log p_{\theta}(\tau)[r(\tau) - b]
$$
But are we allowed to do that? In fact, subtracting a baseline is unbiased in expectation, i.e. the expectation is zero with respect to $b$ : 
$$
\begin{align}
E[\nabla_{\theta}\log p_{\theta}(\tau)b] &= \int p_{\theta}(\tau) \nabla_{\theta}\log p_{\theta}(\tau)b\ d\tau = \int \nabla_{\theta}p_{\theta}(\tau) b \ d\tau\\
&= b\nabla_{\theta}\int p_{\theta}(\tau)d\tau = b\nabla_{\theta}1 = 0
\end{align}
$$
Average reward is not the best baseline, but it’s pretty good!

## Actor Critic

**Discount**

Reward-to-go can get infinitely large in many cases. This is where *discount* come to rescue. As we all know, it's better to get rewards sooner than later. 

To be specific, we multiply a discount factor $\gamma^{t'-t}$ for reward at time $t'$. Here is the policy gradient with discount: 
$$
\nabla_{\theta}J(\theta) = \frac{1}{N}\sum_{i=1}^N \sum_{t=1}^T \nabla_{\theta}\log\pi_{\theta}(a_{i,t}|s_{i,t})\Big(\sum_{t'=t}^T\gamma^{t'-t} r(s_{i,t'},a_{i,t'})\Big)
$$
**Actor Critic**

Actor Critic is a technique that further reduces variance but introduces bias. Concretely, actor critic uses a neural network to fit $V(s)$ (reward-to-go from state $s$). Simultaneously, it evaluates a advantage $A(s_t,a_t)$ for each action, representing how superior or inferior the action is to average performance (similar to baseline).  
$$
A(s_t,a_t) = r(s_t,a_t) + \gamma V(s_{t+1}) - V(s_t) \\
\nabla_{\theta}J(\theta) = \frac{1}{N}\sum_{i=1}^N \sum_{t=1}^T \nabla_{\theta}\log\pi_{\theta}(a_{i,t}|s_{i,t})A(s_t,a_t)
$$
Actor Critic has high bias because neural network can't always fit reward-to-go perfectly. 

**Bias/Variance Tradeoff** 

Traditional policy gradients have no bias but high variance, while Actor Critic has relative low variance but high variance. Can we trade them off to get better performance? 

<img src = "C:\Users\16549\AppData\Roaming\Typora\typora-user-images\image-20240418155434012.png" width = 450>

As shown in the diagram above, we set a border $n$. Before the time step $n$, we use policy gradient and apply Actor Critic after the border: 
$$
A_n(s_t,a_t) = \sum_{t'=t}^{t+n-1} \gamma^{t'-t} r(s_{t'},a_{t'})-V(s_t)+\gamma^{n}V(s_{t+n})
$$
For generalized advantage estimation: 
$$
\delta_t = r(s_t,a_t) + \gamma V(s_{t+1}) - V(s_t)\\
A_{\text{GAE}}(s_t,a_t) = \sum_{t'=t}^{\infin} (\gamma\lambda)^{t'-t}\delta_{t'} = \delta_t+\gamma \lambda A_{\text{GAE}}(s_{t+1},a_{t+1})
$$

## Q-learning

**Double Q-learning**

The target network of Q-learning is given by: 
$$
y_j \leftarrow  r_j + \gamma\max_{a'_j} Q_{\phi'}(s_j',a_j')
$$
The estimate of Q-values are not accurate. Imagine we have two random value $X_1$ and $X_2$ :
$$
E[\max(X_1,X_2)]\geq \max(E[X_1], E[X_2])
$$
Since $Q_{\phi'}(s_j',a_j')$ is noisy, $\max Q_{\phi'}(s_j',a_j')$ overestimates the next value. Note that: 
$$
\max_{a'_j}Q_{\phi'}(s_j',a_j') = Q_{\phi'}(s_j',\arg\max_{a_j'}Q_{\phi'}(s'_j,a_j'))
$$
If the noise in these two parts are decorrelated, the problem goes away. Therefore, we use two network to choose the action and evaluate value respective: 
$$
Q_{\phi_A}(s,a)\leftarrow r+\gamma Q_{\phi_B}(s,\arg\max_{a} Q_{\phi_A}(s,a)) \\ 
Q_{\phi_B}(s,a)\leftarrow r+\gamma Q_{\phi_A}(s,\arg\max_{a} Q_{\phi_B}(s,a))
$$
In practice, we just use current and target network. Specifically, we use current network to choose action and target network to evaluate value: 
$$
y\leftarrow r+\gamma Q_{\phi'}(s,\arg\max_{a'}Q_{\phi}(s',a'))
$$
Gradient descent is applied to current network. There are two common strategies to update target network: 

- *Hard update*: where every $K$ steps we set $\phi'\leftarrow \phi$
- *Soft update*: at each step, we perform $\phi'\leftarrow (1-\tau) \phi' + \tau\phi$

**Q-learning with continuous actions**

Since $\max_{a} Q_{\phi}(s,a) = Q_{\phi}(s,\arg\max_{a} Q_{\phi}(s,a))$, we can train another network such that: 
$$
\mu(\theta) \approx \arg\max_{\phi}(s,a)
$$
We can use the same gradient estimator that we used in our policy gradients algorithms to update our actor in actor-critic algorithms: 
$$
\nabla_{\theta} E_{a\sim \pi_{\theta}(a|s)}Q(s,a)
$$
This equals policy gradients: 
$$
E_{a\sim \pi_{\theta}(a|s)}[\nabla_{\theta}\log \pi_{\theta}(a|s)\cdot Q(s,a)]
$$
Note that the actions $a$ are sampled from policy $\pi_{\theta}$, and we do not need real world data. Therefore, we can sample more example from the given state to reduce variance. 

**Entropy Bonus**

In DQN, we used an $\epsilon$-greedy strategy to decide which action to take. In continuous spaces, we have several options for generating exploration noise. 

One of the most common ways is providing an *entropy bonus* to encourage the actor to be more random, scaled by a "temperature" coefficient $\beta$. The loss can be written as: 
$$
L = E_{a\sim \pi_{\theta}(a|s)}[\nabla_{\theta}\log \pi_{\theta}(a|s)\cdot Q(s,a)] + \beta \mathcal{H}(\pi(s|a))
$$
where entropy is defined as $\mathcal{H}(\pi (s|a))=E_{a\sim \pi}[-\log \pi(a|s)]$. To make sure entropy is also factored into the Q-function, we should also account for it in our target values:
$$
y_t\leftarrow r_t+\gamma\big[Q_{\theta}(s_t,a_t)+\beta \mathcal{H}(\pi(a_{t+1}|s_{t+1})\big]
$$

## Model-Based Reinforcement Learning

**Open-Loop Planning**

In open-loop planning, the actor takes a couple of actions solely based on its current state without receiving immediate feedback: 

<img src = "C:\Users\16549\AppData\Roaming\Typora\typora-user-images\image-20240430165229493.png" width = 500>

The simplest gradient-free optimizer, also known as *random shotting*, generates $N$ independent random action sequence $\{A_1...A_N\}$, where each sequence $A_i = \{a_1^i...a_{m}^i\}$. Then, we select the sequence with the maximum reward. 

The second approach is called *cross-entropy method*(CEM). The procedure is as follow: 

1. pick $A_1...A_N$ from some distribution (e.g. uniform)
2. evaluate $J(A_1)...J(A_N)$ (i.e. reward of the action sequence)
3. pick the *elites* $A_{i_1}...A_{i_M}$ with the highest value, where $M = \lfloor\frac{N}{10}\rfloor$ 
4. refit $p(A)$ to the elites $A_{i_1}...A_{i_M}$

In step four, we compute the mean and variance of the elites along the first dimension. Then we can use the refined gaussian distribution for sampling in step one. 

The soft update can also be applied in CEM. Specifically: 
$$
\mu_{t}^{m+1} = \alpha \cdot \text{mean}(A) + (1-\alpha)\cdot u_{t}^{m}\\
\sigma_{t}^{m+1} = \alpha\cdot \text{var}(A) + (1-\alpha)\cdot \sigma_{t}^{m}
$$
After $K$ iterations, the optimal actions are selected to be the resulting mean of the action distribution.

**Model-Based Reinforcement Learning**

If we know $f(s_t,a_t) = s_{t+1}$, we could use the tools from the open-loop planning. Otherwise, we can use *model-based reinforcement learning*(MBRL) to learn the dynamics $f(s_t,a_t)=s_{t+1}$ and then plan through. 

Intuitively, we propose the MBRL version $0.5$: 

1. run base policy $\pi_0(a_t|s_t)$ (e.g. random policy) to collect $\mathcal{D} = \{(s,a,s')\}$
2. learn dynamics model $f(s,a)$ to minimize $\sum_i||f(s_i,a_i)-s_i'||_2^2$
3. plan through $f(s,a)$ to choose actions. 

The approach basically works well. However, the fatal flaw is that the action distributions of base policy mismatch that of planning method. 

To address the issue, we add the dynamics of planning into dataset $\mathcal{D}$ and modify the dynamics model $f(s,a)$ (same as DAGGER). The MBRL version $1.0$ is as following: 

1. run base policy $\pi_0(a_t|s_t)$ (e.g. random policy) to collect $\mathcal{D} = \{(s,a,s')_i\}$
2. learn dynamics model $f(s,a)$ to minimize $\sum_i||f(s_i,a_i)-s_i'||_2^2$
3. plan through $f(s,a)$ to choose actions. 
4. execute those actions and the resulting data ${(s,a,s')_j}$ to $\mathcal {D}$

Since our model is erroneous, we will makes mistakes in planning. But in many real world cases, we can refine our mistakes as we are collecting more data. Therefore, in step three, we can take the first action of the best sequence and then immediate replan: 

1. run base policy $\pi_0(a_t|s_t)$ (e.g. random policy) to collect $\mathcal{D} = \{(s,a,s')_i\}$
2. learn dynamics model $f(s,a)$ to minimize $\sum_i||f(s_i,a_i)-s_i'||_2^2$
3. plan through $f(s,a)$ to choose actions. 
4. execute the first planned action, observe resulting state $s'$
5. append ${(s,a,s')_j}$ to $\mathcal {D}$

The more you replan, the less perfect each individual plan needs to be. Additionally, we can use shorter horizons to reduce time complexity. 

**Ensembles**

A simple and effective way to improve predictions is to use an ensemble of models. Rather than training one network $f(\theta)$ to make predictions, we’ll train $N$ independently initialized networks $\{f(\theta_n)\}^N_{n=1}$.

At test time, for each candidate action sequence, we’ll generate $N$ independent rollouts and average the rewards of these rollouts to choose the best action sequence.

Another perspective on this issue is that *the model is certain about the data, but we are not certain about the model*. Consequently, we want to estimate the model uncertainty. Concretely, we want to estimate $p(\theta|\mathcal{D})$, whose entropy tells us the model uncertainty. 

To estimate $p(\theta|\mathcal{D})$, we can train $\theta_i$ on $\mathcal{D}_i$, which is sampled with replacement from $\mathcal{D}$. However, sampling with replacement is not necessary since SGD and random initialization usually make the models sufficiently independent. 

