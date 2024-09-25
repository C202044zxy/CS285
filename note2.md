## Exploration

**Introduction**

How can an agent discover high-reward strategies that require a temporally extended sequence of complex behaviors that individually are not rewarding?

Actually, this problem requires a trade-off between exploitation and exploration: 

- Exploitation: doing what you know to yield highest reward. 
- Exploration: doing things you haven't done before, in the hopes of getting higher reward. 

**Optimistic exploration**

In multi-armed bandits problem, each action produce a reward (no state) and the reward follows an unknown distribution. This problem is theoretically tractable. 

Keep track of average reward $\mu_a$ for each action $a$. For exploitation, we pick the action with the highest average reward. Moreover, we add a bonus for exploration: 
$$
a = \arg\max_a \mu_a + \sqrt{\frac{2\ln T}{N(a)}}
$$
where $N(a)$ is the number of times we picked this action. The bonus encourages to take each action until you are sure it's not great. 

The idea can also be used with MDPs. Just use $N(s)$ or $N(s,a)$ to add exploration bonus: 
$$
r^{+}(s,a)=r(s,a)+\sqrt{\frac{2\ln T}{N(s)}}
$$
However in real world problem, we may never see the same thing twice (e.g. continuous). But we can give higher bonus for the states that are less similar than others. 

One way to estimate bonus is using pseudo-counts. Specifically, we fit a model to estimate $p_\theta(s_i)$, representing the density of state $s_i$. After add the state $s_i$ into the dataset, we can obtain the new density $p_{\theta'}(s_i)$. To compute $N(s_i)$, use the following equation: 
$$
p_\theta(s_i) = \frac{N(s_i)}{n}\\
p_{\theta'}(s_i) = \frac{N(s_i)+1}{n+1}
$$
It's easy to solve the equations since there are two equations and two unknowns: 
$$
N(s_i) = \frac{1-p_{\theta'}(s_i)}{p_{\theta'}(s_i)-p_{\theta}(s_i)}p_{\theta}(s_i)
$$

**Random Network Distillation**

As we all know, neural networks produces error in the region of unseen datapoints. The Random Network Distillation (RND) algorithm benefits from this property. 

RND algorithm gives bonus to the region where the neural network error is high. Formally, let $f_{\theta}(s)$ be a randomly chosen vector-valued function represented by a neural network. RND trains another neural network $f_{\phi}(s)$ to match $f_\theta(s)$ under the distribution of dataset: 
$$
\phi = \arg\min_{\phi}\mathbb{E}_{(s,a,s')\sim \mathcal{D}} ||f_{\theta}(s') - f_{\phi}(s')||^2
$$
 If a transition $(s,a,s')$ is in the dataset, the prediction error is expected to be small. On the other hand, for all unseen state-action tuples it is expected to be large. Therefore, the prediction error can be used as exploration bonus: 
$$
r^{+}(s, a) = r(s, a) + ||f_{\theta}(s') - f_{\phi}(s')||^2
$$

## Offline Reinforcement Learning

**Out of Distribution**

In offline RL, the agent has access to a pre-collected dataset $\mathcal{D}$ , which is gathered by a policy $\pi_{\beta}(a|s)$ (e.g. random policy). And we train another policy $\pi_{new}(s|a)$ such that: 
$$
Q(s,a) \leftarrow r(s,a) + E_{a'\sim \pi_{new}} [Q(s',a')]
$$
The algorithm achieve good accuracy when $\pi_{\beta} = \pi_{new}$. However, it is often the case that $\pi_{\beta} = \pi_{new}$ and our algorithm suffer greatly from this kind of distribution shift. 

<img src = "C:\Users\16549\AppData\Roaming\Typora\typora-user-images\image-20240514195115910.png" width = 500>

From the graph, we can conclude that q-value is overestimated. This is because we overestimate the Q-value of ODD (out of distribution) actions (i.e. the actions not in $\mathcal{D}$ ). 

<img src = "C:\Users\16549\AppData\Roaming\Typora\typora-user-images\image-20240514200517847.png" width = 700>

In online RL setting, if an action is overestimated, we will add that action to the dataset and fix it later on. But in offline RL, we have no chance to fix it since the action is out of distribution. Therefore, existing challenges with sampling error and function approximation error in online RL become much more severe in offline RL. 

**Conservative Q-Learning**

The goal of Conservative Q-Learning (CQL) is to preventing overestimation of the policy value. This is done by training with a regularizer that minimizes the soft-maximum of the Q-values and maximizes the Q-value on the state-action pair seen in the dataset. 

The overall CQL objective is given by the standard TD error objective (i.e. Q-Learning objective) augmented with the CQL regularizer weighted by $\alpha$ :
$$
\frac{\alpha}{N} [\sum_{i=1}^N\log\Big(\sum_a \exp\big(Q(s_i,a)\big)\Big)-Q(s_i,a_i)]
$$

**Advantage Weighted Actor Critic**

The AWAC algorithm augments the training process by utilizing the following actor update:
$$
\theta \leftarrow \arg\max_{\theta} \mathbb{E}_{s,a\sim\beta}[\log \pi_{\theta}(a|s)\exp(\frac{1}{\lambda}A(s,a))]
$$
where $A(s,a)$ is defined as the advantage compared to $\mathbb{E}[Q(s,a)]$ under actor $\pi_{\theta}(a|s)$: 
$$
A(s,a) = Q(s,a) - \mathbb{E}_{a'\sim \pi_{\theta}(a|s)} Q(s,a')
$$
This update is similar to weighted behavior cloning. In the update above, the agent regresses onto high-advantage actions with a large weight, while almost ignoring low-advantage actions. Furthermore, the actor will not assignment probability to ODD actions, since we are sampling from the dataset $\mathcal D$ during training. 

The Q function is learnt with a Temporal Difference (TD) Loss: 
$$
\mathbb{E}[(Q(s,a) - r(s,a) - \gamma \mathbb{E}_{a'\sim \pi_{\theta}} Q(s',a'))^2]
$$
Note that next actions $a'$ are sampled from the learned policy $\pi_{\theta}$, meaning that OOD actions will not be sampled if $\pi_{\theta}$ does a good job of fitting the behavior policy.