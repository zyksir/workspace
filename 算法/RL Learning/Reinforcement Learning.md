[TOC]

### Reinforcement Learning

#### Learning and Value-based Methods

- An RL agent包含一下内容：

  - Policy : agent’s behavior function
  - Value function : how good is a state and/or action
  - Model: agent’s representation of the environment

- Markov Decision Processes $$ \langle\mathcal{S}, \mathscr{A}, \mathscr{P}, \mathscr{R}, \gamma\rangle $$ ：RL 的学习环境

  - $ \mathcal{S} $ -a set of states
  - $\mathscr{A}$- a set of actions
  - $\mathscr{P}$ - transition probability function, $\mathscr{P}_{s s^{\prime}}^{a}=\mathbb{P}\left[S_{t+1}=s^{\prime} \mid S_{t}=s, A_{t}=a\right]$
  - $\mathscr{R}$ - reward function, $\mathscr{R}_{s}^{a}=\mathbb{E}\left[R_{t+1} \mid S_{t}=s, A_{t}=a\right]$
  - $\gamma$ - discounting factor for future reward
  - Polity $\pi$ : $\pi(a \mid s)=\mathbb{P}\left[A_{t}=a \mid S_{t}=s\right]$

- **Value-based Methods** :
   核心在于对 value function(state-value function 或是 action-value) 的估计

   $\begin{aligned} v_{\pi}(s) &=\mathbb{E}_{\pi}\left[G_{t} \mid S_{t}=s\right] \\ &=\mathbb{E}_{\pi}\left[R_{t+1}+\gamma R_{t+2}+\gamma^{2} R_{t+3}+\ldots \mid S_{t}=s\right] \\ &=\mathbb{E}_{\pi}\left[R_{t+1}+\gamma\left(R_{t+2}+\gamma R_{t+3}+\ldots\right) \mid S_{t}=s\right] \\ &=\mathbb{E}_{\pi}\left[R_{t+1}+\gamma G_{t+1} \mid S_{t}=s\right] \\ &=\mathbb{E}_{\pi}\left[R_{t+1}+\gamma v_{\pi}\left(S_{t+1}\right) \mid S_{t}=s\right] \end{aligned}$
   $\begin{aligned} Q_{\pi}(s, a) &=\mathbb{E}_{\pi}\left[R_{t+1}+\gamma v\left(S_{t+1}\right) \mid S_{t}=s, A_{t}=a\right] \\ &=\mathbb{E}_{\pi}\left[R_{t+1}+\gamma \mathbb{E}_{a^{\prime} \sim \pi} Q\left(S_{t+1}, a^{\prime}\right) \mid S_{t}=s, A_{t}=a\right] \end{aligned}$

   state-value function 和 action-value的联系：

     - $v_{\pi}(s)=\sum_{a \in \mathscr{A}} \pi(a \mid s) q_{\pi}(s, a)$

     - $q_{\pi}(s, a)=\mathscr{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathscr{P}_{s s^{\prime}}^{\alpha} v_{\pi}\left(s^{\prime}\right)$

     - $v_{\pi}(s)=\sum_{a \in \mathscr{A}} \pi(a \mid s)\left(\mathscr{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathscr{P}_{s s^{\prime}}^{a} v_{\pi}\left(s^{\prime}\right)\right)$

     - $q_{\pi}(s, a)=\mathscr{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \delta} \mathscr{P}_{s s^{\prime}}^{a} \sum_{a^{\prime} \in \mathscr{A}} \pi\left(a^{\prime} \mid s^{\prime}\right) q_{\pi}\left(s^{\prime}, a^{\prime}\right)$

   Optimal Value Function：最重要的还是最后两个式子，可以根据他们递归求解 value function

   - $v_{*}(s)=\max _{\pi} v_{\pi}(s)$
   - $q_{*}(s, a)=\max _{\pi} q_{\pi}(s, a)$
   - $v_{*}(s)=\max _{a} q_{*}(s, a)$ 
   - $q_{*}(s, a)=\mathscr{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \delta} \mathscr{P}_{s s^{\prime}}^{a} v_{*}\left(s^{\prime}\right)$
   - $v_{*}(s)=\max _{a}\left(\mathscr{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \delta} \mathscr{P}_{s s^{\prime}}^{a} v_{*}\left(s^{\prime}\right)\right)$
   - $q_{*}(s, a)=\mathscr{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathscr{P}_{s s^{\prime}}^{a} \max _{a^{\prime}} q_{*}\left(s^{\prime}, a^{\prime}\right)$

- **Dynamic Programming**：
  假设我们对环境有充分的认知(已知 MDP)，

  - Policy Evaluation：利用迭代法计算给定 policy $\pi$的 value function
    $v^{k+1}=\mathscr{R}^{\pi}+\gamma \mathscr{P}^{\pi} v^{k}$

  -  Policy Improvement：利用贪心法

    $\pi^{\prime}(s)=\underset{a \in \mathscr{A}}{\operatorname{argmax}} q_{\pi}(s, a)$

    $q_{\pi}\left(s, \pi^{\prime}(s)\right)=\max _{a \in \mathscr{A}} q_{\pi}(s, a) \geq q_{\pi}(s, \pi(s))=v_{\pi}(s)$

  - value iteration: update policy every iteration
    $v_{*}(s) \leftarrow \max _{a \in \mathscr{A}}\left(\mathscr{R}_{s}^{a}+\gamma \sum_{s^{\prime} \in \mathcal{S}} \mathscr{P}_{s s^{\prime}}^{a} v_{*}\left(s^{\prime}\right)\right)$

  - policy iteration: 1. 用任意Policy Evaluation估计$v_{\pi}$ 2. 用任意Policy improvement生成$\pi^{\prime}$
    
  - 可以证明，这样迭代下去，最后 value function 会以$\gamma$的比例线性收敛到最优解

  - 每个迭代复杂度是$O(m n ^2)$ ，m 是action 数目，n 是 state 数目

  - 
  
##### Monte-Carlo Methods: 
learn from **complete** episodes
    是unbiased的
    要求episodes必须终止

  - 给定policy $\pi$ ，和环境交互得到 $S_{1}, A_{1}, R_{2}, \ldots, S_{k} \sim \pi$ ，然后计算rewards: $G_{t}=R_{t+1}+\gamma R_{t+2}+\ldots+\gamma^{T-1} R_{T}$
    我们使用平均值来代替期望，算出 $v_{\pi}(s)$
  
  - evaluation阶段：如果多次经过一个状态，每次：$N(s) \leftarrow N(s)+1, S(s) \leftarrow S(s) + G_t$ ，最后计算$V(s) \leftarrow S(s)/N(s)$，这个过程可以等价为：
    $N\left(s_{t}\right) \leftarrow N\left(s_{t}\right)+1$
    $V\left(s_{t}\right) \leftarrow V\left(s_{t}\right)+\frac{1}{N\left(s_{t}\right)}\left(G_{t}-V\left(s_{t}\right)\right)$
  
    并泛化为：$V\left(s_{t}\right) \leftarrow V\left(s_{t}\right)+\alpha\left(G_{t}-V\left(s_{t}\right)\right)$
  
    评估 $Q$，方式与评估状态相似
    $N\left(S_{t}, A_{t}\right) \leftarrow N\left(S_{t}, A_{t}\right)+1$
    $Q\left(S_{t}, A_{t}\right) \leftarrow Q\left(S_{t}, A_{t}\right)+\frac{1}{N\left(S_{t}, A_{t}\right)}\left(G_{t}-Q\left(S_{t}, A_{t}\right)\right)$
    
  - improvement阶段：$\pi(a \mid s)=\left\{\begin{array}{ll}\epsilon / m+1-\epsilon & \text { if } a^{*}=\underset{a \in \mathscr{A}}{\arg \max } Q(s, a) \\ \epsilon / m  &\text { otherwise }\end{array}\right.$
    只要$\epsilon_{k}=\frac{1}{k}$，就可以保证该策略是GLIE的，即每个state-action对都会被无限次的探索到($\lim _{k \rightarrow \infty} N_{k}(s, a)=\infty$)，且该策略会收敛成贪心策略
  
  - 总体就是：sample-》评估Q函数-》修改policy-》循环

##### Sarsa

- TD Learning: learn from **incomplete** episodes -》 是biased的，不要求episodes终止

- Sarsa 伪代码

  1. 初始化$Q(s, a), S$

  2. 根据$Q(s, a), S$，按照$\epsilon-greedy$策略选择$A$

  3. 如下循环直至S为终止态

     1. 观察$R, S^{\prime}$，并根据$Q(s, a), S^{\prime}$选择$A^{\prime}$
      2. $Q(S, A) \leftarrow Q(S, A)+\alpha\left(R+\gamma Q\left(S^{\prime}, A^{\prime}\right)-Q(S, A)\right)$
     3. $S \leftarrow S^{\prime} ; A \leftarrow A^{\prime}$
     
     

- off-policy learning

  - 我们观察到的策略$\mu(a \mid s)$只是用于探索、并不是我们最终使用的目标策略$\pi(a \mid s)$ 

  - 数学依据 importance sampling：$\begin{aligned} \mathbb{E}_{X \sim P}[f(X)] &=\sum P(X) f(X) \\ &=\sum Q(X) \frac{P(X)}{Q(X)} f(X) \\ &=\mathbb{E}_{X \sim Q}\left[\frac{P(X)}{Q(X)} f(X)\right] \end{aligned}$

  - offline MC: $G_{t}^{\pi / \mu}=\frac{\pi\left(A_{t} \mid S_{t}\right)}{\mu\left(A_{t} \mid S_{t}\right)} \frac{\pi\left(A_{t+1} S_{t+1}\right)}{\mu\left(A_{t+1} \mid S_{t+1}\right)} \ldots \frac{\pi\left(A_{T} \mid S_{T}\right)}{\mu\left(A_{T} \mid S_{T}\right)} G_{t}$

    由于importance sampling会大大增加方差，因此不用这个

  - offline TD: $V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left(\frac{\pi\left(A_{t} \mid S_{t}\right)}{\mu\left(A_{t} \mid S_{t}\right)}\left(R_{t+1}+\gamma V\left(S_{t+1}\right)\right)-V\left(S_{t}\right)\right)$

  

##### Q-Learning

  - $$
    \begin{aligned}
    TD_{error} &\leftarrow r_{t+1}+\gamma \max _{a_{t+1}} Q\left(s_{t+1}, a_{t+1}\right)-Q\left(s_{t}, a_{t}\right) \\
    Q\left(s_{t}, a_{t}\right) &\leftarrow Q\left(s_{t}, a_{t}\right)+\alpha TD_{error}
    \end{aligned}
    $$
    
  - 由于目标策略是贪心的，探索策略是$\epsilon$ -greedy的)，因此这是一个 offline 的策略
    
  - 伪码：

     ```python
     s_t = env.reset() # init state
     for t in range(max_episode_length):
         a_t = policy(s_t)
         s_t_plus_one, reward, _ = env.step(s_t, a_t)
         td_error = reward + gamma * max(Q_table[s_t_plus_one]) - Q_table[s_t][a_t]
         Q_table[s_t][a_t] = Q_table[s_t][a_t] + lr * td_error
         s_t = s_t_plus_one
     ```

     

##### Deep Q Network(DQN)：

- 利用神经网络来估计$Q(s_t, a_t)$，而不是用 table 来记录

   - $TD_{error} = \left(r_{t}+\gamma \max _{a_{t+1}} Q^{\operatorname{target}}\left(s_{t+1}, a_{t+1}\right)-Q\left(s_{t}, a_{t}\right)\right)$
   - 使用了experience replay和target network
      1. 将$\left(s_{t}, a_{t}, r_{t+1}, s_{t+1}\right)$存在一个 memory $D$中，从$D$中 sample mini-batch 
      2. $\mathscr{L}(\mathbf{w})=\mathbb{E}_{s_t, a_t, r_t, s_{t+1} \sim \mathscr{D}}\left[\left(r_t+\gamma \max _{a_{t+1}} Q\left(s_{t+1}, a_{t+1} ; \mathbf{w}^{-}\right)-Q(s_t, a_t ; \mathbf{w})\right)^{2}\right]$，利用这个来优化$\mathbf{w}$。每隔$\mathbf{w}^-$不传导梯度，一定时间令$\mathbf{w}^-=\mathbf{w}$。
   - 变种：$\max _{a_{t+1}} Q\left(s_{t+1}, a_{t+1} ; \mathbf{w}^{-}\right)$ 这一步会导致overestimation，因为$\mathbb{E}\left[\max \left(X_{1}, X_{2}\right)\right] \geq \max \left(\mathbb{E}\left[X_{1}\right], \mathbb{E}\left[X_{2}\right]\right)$ 
      为了减少其带来的影响，可以使用Double DQN，即 $\max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime} ; \mathbf{w}^{-}\right)=Q\left(s^{\prime}, \arg \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime} ; \mathbf{w}\right) ; \mathbf{w}^{-}\right)$ 
   - 还有很多其他 DQN 的变种

#### From policy methods to PAC bounds analysis

- Policy Iteration

  1. policy evaluation: $V(s) \leftarrow R(s, \pi(s))+\gamma \sum_{s^{\prime}} \operatorname{Pr}\left(s^{\prime} \mid s, \pi(s)\right) V\left(s^{\prime}\right), \forall s$ 直至收敛
  2. policy improvement: $\pi(s) \leftarrow \arg \max _{a \in \mathbb{A}} R(s, a)+\gamma \sum_{s^{\prime}} \operatorname{Pr}\left(s^{\prime} \mid s, a\right) V\left(s^{\prime}\right), \forall s$

##### Policy gradient

- 我们希望最大化目标函数：$J(\theta)=\sum_{s \in S} d^{\pi}(s) V^{\pi}(s)=\sum_{s \in S} d^{\pi}(s)\left(\sum_{a \in A} \pi_{\theta}(a \mid s) Q^{\pi}(s, a)\right)$ 
  其中 $d^{\pi}(s):=\lim _{t \rightarrow \infty} p\left(S_{t}=s \mid s_{0}, \pi_{\theta}\right)$ 是马尔可夫链的极限分布

- 根据Policy gradient theorem， $\nabla_{\theta} J(\theta)=\sum_{s \in S} d^{\pi}(s)\left(\sum_{a \in A} Q^{\pi}(s, a) \nabla_{\theta} \pi_{\theta}(a \mid s)\right)$ 
  $$
  \begin{aligned}
  \nabla_{\theta} V^{\pi}\left(s_{0}\right) & =\sum_{s \in S} d^{\pi}(s) \sum_{a \in A}\left(Q^{\pi}(s, a) \nabla_{\theta} \pi_{\theta}(a \mid s)\right) \\
  & \propto \sum_{s \in S} \mu(s) \sum_{a \in A}\left(\pi_{\theta}(a \mid s) Q^{\pi}(s, a) \frac{\nabla_{\theta} \pi_{\theta}(a \mid s)}{\pi_{\theta}(a \mid s)}\right) \\
  & =E_{s \sim \mu, a \sim \pi}\left[Q^{\pi}(s, a) \nabla_{\theta} \ln \pi_{\theta}(a \mid s)\right] \\
  & =\mathrm{E}_{s \sim \mu, a \sim \pi}\left[G_{t} \nabla_{\theta} \ln \pi_{\theta}(a \mid s)\right]
  \end{aligned}
  $$

- 伪码：
  
  1. Sample trajectories $\left\{\tau_{i}\right\}$ with horizon $H$ using $\pi_{\theta}(a \mid s)$
     $G_{i, t} \leftarrow \sum_{t=t^{\prime}}^{H} \gamma^{t-t^{\prime}} R\left(s_{i, t}, a_{i, t}\right)$
     $V_{t} \leftarrow \frac{1}{M} \sum_{i=1}^{M} G_{i, t}$ 其引入是为了减少方差
     $A\left(s_{i, t}, a_{i, t}\right) \leftarrow G_{i, t}-V_{t}$ 
     $\Delta \leftarrow \sum_{i, t} \nabla_{\theta} \log \pi_{\theta}\left(a_{i, t} \mid s_{i, t}\right) A\left(s_{i, t}, a_{i, t}\right)$
     $\theta \leftarrow \theta+\alpha \Delta$
- PAC：目前讲到的几个算法(Q-learning 和 policy gradient)都涉及sample，那么要 sample 多少才可以呢？


#### Non-Convex Optimisation: Survey and ADAM's Proof



#### Model-based Reinforcement Learning

- **Why&What  Model-based RL**: DRL has very low data efficiency, one solution is model based RL. we try to build a model to simulate environment and train the policy based on this model. since we don't need to interact with real world, the data efficiency could be improved.

- | Model-free RL                              | Model-based RL                            |
  | ------------------------------------------ | ----------------------------------------- |
  | best asymptotic performance                | On-policy learning(once model is learned) |
  | suitable for DL architecture with big data | higher sample efficiency                  |
  |                                            |                                           |
  | Suffer from instabilities of off-policy    | Suffer from model compounding error       |
  | Very low sample efficiency                 |                                           |

  **Dyna-Q**

  ![4_1](./pic/4_1.png)

  Initialize $Q(s, a)$ and $M$ odel $(s, a)$ for all $s \in \mathcal{S}$ and $a \in \mathcal{A}(s)$ 
  Do forever:
  	(a) $S \leftarrow$ current (nonterminal) state
  	(b) $A \leftarrow \varepsilon-\operatorname{greedy}(S, Q)$
  	(c) Execute action $A$; observe resultant reward, $R,$ and state, $S^{\prime}$
  	(d) $Q(S, A) \leftarrow Q(S, A)+\alpha\left[R+\gamma \max _{a} Q\left(S^{\prime}, a\right)-Q(S, A)\right]$
  	(e) $\operatorname{Model}(S, A) \leftarrow R, S^{\prime}$ (assuming deterministic environment)
  	(f) Repeat $n$ times:
  			$S \leftarrow$ random previously observed state
  			$A \leftarrow$ random action previously taken in $S$ 
  			$R, S^{\prime} \leftarrow M$ odel $(S, A)$
  			$Q(S, A) \leftarrow Q(S, A)+\alpha\left[R+\gamma \max _{a} Q\left(S^{\prime}, a\right)-Q(S, A)\right]$

- Key Questions of Model-based RL
  - Does the model really help improve the data efficiency?
  - Inevitably, the model is to-some-extent inaccurate. When to trust the model?
  - How to properly leverage the model to better train our policy?
  
- PETS

  - Basic idea: tries to sample actions that yield high reward
  - Initialize data $D$ with a random controller for one trial. 
    **for** Trial $k=1$ to $K$ **do** 
    	Train a $P E$ dynamics model $\tilde{f}$ given $\mathbb{D}$. 
    		**for** Time $t=0$ to TaskHorizon **do** 
    			**for** Actions sampled $a_{t: t+T} \sim \operatorname{CEM}(\cdot), 1$ to $\mathrm{NSamples}$ **do** 
    				Propagate state particles $s_{\tau}^{p}$ using $T S$ and $f \mid\left\{\mathbb{D}, \boldsymbol{a}_{t: t+T}\right\}$ 
    				Evaluate actions as $\sum_{\tau=t}^{t+T^{\prime}} \frac{1}{P} \sum_{p=1}^{P} r\left(\boldsymbol{s}_{\tau}^{p}, \boldsymbol{a}_{\tau}\right)$
    				Update CEM( $\cdot$ ) distribution. 
    			Execute first action $\boldsymbol{a}_{t}^{*}$ (only) from optimal actions $\boldsymbol{a}_{t: t+T}^{*}$ 
    			Record outcome: $\mathbb{D} \leftarrow \mathbb{D} \cup\left\{\boldsymbol{s}_{t}, \boldsymbol{a}_{t}^{*}, \boldsymbol{s}_{t+1}\right\} .$
  - what is $CEM$ : Cross Entropy Method
    $\operatorname{loss}(\boldsymbol{\theta})=-\sum_{n=1}^{N} \log \tilde{f}_{\boldsymbol{\theta}}\left(\boldsymbol{s}_{n+1} \mid \boldsymbol{s}_{n}, \boldsymbol{a}_{n}\right)$
    $\tilde{f}=\operatorname{Pr}\left(\boldsymbol{s}_{t+1} \mid \boldsymbol{s}_{t}, \boldsymbol{a}_{t}\right)=\mathcal{N}\left(\boldsymbol{\mu}_{\theta}\left(\boldsymbol{s}_{t}, \boldsymbol{a}_{t}\right), \boldsymbol{\Sigma}_{\theta}\left(\boldsymbol{s}_{t}, \boldsymbol{a}_{t}\right)\right)$
    Gaussian distribution可以抓住两种不确定性：一种是在未采样区域的不确定性；另一种是采样数据部分的不确定性
  - Improvement: POPLIN: maintain a policy to sample actions given the current simulated state -> see later in estimation learning

- Theoretic Bound

  - **SLBO**: 
    
    - Assumption : $V^{\pi, M^{\star}} \geq V^{\pi, \widehat{M}}-D_{\pi_{\mathrm{ref}}, \delta}(\widehat{M}, \pi), \quad \forall \pi$ s.t. $d\left(\pi, \pi_{\mathrm{ref}}\right) \leq \delta$
      $\widehat{M}=M^{\star} \Longrightarrow D_{\pi_{\mathrm{ref}}}(\widehat{M}, \pi)=0, \quad \forall \pi, \pi_{\mathrm{ref}} $
      $D_{\pi_{\mathrm{ref}}}(\widehat{M}, \pi)$ is of the form $\underset{\tau \sim \pi_{\mathrm{ref}}, M^{\star}}{\mathbb{E}}[f(\widehat{M}, \pi, \tau)]$ e.g. $D_{\pi_{\text {ref }}}(\widehat{M}, \pi)=L \cdot_{S_{0}, \ldots, S_{t}, \sim \pi_{\text {ref }}, M^{\star}}\left[\left\|\widehat{M}\left(S_{t}\right)-S_{t+1}\right\|\right]$
      
    - Theorem based on assumption: 
<<<<<<< HEAD:算法/RL Learning/Reinforcement Learning.md
      for Algorithm : $\begin{aligned} \pi_{k+1}, M_{k+1}=& \underset{\pi \in \Pi, M \in \mathcal{M}}{\operatorname{argmax}} V^{\pi, M}-D_{\pi_{k}, \delta}(M, \pi) \\ & \text { s.t. } d\left(\pi, \pi_{k}\right) \leq \delta \end{aligned}$
  
- 
  - 
=======
      for Algorithm(TRPO) : $\begin{aligned} \pi_{k+1}, M_{k+1}=& \underset{\pi \in \Pi, M \in \mathcal{M}}{\operatorname{argmax}} V^{\pi, M}-D_{\pi_{k}, \delta}(M, \pi) \\ & \text { s.t. } d\left(\pi, \pi_{k}\right) \leq \delta \end{aligned}$
  
    $V^{\pi_{0}, M^{\star}} \leq V^{\pi_{1}, M^{\star}} \leq \cdots \leq V^{\pi_{T}, M^{\star}}$
      
    - 算法：
      Initialize model network parameters $\phi$ and policy network parameters $\theta$ 
      Initialize dataset $\mathcal{D} \leftarrow \emptyset$ 
          for $n_{\text {outer }}$ iterations do
              $\mathcal{D} \leftarrow \mathcal{D} \cup\left\{\right.$ collect $n_{\text {collect }}$ samples from real environment using $\pi_{\theta}$ with noises $\}$ 
              for $n_{\text {inner }}$ iterations do
                  for $n_{\text {model }}$ iterations do 
                      optimize (6.1) over $\phi$ with sampled data from $\mathcal{D}$ by one step of Adam
                  for $n_{\text {policy }}$ iterations do 
                      $\mathcal{D}^{\prime} \leftarrow\left\{\right.$ collect $n_{\text {trpo }}$ samples using $\widehat{M}_{\phi}$ as dynamics $\}$ 
                      optimize $\pi_{\theta}$ by running TRPO on $\mathcal{D}^{\prime}$
    
      Model optimization loss： $\mathcal{L}_{\phi}^{(H)}\left(\left(s_{t: t+h}, a_{t: t+h}\right) ; \phi\right)=\frac{1}{H} \sum_{i=1}^{H}\left\|\left(\hat{s}_{t+i}-\hat{s}_{t+i-1}\right)-\left(s_{t+i}-s_{t+i-1}\right)\right\|_{2}$
      Policy optimization target： $\max _{\phi, \theta} \quad V^{\pi_{\theta}, \operatorname{sg}\left(\widehat{M}_{\phi}\right)}-\lambda_{\left(s_{t: t+h}, a_{t: t+h}\right) \sim \pi_{k}, M^{\star}} \mathbb{E}\left[\mathcal{L}_{\phi}^{(H)}\left(\left(s_{t: t+h}, a_{t: t+h}\right) ; \phi\right)\right]$
    
    - 补充1：$\pi_{ref}$是使用采样数据的policy，$\hat{}$代表虚拟环境
    
    - 补充2：这里有一个很强的假设，$\hat{M}$可以取到$M$，当虚拟模型不太准确的时候，其带来的误差会很大
    
  - MBPO：最大的贡献就是研究了该向前模拟几步，核心和Dyna很像
    
    - $\eta[\pi] \geq \eta^{\mathrm{branch}}[\pi]-2 r_{\max }\left[\frac{\gamma^{k+1} \epsilon_{\pi}}{(1-\gamma)^{2}}+\frac{\gamma^{k}+2}{(1-\gamma)} \epsilon_{\pi}+\frac{k}{1-\gamma}\left(\epsilon_{m}+2 \epsilon_{\pi}\right)\right]$
      每次迭代之后policy变化量：$\epsilon_{\pi}=\max _{s} D_{T V}\left(\pi \| \pi_{D}\right)$
      每次迭代之后model变化量：$\epsilon_{m}=\max _{t} \mathbb{E}_{s \sim \pi_{D, t}}\left[D_{T V}\left(p\left(s^{\prime}, r \mid s, a\right) \| p_{\theta}\left(s^{\prime}, r \mid s, a\right)\right)\right]$
      通过这里定理，求导，我们很卑微的发现k=0的时候lower bound取到最大值，也就是说
>>>>>>> cff7417afa57c03dbb6f572eb17a99c06b5ec7fc:Reinforcement Learning.md
    
    - 修改使用新的model变化量：$\epsilon_{m^{\prime}}=\max _{t} \mathbb{E}_{s \sim \pi_{t}}\left[D_{T V}\left(p\left(s^{\prime}, r \mid s, a\right) \| p_{\theta}\left(s^{\prime}, r \mid s, a\right)\right)\right]$，则
      $\eta[\pi] \geq \eta^{\mathrm{branch}}[\pi]-2 r_{\max }\left[\frac{\gamma^{k+1} \epsilon_{\pi}}{(1-\gamma)^{2}}+\frac{\gamma^{k} \epsilon_{\pi}}{(1-\gamma)}+\frac{k}{1-\gamma}\left(\epsilon_{m^{\prime}}\right)\right]$ ，现在求导，当$\frac{\mathrm{d} \epsilon_{m^{\prime}}}{\mathrm{d} \epsilon_{\pi}}$ 足够小的时候(现实中他们的确足够小)，k是大于0的。这样说明，在一定范围内(前k步)，我们是可以相信这个model的。
    
    - 算法：
    
      Initialize policy $\pi_{\phi},$ predictive model $p_{\theta},$ environment dataset $\mathcal{D}_{\text {env }},$ model dataset $\mathcal{D}_{\text {model }}$ 
      for $N$ epochs do 
      	Train model $p_{\theta}$ on $\mathcal{D}_{\text {env }}$ via maximum likelihood 
      	for $E$ steps do 
      		Take action in environment according to $\pi_{\phi}$; add to $\mathcal{D}_{\text {env }}$ 
      		for $M$ model rollouts do 
      			Sample $s_{t}$ uniformly from $\mathcal{D}_{\text {env }}$ 
      			Perform $k$ -step model rollout starting from $s_{t}$ using policy $\pi_{\phi} ;$ add to $\mathcal{D}_{\text {model }}$ 
      		for $G$ gradient updates do 
      			Update policy parameters on model data: $\phi \leftarrow \phi-\lambda_{\pi} \hat{\nabla}_{\phi} J_{\pi}\left(\phi, \mathcal{D}_{\text {model }}\right)$
    
    - 补充：为啥使用TV distance而不是KL divergence，因为前者满足三角不等式
    
    - MBPO和SLBO的差别：前者使用TRPO训练，后者使用SSA训练；前者交替的训练model和policy，后者把model训练好之后再训练policy
  
-  Backpropagation through paths

  - 之后的a



#### Control as Inference

- probabilistic graphical models
- 











