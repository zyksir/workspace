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

- **Monte-Carlo Methods**: 
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

- TD Learning: learn from **incomplete** episodes -》 是biased的，不要求episodes终止

  - 根据S选择A，观察得到R和$S^{\prime}$，再根据$S^{\prime}$得到$A^{\prime}$ $Q(S, A) \leftarrow Q(S, A)+\alpha\left(R+\gamma Q\left(S^{\prime}, A^{\prime}\right)-Q(S, A)\right)$

- off-policy learning

  - 我们观察到的策略$\mu(a \mid s)$只是用于探索、并不是我们最终使用的目标策略$\pi(a \mid s)$ 

  - 数学依据 importance sampling：$\begin{aligned} \mathbb{E}_{X \sim P}[f(X)] &=\sum P(X) f(X) \\ &=\sum Q(X) \frac{P(X)}{Q(X)} f(X) \\ &=\mathbb{E}_{X \sim Q}\left[\frac{P(X)}{Q(X)} f(X)\right] \end{aligned}$

  - offline MC: $G_{t}^{\pi / \mu}=\frac{\pi\left(A_{t} \mid S_{t}\right)}{\mu\left(A_{t} \mid S_{t}\right)} \frac{\pi\left(A_{t+1} S_{t+1}\right)}{\mu\left(A_{t+1} \mid S_{t+1}\right)} \ldots \frac{\pi\left(A_{T} \mid S_{T}\right)}{\mu\left(A_{T} \mid S_{T}\right)} G_{t}$

    由于importance sampling会大大增加方差，因此不用这个

  - offline TD: $V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left(\frac{\pi\left(A_{t} \mid S_{t}\right)}{\mu\left(A_{t} \mid S_{t}\right)}\left(R_{t+1}+\gamma V\left(S_{t+1}\right)\right)-V\left(S_{t}\right)\right)$

  - Q-Learning: $Q\left(S_{t}, A_{t}\right) \leftarrow Q\left(S_{t}, A_{t}\right)+\alpha\left(R_{t+1}+\gamma Q\left(S_{t+1}, A^{\prime}\right)-Q\left(S_{t}, A_{t}\right)\right)$， $A_{t} \sim \mu\left(\cdot \mid S_{t}\right)$， $A^{\prime} \sim \pi\left(\cdot \mid S_{t+1}\right)$
    目标策略是贪心的，探索策略是$\epsilon$ -greedy的，则 $ R_{t+1}+\gamma Q\left(S_{t+1}, A^{\prime}\right) =R_{t+1}+\max _{a^{\prime}} \gamma Q\left(S_{t+1}, a^{\prime}\right)$
    Q 函数可以这么计算：$Q(S, A) \leftarrow Q(S, A)+\alpha\left[R+\gamma \max _{a} Q\left(S^{\prime}, a\right)-Q(S, A)\right]$

- DQN：实际应用的时候，































