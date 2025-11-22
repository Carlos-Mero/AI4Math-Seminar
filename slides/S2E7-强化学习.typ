#import "@preview/may:0.1.1": *
#show: may-pre.with(
  config-info(
    title: [强化学习],
    subtitle: [AI4Math讨论班第二季-07],
    author: [晚星],
    institution: [求真书院],
    date: datetime.today()
  )
)

#set quote(block: true)
#let hl(body) = highlight(fill: skyl)[
    #set text(weight: "bold")
    #body
  ]
#let tcslide(body1, body2) = grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  align(horizon, body1),
  align(center, body2)
)

#image-slide(img: image("./media/s2e7.png"))

#title-slide()

= 新闻环节

== Gemini 3 pro

#tcslide([
  Google最新的旗舰大语言模型#link("https://deepmind.google/models/gemini/", "Gemini 3 Pro")于2025年11月19日正式发布，在多项性能指标上力压同期最强的GPT-5.1和Grok-4.1。官方更是直接称其为「最先进的推理模型」。

  这是Google DeepMind自两年前Bard模型的惨败后再次取回大模型SOTA的位置。
], image("./media/gemini_release.png"))

== Gemini 3 pro

#tcslide([
  Gemini 3 pro最核心的能力包括

  - 显著更强的数学和代码等推理能力
  - 支持文本、图像、视频、音频全模态
  - 更强的提示与指令遵循能力
  - 长任务规划与执行能力
  - 工具使用与智能体

  虽然邪恶的闭源传统使得我们对它的技术方案一无所知，但是从这里我们也可一窥工业界大模型的发展方向。
  // 求真书院和DeepSeek的合作事项

  详情可参考#link("https://mp.weixin.qq.com/s/m5DnddinQuH_SxXEyUbYaA", "相关报道")。
], image("./media/gemini3benchmarks.png"))

== Nana banana pro

#tcslide([
  就在Gemini 3 pro发布之后仅一天，Google的代表性图像生成模型也迎来了一波大幅更新：#link("https://deepmind.google/models/gemini-image/pro/", "Nano Banana Pro")正式发布。其关键性能进步在于：

  - 支持至多4K分辨率图像的生成和多种长宽比
  - 更细致的编辑控制选项
    - 更强的动嘴修图能力
  - 完善的世界知识
  - 显著更好的文本控制能力

  这也标志着Google在图像生成领域也同时取得了一定的领先位置。
  // 之前Bard失败的部分原因是当时Google的绘画模型强烈地拒绝画白人
], box([#image("./media/nbpro_1.png") #image("./media/nbpro2.png")]))

#focus-slide([
  #set page(margin: 1.4em)
  而在这一切成果的背后，其基础技术仍然是Transformer神经网络与基于随机梯度下降的训练流程。
])

= 经典的强化学习

== 回顾：监督训练

#tcslide([
  在此前我们已经介绍过大语言模型的基本构成：

  - 模型结构：Transformer
  - 工作方式：自回归生成
  - 优化方法：AdamW等

  而有了这一切之后，剩下的就只需要根据我们的最终目标构造特定的损失函数作为优化目标，并朝着这个方向训练模型即可。

  这里最经典的方法也就是监督学习 (Supervised Learning, SL)，它就是参考着神经网络或者机器学习最初的想法，直接通过现有数据做函数拟合。
], [
  #set align(left)
  例如在基础的图像生成当中使用的方法基本是首先收集大量的图片 (可能带有类别或文本标注)，然后计算神经网络预测重建的图像和原数据的二范数的平方。

  $ cal(L)(theta) = EE ||x_0 - x_(0, theta) (x_t, t)||^2 $

  而大语言模型的预训练和监督微调阶段主要所做的事情也就是模仿训练数据的模式，它所使用的损失函数也就是交叉熵损失：

  $ cal(L)(theta) = EE_(x tilde cal(D)) sum log p_theta (x_t | x_(< t)) $

  它与强化学习的主要差异仅在损失函数的构造上。
  // 这种方法属于简单有效但饱受批判。几乎AI领域的dalao都不认可LLM这条路线
])

== 强化学习的基本概念

#tcslide([
  人们普遍相信最具扩展潜力的学习方法必然建立在自主性的基础上。即一个智能体在观测和行动等与环境的交互过程当中不断地自主学习与进化。其中常常涉及到的几个概念是：

  / State: ($cal(S)$) 这是系统所有可能状态的集合。
  / Action: ($cal(A)$) 这是智能体在各个状态下所能够采取的动作的集合。
  / Transition Function: ($T: cal(S) times cal(A) times cal(S) -> [0, 1], (s, a, s') |-> P(s' | s, a)$) 这一函数表征在某一个给定状态以及动作下各种可能的状态改变发生的概率。
], box([
  #set align(left)
  / Reward: ($r: cal(S) times cal(A) -> RR$) 表示在特定状态下采取动作时得到的反馈。

  此时这一智能体通常就被表达为一个策略 (policy) 函数 $pi(a | s)$，预测在每个情形下选取动作的概率。

  #image("./media/xc_act.png")
]))

== 经典强化学习方法

#tcslide([
  / 回报函数: $R(tau) = sum gamma^t r_t$，通常针对特定任务，根据reward进行定义，代表某一个动作轨迹所带来的最终回报。
  / 价值函数: $V^pi (s) = EE_(a_t tilde pi(s_t)) R(tau)$，指的是在特定状态以及策略下回报函数的期望。
  / 动作价值函数: $Q^pi (s_0, a_0)$，指的是在特定状态下执行某一个动作之后，特定策略下回报函数的期望。

  根据这些我们就可以得到强化学习理论当中经典的贝尔曼方程 (Bellman Equation)，它说明了一个状态的价值等于当前获得的奖励加上后续状态价值的加权平均。
],[
  #set align(left)
  $
  V^pi (s) &= sum_(a in cal(A)) pi(a | s) (r(s, a) +\ &gamma sum_(s') P(s' | s, a) V^pi (s'))
  $

  接着根据最优情形下的贝尔曼方程进行迭代，学习寻找动作价值函数的方法就被称为Q-Learning。这是一个与现代深度学习完全不同的古典算法。其具体算法表示为

  $ Q(s, a) &<- Q(s, a) +\ &alpha (r(s, a) + gamma max_(a') (Q(s', a') - Q(s, a))) $
  // 古典时代大家普遍都在考虑多步的任务
])

== 策略梯度

#tcslide([
  经典的策略梯度REINFORCE算法提出于1992年，在它之前强化学习的主流是价值函数方法，但它们很难直接表示一个复杂的随机策略。REINFORCE的训练目标就是直接对参数化策略 $pi_theta (a | s)$ 做梯度上升。其核心公式就在于：

  $
  J(theta) = EE_(tau tilde pi_theta) [R(tau)]
  $

  它同时可以被简化为

  $
  gradient_theta J(theta) = EE_tau [sum_t gradient_theta log pi_theta (a_t | s_t) G_t]
  $
],[
  #set align(left)
  此处的 $G_t$ 代表的是从 $t$ 时刻往后累计的回报。

  直觉上REINFORCE所做的就是将高回报的轨迹概率提高，同时将低回报的轨迹概率降低。如此就可以训练出整体更优的策略了。

  不过在当时的应用场景当中REINFORCE的问题就在于每一条动作轨迹回报波动很大，这导致其对于梯度的估计噪声很大且训练并不稳定。
  // 例如走迷宫的例子

  自此之后基于policy做优化的强化学习算法逐步成为主流。
])

== Actor-Critic与策略梯度

#tcslide([
  #link("https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf", "Actor-Critic")的思想在强化学习理论早期就已出现，而后来的神经网络则成为了标准的拟合工具用于扮演这两个角色。其中：

  - Critic学习每一个状态的价值信息
  - Actor根据Critic的反馈调整策略

  这里额外Critic的意义就在于拟合 $V^pi (s_t)$ 评估每一个状态的平均价值，并以此平衡各步动作，偏好相对更优的动作而非绝对更优。如此即可实现更好的训练稳定性。
], [
  #set align(left)
  在引入Critic之后我们就可以将REINFORCE当中的 $G_t$ 替换为 $G_t - V_theta (s_t)$，作为优势函数的估计。而精确的优势函数定义则是

  $
  A^pi (s_t, a_t) = Q^pi (s_t, a_t) - V^pi (s_t)
  $

  Advantage function的概念由此引入，这一思路也被后来的各种强化学习工作所继承。
])

== TRPO算法

#tcslide([
  TRPO (Trust Region Policy Optimization) 算法最早在2015年提出，它是深度强化学习当中的一个重量级工作。TRPO所使用的目标函数核心项是

  $
  EE [(pi_theta (a | s))/(pi_(theta_"old") (a | s)) hat(A) (s, a)]
  $

  同时额外引入了可信域约束，进一步增强了稳定性，具体而言它就是：

  $
  EE["KL" (pi_(theta_"old")) (dot | s) || pi_theta (dot | s)] <= delta
  $
], [
  #set align(left)
  在这里TRPO训练使用的轨迹都由未经训练的旧策略 $pi_(theta_"old") (a | s)$ 生成，这也可以更好地保持训练的稳定性。也正因此TRPO引入了重要性采样来修正其中的偏差，也就是使用

  $ r_t (theta) = (pi_theta (a_t | s_t))/(pi_(theta_"old") (a_t | s_t)) $

  代替直接通过新策略模型计算的梯度，由此正确地估计新策略下的优势函数值。

  实践上TRPO通常需要使用共轭梯度等方法来进行求解，实现上通常也非常复杂。
])

== PPO算法

#tcslide([
  PPO (Proximal Policy Optimization) 于2017年发布，它是对TRPO算法的一次大幅简化，并取得了良好的性能表现。在此后的数年里PPO一直是强化学习算法的标杆。

  它最主要的优化就是将TRPO当中复杂的可信域计算简化为一个简单的策略商截断，其目标主项就是：

  $
  EE [min (r_i (theta) hat(A)_t, "clip"(r_t (theta), 1-epsilon, 1+epsilon) hat(A)_t)]
  $

  也就是将偏离旧策略太多的 $r_t$ 直接截断到一个固定值，并参与后续的各种计算。
], [
  #set align(left)
  PPO属于一个在理论上并不优雅，但是工程实践当中却非常方便，能够有效防止策略变化过大导致的问题。

  后来在实践当中也常常向它的损失函数当中额外添加熵或者KL散度两项，以鼓励更多样的探索轨迹或者保持更好的稳定性。

  $
  H(pi_theta (dot | s)) = - sum_a pi_theta (a | s) log pi_theta (a | s)\
  $
  $
  &D_"KL"(pi_theta (dot | s) || pi_(theta_"old") (dot | s))\ &= sum_a pi_(theta_"old") (a | s) log (pi_(theta_"old") (a | s))/(pi_theta (a | s))
  $
])

= LLM中的RL

== 将语言模型视作策略

#tcslide([
  在前大语言模型时代，NLP领域就存在许多使用RL算法去优化不可导指标的工作。而现今大语言模型的工作方式是自回归生成，或者称为next-token prediction，它可以在大或小两种不同尺度上视作一个策略模型：

  小尺度上：

  - 状态：补全时的上文
  - 动作：生成下一个token
  - 策略：就是语言模型 $pi_theta (a_t | a_(< t))$

  与此同时在大尺度上，例如基于LLM构建的智能体任务当中，有
], [
  #set align(left)
  - 状态：上文内容和环境观测结果
  - 动作：生成一段代码并交给程序执行
  - 策略：从当前状态出发生成出下一段动作

  那么由此我们就可以在不同的尺度上对大语言模型应用强化学习算法，并训练它实现我们所期望的各种功能。

  接下来的问题就是如何构建强化学习算法所需要的环境reward信息。
])

== RLHF微调

#tcslide([
  RLHF (Reinforcement Learning from Human Feedback) 算法首次提出于OpenAI 2022年的一篇论文#link("https://cdn.openai.com/papers/Training_language_models_to_follow_instructions_with_human_feedback.pdf?utm_source=chatgpt.com", "InstructGPT")，其中首次通过RLHF这一算法赋予了大语言模型强大的指令遵循能力以及执行语言领域通用任务的能力。它也是大模型发展早期最受关注的算法之一。

  RLHF的实现一般分三步

  1. 从预训练模型出发收集大量人工标注的指令与回答数据，直接进行监督训练微调，由此得到一个还不错会听话的初始policy。
], [
  #set align(left)
  2. 给定同一个prompt并从初始policy当中采样多条模型回答，并让人类标注者提供偏好反馈。后再训练一个reward model来为每一个回答预测一个标量的reward。(此时这样的reward model通常就是直接初始化一个大语言模型并接上一个标量输出头)
  3. 使用PPO算法做强化学习，同时需要额外加上许多约束防止崩溃。

  实际上现代大语言模型已经基本不再直接使用RLHF算法，在获取到足够多的标注数据之后只需简单的SFT就已有足够强的指令遵循效果了。
])

== DPO直接优化

#tcslide([
  DPO (Direct Preference Optimization) 算法首次提出于2023年，它是针对RLHF流程的一次大幅简化。

  #image("./media/rlhf_dpo.png")

  它直接从偏好数据对出发，不经过中间训练reward model和完整的强化学习流程，而是直接从偏好数据对 $(x, y^+, y^-)$ 出发构建损失函数。

  DPO具体的目标函数计算公式是：
], [
  #set align(left)
  $
  EE_(x, y^+, y^-) [(log pi_theta (y^+ | x) - log pi_"ref" (y^+ | x))\ - (log pi_theta (y^- | x) - log pi_"ref" (y^- | x))]
  $

  可以看到它在形式上已经和直接的监督训练差别不大了，仅仅是根据对比数据构造了一些相对损失。由此DPO算法在偏好对齐训练上能够做到简便、稳定，且效果上几乎不输甚至可能略优于完整的PPO-based RLHF。

  当然其实现代很多时候单纯地使用交叉熵损失做SFT就已经够用了。
])

== GRPO再简化

#tcslide([
  GRPO (Group Relative Policy Optimization) 算法首次提出于2024年的论文#link("https://arxiv.org/abs/2402.03300", "DeepSeek-Math")。它在PPO的基础上进一步进行简化，删去了value model，并直接使用多组回复的平均reward作为基准计算优势函数。如此一来优势函数就直接是

  $ hat(A)_(i,t) = (r_i - "mean"(vb(r)))/("std" (vb(r))) $

  这对于数学、代码等简单的问题设定而言非常直接且有效。
], [
  #set align(left)
  #image("./media/ppo_and_grpo.png")
  #image("./media/grpo_formula.png")
])

== 深度思考的时代

#tcslide([
  2025年初左右兴起的推理模型技术就是强化学习技术在大语言模型当中的又一轮高光。而在后续的改良工作当中人们发现进一步删去reward model使用可验证损失，删去reference model以及KL散度项，都不会影响训练的稳定性。由此产生出的算法被统称为RLVR (Reinforcement Learning from Verifiable Reward)。

  大语言模型在针对推理问题的大量强化学习训练下自发地产生了深度思考的能力，GRPO也在这一年里成为了强化学习算法的新基准。
],
  box([
    #image("./media/dsr1_deepthink.png")
    #image("./media/dsr1_ahamoment.jpeg")
  ])
)

== Dr. GRPO - Done right

#tcslide([
  Dr. GRPO是相对标准GRPO算法的一次有效改进，它基于对GRPO算法的一些关键观察并修正了其中可能存在的一些偏差，并在实验当中取得了更好的效果。

  原本GRPO损失函数当中存在的一个关键问题在于长度偏差，因为reward按照response-level进行分配，除以token数量平均之后就会在token level产生长度偏置。具体体现在：

  - 短而正确的回复每个token的reward比长而正确的更高
  - 长而错误的回复每个token的reward比短而错误的更高
], [
  #set align(left)
  #image("./media/dr_grpo.png")
  这会意外地引导模型偏好长而错误的回复，产生很多无用的错误成果。

  Dr. GRPO的另一项改进在于删去了优势函数当中的归一化，因为他们认为这种归一化会加强极端情况而削弱回复正确率方差较大时的情形，而后者才是最有学习价值的。
  // 这一观点我表示存疑
])

== GSPO - 序列层级优化

#tcslide([
  GSPO (Group Sequence Policy Optimization) 则由阿里巴巴通义实验室于2025.7.28首次提出，它指出了原本GRPO当中的另一个关键问题：它错误地使用了PPO式的重要性采样。

  由于TRPO当年重要性采样所期望实现的目标是根据随机变量的分布差异修正reward数值，但是这里reward是在序列级生效的，而非token层级。因此GSPO当中将策略商的数值修改为了

  $
  s_i (theta) = exp(1/(|y_i|) sum_t log (pi_theta (y_(i, t) | x, y_(i, < t)))/(pi_(theta_"old") (y_(i, t) | x, y_(i, < t))))
  $
], [
  #set align(left)
  #image("./media/gspo_effects.png")
  GSPO当中的clipping机制也在序列级实现。实验当中GSPO确实呈现出了更强的性能表现，尤其在MoE模型当中优势非常明显。
])

== 模仿与探索的学习方式

#tcslide([
  // 例如美术工作者对于图像生成模型的评价
  实际上即使深度思考能力也未必需要由深度学习得到，只需要获取相应的长思维链数据进行微调，就可以直接指导语言模型学会深度思考的工作方式，并取得显著的性能提升 (参见DeepSeek-R1与Qwen3等等论文)。

  如果对比参看标准监督训练和REINFORCE，DPO等的损失函数的话，除去reward放缩外，基本上可以认为SFT就是只有正例的RL算法。或者另一方面可以认为RL也是某种模仿训练的数据来自自身探索而非人为标注的微调手段。
], box([
    #image("./media/training_pipeline_qwen_3.png")
    #image("./media/kimi_k2_agent_synthesis.png")
]))

#focus-slide([
  #set page(margin: 1.4em)
  下回预告

  本集当中我们已经完成了现代语言模型各方面技术的讲解。在下集当中我们将会讲解VAE, GAN, Diffusion, Flow Matching等视觉生成模型的相关情况，作为AI系列的补完。
])
