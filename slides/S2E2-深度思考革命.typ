#import "@preview/may:0.1.1": *
#show: may-pre.with(
  config-info(
    title: [深度思考革命],
    subtitle: [AI4Math讨论班第二季-02],
    author: [黄砚星],
    institution: [求真书院],
    date: datetime.today()
  )
)

#set quote(block: true)
#let hl(body) = highlight(fill: skyl)[
    #set text(weight: "bold")
    #body
  ]

#image-slide(img: image("./media/s2e2.png"))

#title-slide()

#image-slide(img: image("./media/nlp_united.png"), body: [上回说到NLP领域大一统之后，人们开始寻找新的扩张方向。])

#image-slide(img: image("./media/llm_timeline.png"), body: none)

= 在ChatGPT之后

== GPT系列自身的演进

OpenAI在ChatGPT获得成功之后正式黑化为了ClosedAI，不过从他们后续的工业级大模型上也能看到LLM整体的发展脉络。

/ 2022-11-30: ChatGPT首次发布，具有4096 tokens的上下文长度及强大的指令遵循能力，可以执行各种NLP任务
/ 2023-03-14: GPT-4，各方面性能均有显著进步，上下文扩展到8192 tokens
/ 2023-11-06: GPT-4 Turbo，上下文扩展到128k tokens，并首次支持了图像多模态输入
/ 2024-05-13: GPT-4o (Omni)，价格更低的同时各方面性能显著进步，原生支持文本、图像和语音模态
/ 2024-09-12: o1系列，第一个推理模型，在数学、编码等任务上实现了性能飞跃式进步
/ 2025-04-14: GPT-4.1系列，强调编码能力、工具调用与百万级上下文的非推理模型
/ 2025-04-16: o3 / o4-mini，性能再次显著进步，并着重提升了多模态推理能力
/ 2025-08-07: GPT-5系列发布，首次整合各形态模型与成熟的工具使用能力

== Scaling Law - 并非一切的答案

#grid(
  columns: (1fr, 1fr),
  [
  Scaling Law最早于2020年的一篇论文_Scaling Laws for Neural Language Models_当中提出，其中指出语言模型的验证损失下限与参数量、数据量之间近似呈现出对数关系，由此可以预测更大模型的性能会更好。

  2022年提出的Chinchilla规则则进一步明确了经验上数据与参数量的最优配比 (尽管现在已经不太适用)

  而实际上在高中数学数据集MATH上

  - GPT-3 (175B, few shot): 5.6% (2021.5)
  - Qwen2.5-Math-7B (cot): 83.6% (2024.9)
  - Qwen3-1.7B (long-cot): 93.4% (2025.5)
],
  align(center, image("./media/scaling_law.jpeg"))
)

== 长上下文技术

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  [
  Transformer架构本身会并行地处理各个token而忽略位置信息，因此通常的处理都是在embedding阶段将位置信息加入语义向量当中。

  ChatGPT及之前的模型大多采用绝对位置编码，因此训练后模型对于各个token的处理显著依赖于绝对位置，难以扩展外推。

  RoPE位置编码通过向量旋转的方式编码位置信息，因此两向量内积的结果只与相对位置有关，可以非常容易地推广到无比长的上下文范围。

  长上下文也依赖于后来的各种工程优化。
],
  [
  $
  vb(q)_m^TT vb(k)_n &= (vb(R)_m vb(W)_q vb(x)_m)^TT (vb(R)_n vb(W)_k vb(x)_n)\
  &= vb(x)^TT vb(W)_q vb(R)_(n-m) vb(W)_k vb(x)_n
  $

  #figure(
    caption: [旋转位置编码RoPE是实现长上下文的最关键技术之一],
    image("./media/rope.png")
  )
]
)

== 模型架构优化

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  [
  针对LLM或者Transformer模型的结构优化也是一大重要方向。目前大多数这方面的工作都围绕着计算性能优化展开，这是因为：

  - LLM的性能表现理论匮乏，难以通过结构改良针对优化
  - 神经网络的计算效率与复杂度则相当明确

  这方面的代表性成果可能包括

  - FlashAttention
  - MoE架构
  - Native Sparse Attention
  - 等等等等
],
  figure(
    caption: [DeepSeek的代表作之一，DeepSeek-V3的架构细节],
    image("./media/dsmoe.png"))
)

== 多模态融合 - 统一到语义空间

#align(center, image("./media/qwen3vl_arc.jpg"))

== 数据 - 可能是性能进步的第一动因

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  [
  对于LLM而言，训练数据的质量与数量对于其性能表现有着至关重要的影响。

  - 大模型预训练的数据主要来自于互联网
  - 互联网数据显然是低质量内容为主
  - 后训练数据往往依赖人工标注，数量与多样性都很有限

  数据筛选和合成都是各厂的看家本领。

  #image("./media/dataman_des.png", height: 35%)
],
  figure(
    caption: [可见o3用的数据质量仍然堪忧],
    image("./media/data_pollution_in_gpt_4o.png"))
)

== 而在这一大批专业的工作当中……

#image("./media/cot_prompting.png")

== 思维链 - 从语言走向推理

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  [
  思维链的概念最初来自于Google 2022年1月的一篇论文_Chain-of-Thought Prompting Elicits Reasoning in Large Language Models_，目前它的引用量已经突破了两万。

  研究者发现只需要如上修改prompt方案，即可实现数倍的性能提升！

  与此同时如果更进一步，实际上只需要增加一句prompt: "Let's think step by step"，就可以激发zero-shot CoT的能力，实现性能的显著提升。再之后更可以直接在训练数据当中加入更多推理文本，使其取得不断变强的zero-shot能力。
],
  image("./media/cot_effect.png")
)

#image-slide(img: image("./media/qwen2.5-math.png"))

#image-slide(img: image("./media/xc_deepthink.png"), body: [只是……感觉还差了点什么])

= OpenAI草莓的秘密

== 为什么是草莓？

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  align(horizon, [
    Strawberry🍓来自于此前广为流传的一个大模型Case，这在当时被普遍认为是LLM推理能力不足的体现 (实际并不是)

    后来OpenAI在2025.7-2025.9之间也多次使用🍓进行营销
  ]),
  align(center, image("./media/strawberry_reasoning.jpeg"))
)

== 第一个推理模型 - o1

OpenAI于2024.9.12发布了此前内部代号为🍓的推理模型 (Large Reasoning Model, LRM) o1的两个预览版：o1-mini和o1-preview。它们实现了在推理类任务上性能的大幅跃进

#align(center, image("./media/o1_performance.png", height: 80%))

== 震撼的超长思维链

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  [
  OpenAI同时公布了o1模型在解决各类推理问题时所使用的超长思维链。

  其内部思维极度复杂与细致，像是一位真正领域专家的内心活动，令人大受震撼。

  但关于模型实现的具体细节，ClosedAI基本完全保密，只提到了一个关键词：强化学习。

  #image("./media/o1-benchmarks.png")
],
  align(center, image("./media/longcot-o1.png"))
)

#image-slide(img: image("./media/useRL.jpg"))

= 强化学习是什么

== 强化学习的基本概念

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  align(horizon, [
  强化学习 (Reinforcement Learning, RL) 是一种从经验当中学习的方法。

  这也是Sutton认为最有扩展潜力的学习方式。

  它的灵感最早可能来自于对生物学习或者生理学的抽象，一个智能体通常通过与环境交互，观察反馈并寻找最大化奖励信号的方法。

  RL此前已在游戏、robotics等领域有了广泛的应用。
]),
  image("./media/rl_overview.jpeg")
)

== AlphaGo - 强化学习的高光之一

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  [
  游戏活动通常动作空间较小，同时具有相当明确的reward，就很适合强化学习发挥作用 (相比较更现实的情况)

  2016年AlphaGo依靠RL训练击败围棋世界冠军，此后AI在围棋活动当中逐步取得了远超人类的能力。

  AlphaGO对于策略模型的训练方式大致是，根据棋局胜负$z in {1, -1}$分别定义正或负的损失函数$z log p_theta (a | s)$，以此调高获胜动作的概率并降低失败动作的概率。

  最后再辅以价值网络评估以及MCTS搜索算法，即可实现超人类的棋力。
],
  align(center, image("./media/alphago.jpeg"))
)

== AlphaZero - 已经没有人类了

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  align(horizon, [
  AlphaGo在第一阶段的训练当中仍然依赖于大量拟合人类棋谱，对于数据需求很大。

  在一两年之后发布的AlphaZero当中，Google将AlphaGo的模型结构和训练流程大幅简化，将策略网络和价值网络合并为同一个resnet，并去除了拟合人类棋谱的过程，从零开始自我对弈训练。

  AlphaZero由此实现了泛化能力和性能的大幅增强，可以以60:40的胜率战胜此前的AlphaGo Zero，而AlphaGo Zero对阵AlphaGo的胜率则是100:0。
]),
  align(center, [#image("./media/mcts_alphago.jpeg") #image("./media/alphazero.jpeg")])
)

== Reward Hack - RL中的有趣现象

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  align(horizon, [
  Goodhart效应: 当一个指标变成目标，它就不再是一个好的指标。

  对于围棋而言，棋局胜负已经是一个充分客观且可靠的优化目标，但是对于更多现实应用而言却仍然可能造成各种问题：

  - 钻反馈规则的漏洞
  - 拍马屁/迎合用户
  - “看人下菜”与话术操控
  - 等等等等

  最终会导致训练崩溃并严重影响扩展与泛化潜力。
]),
  [
  #set align(center)
  #image("./media/boat_rwhack.gif", height: 40%)
  #image("./media/rlhf_rwhack.jpeg", height: 50%)
]
)


== 数学 - 也是另一种游戏吗？

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  align(horizon, [
  相比较麻烦的现实问题而言，数学任务有这样几项特点：

  - 相对有限的动作空间（应用定理做推导、计算）
  - 存在明确可验证的reward（定理证明的正确性是客观的，且基本没有漏洞）
  - 有趣且充满挑战性

  那么……是否针对数学问题应用强化学习，也能够取得令人惊叹的效果呢？
]),
  image("./media/math_axioms.png")
)

== 针对数学问题的RL尝试

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  align(horizon, [
  早在OpenAI o1之前，针对数学问题的各个代表性工作，如DeepSeek-Math和Qwen2.5-Math等模型的训练过程当中都已经普遍采用了RL算法来增强解题能力。

  但是经典的PPO强化学习算法要同时训练polycy model, reward model和value model三个模型，并额外保存一个reference model用于维持稳定。GRPO则简化去除了value model，但仍然相当复杂且收效并不显著。

  #image("./media/rl_results.png")
]),
  box([
    #image("./media/ppo_and_grpo.png")
    #image("./media/grpo_formula.png")
  ])
)

== PRM会更好吗？

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  align(horizon, [
  OpenAI o1发布之后，他们此前于2023年5月发布的论文_Let's Verify Step by Step_也开始受到大量关注。

  在这篇论文当中他们

  - 人工标注了MATH当中1.2万道题的7.5万个解答，总计80万个步骤的正确性
  - 在此基础上训练了大规模的outcome reward model和process reward model用于RL训练
  - 最终在GPT-4上实验验证了逐步评分的reward model训练效果更好
]),
  align(center + horizon, image("./media/lv_step_by_step.png"))
)

= 真相揭晓

#image-slide(img: image("./media/dsr1_release.png"))

== DeepSeek-R1正式发布

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  [
  DeepSeek-R1于2025年1月21日半夜发布，它是o1之后的第一个大型推理模型，并且：

  - 在各项性能指标上完全匹敌当时最强的最新版OpenAI o1！
  - API调用价格仅有o1的百分之一！
  - 完全公开权重以及训练技术的各种细节！

  #image("./media/dsr1_pyq.png", height: 40%)
],
  align(center, image("./media/dsr1_benchmarks.png"))
)

== 至简即是至强

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  [
  如非必要，全部砍掉！

  DeepSeek-R1 / R1-Zero所采用的训练方法是在GRPO的基础上进一步简化，其最主要的改进点在于

  - 在不进行指令微调SFT的情况下直接进行RL
  - 放弃使用reward model，而直接使用标准答案比对作为reward
  - 放开token数上限，允许模型自由探索

  在后续的GSPO等改进工作当中，人们更是进一步删去了reference model，此时的训练效果反而更好。
],
  box([
    #image("./media/dsr1_prompt.png")
    #image("./media/dsr1_improvement.png")
  ])
)

== RLVR的连锁反应

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  align(horizon, [
  DeepSeek-R1的这套训练方案后来被称为RLVR (RL with Verifiable Reward)，经过reward简化可以产生一系列的连锁反应：

  - 直接答案比对取代reward model
  - 数学问题奖励稀疏因此不需要value model
  - 答案比对稳定可靠，避免了reward hack，因此也不再需要reference model
  - 长时间的稳定RL训练带来了
    - 深度思考的自发涌现
    - 推理能力显著提升
]),
  box([
    #image("./media/dsr1_deepthink.png")
    #image("./media/dsr1_ahamoment.jpeg")
  ])
)

== 从书呆子到全面发展

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  align(horizon, [
    仅经过RLVR训练的DeepSeek-R1-Zero实际上还并不适用于各种通用任务，仍需要经过一套复杂的训练流程才能得到全面发展的DeepSeek-R1。

    在报告当中R1-Zero主要表现出的问题仍然包括

    - 思维链语言混杂，思考过程混沌不可读
    - Zero模型在数学和代码任务上专精训练，但无法应付任何日常事务，例如“你好”
  ]),
  image("./media/full_trainingflow_dsr1.png")
)

== 深度推理的时代

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  [
  DeepSeek-R1论文于2025年9月18日受邀稿登上Nature正刊封面，可谓实至名归。

  在DeepSeek公开深度推理的训练技术之后，各个大模型厂商都纷纷推出了自己的推理模型：

  - OpenAI: o3, GPT-5
  - Google: Gemini 2.5 Pro
  - Anthropic: Claude 4 / Claude 4.5
  - 阿里巴巴: Qwen3 series
  - 月之暗面: Kimi k1.5等 (独立发现)
  - ...

  一片勃勃生机万物竞发的景象已在眼前
],
  align(center, image("./media/dsr1_nature.jpeg"))
)

= 未来在何方？

== RLVR的发展与困境

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  [
  #quote(attribution: [Richard Sutton])[_"Large language models are trying to get by without having a goal… That’s just exactly starting in the wrong place."_]

  RLVR后续还有许多的改进与发展，但是其扩展潜力似乎已经开始遇到了瓶颈，主要问题可能有：

  - 不适用于无明确反馈的任务场景
    - (即使是数学证明题)
  - 难以针对长线任务做优化
  - 依然没有给予大模型足够的自主性
],
  [
  #image("./media/ttrl_illu.png")
  #image("./media/rl_quest.png", height: 55%)
]
)

== 天下大势 - 合久必分？

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  align(horizon, [
  推理模型即使在简单问题上也倾向于花费大量的token进行思考，这种「过度思考」这一问题似乎渐渐地带来了LLM和LRM的分裂。

  目前各家的解决方案基本是

  - DeepSeek: 根据特殊token选择思考长度
  - Qwen：特殊token $->$ 独立模型
  - OpenAI：特殊token + 独立模型
    - GPT-5则通过引入额外的router自动选择模型，实现了表面统一
  - Anthropic：独立模型
  - Google：似乎是独立模型
]),
  box([
    #image("./media/over_thinking.png")
    #image("./media/optimal_thinking.png")
])
)

== 自主智能体的大饼

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  [
  自主性与长程工作能力或许就会是大模型的下一个突破点。

  近期也有论文指出，大模型在单步任务上对数级的性能增长也可能带来长程任务上超指数级的提升！这样的任务可能包括

  - 数学等领域的科研任务
  - 大型软件工程
  - 辅助行政事务
  - ...

  但当前大模型仍然缺乏足够的自主性，当前基于prompt人工搭建智能体的模式或许也并不正确...
],
  box([
    #image("./media/long_term_task_setting.png")
    #image("./media/bigbing_of_long_term_task.png")
  ])
)

#focus-slide([敬请期待这片尚不知晓的未来吧！])

#focus-slide([后续七次讨论班我们会讲解目前\ 深度学习技术的各种细节，\ 如有兴趣也欢迎前来！])
