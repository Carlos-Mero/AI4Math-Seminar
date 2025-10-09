#import "@preview/may:0.1.1": *
#show: may-pre.with(
  config-info(
    title: [大语言模型],
    subtitle: [AI4Math讨论班第二季-01],
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

#title-slide()

= 前言

#focus-slide([请问如果仅使用C语言及其标准库编写一个大语言模型，大致需要多少行代码呢？])

#focus-slide([答案是——1000行足够！])

== 1000行代码的大语言模型

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  [
  参考自链接: #link("https://github.com/karpathy/llm.c", "https://github.com/karpathy/llm.c")

  其中包含有

  - 完整的模型结构
  - 训练代码
  - 推理代码
  - 应用CPU并行计算优化
  - 完整详细的注释

  其中使用的dataloader, tokenizer等代码没有包含在内。这部分同样总计数百行。

],
  image("./media/llm.c_code_0.jpeg"),
)

== Sutton苦痛的教训

#align(
  center + horizon,
  [
七十年AI发展为我们给出的答案便是…… (无论人们能够学到多少)

#image("./media/bitter_lesson.png")
]
)

#image-slide(body: [集中一点，登峰造极！], img: image("./media/llm_principle.jpeg"))

#focus-slide([最丑陋也最优雅的工程——

欢迎来到人工智能！])

= 从人工智能到深度学习

== 从幻想到现实 - 从人工到智能

AI概念的演进同样代表了AI发展的三个阶段：#hl([人工智能 - 机器学习 - 深度学习])；单纯从幻想层面可能可以追溯到人造人，蒸汽朋克等等

#align(
  center,
  image("./media/ai_history.jpeg", height: 80%)
)

== 人工智能 - 意欲何为？

#quote(attribution: [Herbert A. Simon & Allen Newell, 1958])[
  _within ten years a digital computer will be the world’s chess champion, it will also discover and prove an important new mathematical theorem._
]

#quote(attribution: [Herbert A. Simon, 1965])[
  _"machines will be capable, within twenty years, of doing any work a man can do."_
]

// Simon逝世于2001年

「#hl([让机器像人一样思考])」是自AI概念诞生以来便始终存在的长久夙愿。

自第一次AI寒冬后，人们更多开始务实地考虑在各个#hl([特定任务])上可评测的性能提升与泛化能力。

直到2022年底，大模型时代之后，人们才开始重新考虑#hl([通用人工智能的可能性])。

== 最早的语言模型chatbot - Eliza

ELIZA的开发时间在1964 - 1967年间。

Eliza主要应用技术是完全基于规则的「模式匹配」和「文本替换」。可以一定程度上挑战图灵测试。但即使在当时人们也不认同将Eliza等同于智能。

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  align(
    center + horizon,
    image("./media/eliza_intro.jpg")
  ),
  align(
    center + horizon,
    image("./media/ELIZA_conversation.png")
  )
)

== 机器学习：后退一步海阔天空

如果按照标准的计算机程序，考虑两个相似的过程

/ 高级语言编译 (C语言 $->$ 机器码):\

  源文件 $->$ 预处理 $->$ 词法分析 $->$ 抽象语法树 $->$ 语义检查 $->$ 中端/后端优化 $->$ 生成汇编 $->$ 链接 $->$ 可执行文件

  为实现这一功能，GCC编译器源代码数量已经达到了1200万-2000万行量级。

/ 自然语言机器翻译 (中译英):\
  中文 $->$ ??? $->$ 英文?

考虑到自然语言的#hl([极端复杂性])与#hl([数不清的特例])，加上现实当中存在各种语言互相翻译的需求，严格的机器翻译程序基本上不可能完成。

但是作为信息处理工具，如果愿意放弃对于精确性的追求，或许还可以找到其他的出路。

== 世界的真理总是优雅的……大概吧

我们总可以将信息处理过程抽象为一个函数 (或者概率) 映射，从输入到输出。

而相比较精确地应用计算机指令构造出这些函数，我们也可以通过#hl([近似拟合])的方法达到效果。它在很多时候能够呈现出更多的实用性。

例如 $hat(y) = theta_0 + theta_1 x$，线性回归模型也可以视作最经典的机器学习模型，总计有两个参数。而在完成模型构建之后我们就可以根据采集到的数据点调整确定参数取值，以获取最优的模型。

现代大模型通常都有极大的参数量，以此拟合出及其复杂的信息处理映射，例如

#table(
  columns: 5,
  "GPT-3", "DeepSeek-V3系列", "Qwen3-max", "Hunyuan Image 3", "Hunyuan 3D 2.5",
  "1750亿", "6710亿", "1万亿", "800亿", "100亿"
)

== 深度神经网络

深度神经网络是目前最主流的机器学习模型，使用深度神经网络进行机器学习的方法就被称为深度学习。

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  [
  深度神经网络当中有两种最主要的计算层：
  - #hl([线性层])
  - #hl([卷积层])
  它们在和各种辅助性的激活层、正则化层、残差连接等复合叠加起来，就构成了神经网络的主体。

  其中目前大语言模型所应用的Transformer架构当中的计算层仅有线性层一种。
],
  image("./media/intro_dl.png")
)

那么……为什么一定要多层复合呢？

== 深度炼丹秘法

我们首先考虑最简单的单层线性神经网络的情形，此时我们模型的输入为向量 $vb(x) in RR^m$，期望输出为 $vb(y) in RR^n$，那么我们的参数矩阵就是 $vb(W) in RR^(n times m)$，模型所提供的输出就是 $vb(hat(y)) = vb(W) vb(x)$。我们的目标则是调整参数矩阵 $W$，使实际输出 $vb(hat(y))$ 和预期值 $vb(y)$ 尽量接近。

但是这样的模型会存在两个显著的问题：

- 线性模型#hl([只能有效拟合线性关系]) (有效指拟合误差可以逼近无限小)
- 它在模型结构和参数量上#hl([缺少扩展潜力])

假设我们强行添加一个隐藏层 $vb(z) in RR^l$，并且叠加两层线性层 $vb(A) in RR^(n times l), vb(B) in RR^(l times m)$，那么此时模型就变成了 $vb(hat(y)) = vb(A) (vb(B) vb(x)) = (vb(A) vb(B)) vb(x)$，#hl([和单个线性层完全没有区别！])

1969年，Minsky & Papert在《Perceptrons》书中指出当时的神经网络连最简单的异或xor函数都无法拟合，严重打击了人们对于AI技术的信心，是第一次AI寒冬的成因之一。

== 超越想象力的开始

在神经网络的后续发展当中，人们想到如果能够在两个线性层之间添加一个非线性的激活函数 $phi$，那么复合得到的模型 $vb(hat(y)) = (vb(A) compose phi compose vb(B)) vb(x)$ 就不会直接坍缩到线性。而根据复合函数求导的链式法则得到的反向传播算法，也能够支持这样多层神经网络的训练。

#grid(
  columns: (1fr, 0.5fr),
  gutter: 0.4em,
  [
  当前最常使用的激活函数是#hl([线性整流函数 (ReLU)]) 及其各种变种。它的具体表达式是

  $ "relu"(x) = max(x, 0) $

  如此一来隐藏层数以及参数量都可以#hl([无限扩展堆叠])上去，例如在Qwen3首发系列的语言模型当中，模型参数量在6亿到2350亿之间，堆叠的Transformer层数则在28层至94层之间。

  但仅仅像这样提高参数量的话，模型的拟合能力是否真的能够提高呢？
],
  image("./media/relu_act.png")
)

== 惊为天人 - 通用近似定理

#quote(attribution: [Ali Rahimi, NeurIPS 2017])[
  _"Machine learning has become alchemy."_ (机器学习已经变成了炼金术)
]

在泛函分析当中有一个经典的Stone-Weirstrass定理，它说的是对于任意一个紧致的Hausdorff空间 $K$，定义一个函数系 $C(K, RR) = {f: K -> RR | f "is continuous"}$，令 $||f|| = sup_(x in K) |f(x)|$。

那么此时 $C(K, RR)$ 就可以构成一个赋范线性空间，并且是一个代数。此时对于其中的一个子代数 $cal(A)$，如果它满足下面的两个条件：

- $cal(A)$ 当中包含所有常值函数
- $forall xi != eta, exists g in cal(A), g(xi) != g(eta)$

那么这个子代数 $cal(A)$ 在 $C(K, RR)$ 当中就是#hl([稠密的])。

#pagebreak()

从这个定理出发就可以推出AI领域的一个重要结论，被称为「通用近似定理」：

一个三层的神经网络足以拟合任意一个紧集之间的可测函数。也就是说，当隐藏层参数趋于无穷时，这种模型的拟合误差可以无限小。(当然这个定理存在很多变种表述，某些特定版本也未必需要依赖于泛函分析证明)

而这个结果甚至几乎并不依赖于激活函数的选取，通常只需要加入一个正常的非线性函数即可。

从这里开始神经网络在理论上就实现了一个重要的转变：

只能拟合线性函数 $==>$ #hl([可以拟合几乎任意函数！])

可以说未来神经网络所能取得的各种超乎想象的能力最终都来源于此。

== 数据、损失函数与训练

通用近似定理仅仅保证了完美拟合的存在性，而寻找这一参数的过程则依赖于#hl([模型训练])。

当前模型训练的基本方法是首先将模型#hl([初始化为随机参数])，根据训练目标设计一个连续可微的#hl([损失函数]) (或目标函数) 来衡量训练目标，再通过#hl([类梯度下降])和#hl([反向传播])方法来调整参数寻找#hl([使损失函数最小]) (或使目标函数最大) 的参数。下表会给出一些例子

#table(
  columns: 4,
  "训练方案", "损失函数主项", "通常数据需求量", "训练目标",
  "LLM预训练", [$sum y_i log p_i$], "数十万亿tokens", "学习语言的基本规律，掌握各种知识信息",
  "LLM监督训练", [$sum y_i log p_i$], "数十万样本", "偏好对齐以及指令遵循",
  "推理强化学习", [$EE_t (pi_theta (a_t | s_t))/(pi_(theta_"old") (a_t | s_t)) A_t$], "数千道题", "强化深度推理能力",
  "flow matching", [$EE ||v_theta - u||^2$], "数十亿图文对", "训练文生图的基本能力",
  "AlphaZero", [$EE (z-v)^2 + pi^TT log p$], "无", "自我对弈强化围棋能力"
)

== 深度学习的时代

#grid(
  columns: (1fr, 1fr),
  [
  #quote(attribution: [Geoffrey Hinton])[
    _"I'm proud of the fact that I stuck with neural networks even when people said they were rubbish."_
  ]

  深度学习的一般范式就由上述四个概念组成：

  - 模型结构 (深度神经网络)
  - 数据
  - 损失函数
  - 优化方法

  到今天几乎所有的现代AI领域的成果都属于深度学习的范畴。
],
  figure(
    caption: [Hinton发表诺奖获奖感言],
    image("./media/hinton.jpeg", height: 90%)
  )
)

#focus-slide([中场休息时间！])

= 语言模型

== 连续与离散的世界

// 两边各自都有自己的大饼，世界模型和AGI

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  figure(
    caption: [
    计算机视觉 (CV)

    连续与具体的世界
  ],
    image("./media/tas_in_classroom.png", height: 80%)
  ),
  figure(
    caption: [
    自然语言处理 (NLP)

    离散与抽象的世界
  ],
    box(
      height: 80%,
      align(
        center + horizon,
        [两位助教老师正并排坐在一间大学的阶梯教室当中，两人正在互相交谈着什么事情。他们的身边还有形形色色的其他同学。]))
  )
)

== 语言模型的发展历程

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  [
  #quote(attribution: [Ludwig Wittgenstein])[
    _"The limits of my language mean the limits of my world."_
  ]

  #line()

  语言与推理向来被视作人类最重要的能力之一。创建一个能够掌握，理解与运用语言的信息处理系统也是长久的夙愿。

  长期以来NLP领域都存在多个彼此独立，又有诸多共同点的研究方向。

  而#hl([统一与通用化])则是NLP领域长期的趋势。
],
  [
  #figure(
    caption: [LM与LLM的相关论文数量],
    image("./media/lm_paper_count.png"))
  #figure(
    caption: [LM发展的四个阶段 (2025年后或许可称为LRM时代)],
    image("./media/lmprogress.png"))
]
)

== 各种语言相关的任务

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  [
  传统的NLP任务包括

  - 机器翻译
  - 文本分类
  - 问答系统
  - 文本摘要
  - 情感分析
  - ...

  等等

  如果我们暂时将深度神经网络只当作一个可以#hl([拟合特定函数的黑盒])，那么如果希望使用它来解决这些任务，还需要做哪些准备工作呢？
],
  align(center, image("./media/nlp_tasks_wordcloud.png"))
)

== 词嵌入 - 离散到连续

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  [
  深度神经网络仅接受有实意的连续输入。因此处理语言任务时通常需要:

  - 将输入语言切分为词或#hl([token的序列])
  - 将每一个切分单元#hl([独立映射到语意向量])
  - 输入神经网络并计算得到结果

  在这里语意向量本身就可以作为参数参与训练。而在深度学习发展早期，nlp领域就已经产生了Word2Vec, GloVe这样的工作。它们各自通过预训练方案构建了一套通用的词向量表，并可以直接适配多项任务。

  嵌入后的词向量同时也会呈现出许多有趣的性质。
],
  figure(
    caption: [Word embedding结果示意图],
    image("./media/word_embedding.jpeg"))
)

== 从图像分类器到语言模型?

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  [
  图像分类是CV领域最经典的任务之一，而它的实现思路其实和现代大语言模型异曲同工。

  图像分类任务：连续的图像信息 $->$ 离散的类别标签

  但是我们无法直接对离散的输出结果进行求导优化，因此图像分类器实际采用的方法是#hl([输出各个标签的概率])。

  在后续训练当中通常所使用的损失函数就是#hl([交叉熵损失])：$cal(L) = EE[y log p_y]$。
],
  [
  #set align(center)
  #image("./media/mnist.png", height: 40%)
  #image("./media/mnist_classifier.jpeg", height: 50%)
]
)

== 概率采样 - 连续到离散

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  [
  现代语言模型在生成侧采用的做法与图像分类器基本相同，每次生成时都会#hl([同时输出token列表当中每一个token的概率])。而后则会通过各种算法采样得到单个离散的token。

  常见的采样算法包括:

  - greedy, 直接选取概率最高的token
  - top p, 在概率总和高于$p$的token当中随机采样
  - top k, 在概率前$k$高的token当中随机采样
  - temperature, 手动修改概率调控分布熵

],
  figure(
    caption: [Qwen3-8B模型当中的部分token及其对应id，token总量为151669],
    image("./media/tokens_in_qwen3.png"))
)

== 走向统一的NLP

#grid(
  columns: (1fr, 0.7fr),
  gutter: 0.4em,
  [
  早在2013年的Word2Vec这项工作当中，人们就已经尝试了通过无监督训练的方式构造通用化的词向量表，用于各种专用任务上。

  那么如果更进一步，是否可能构建一个#hl([通用的NLP模型])，使其中绝大部分结构都具有在不同任务上的复用能力，而仅需要做少量微调？

  或者说，如果#hl([任务本身就是「理解和运用语言」])，那么应该如何做到这一点呢？

  自2017年transformer架构首次针对机器翻译任务提出以来，NLP领域渐渐分出了两种通用的#hl([无监督预训练范式])，GPT和BERT。
],
  image("./media/nlp_map.png")
)

== 填空题与续写题

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  align(horizon, [BERT (Bidirectional Encoder Representations from Transformers) 的预训练模式，通过随机删除文本当中的token，并让模型预测被删去的token，以此建立#hl([语言理解能力])]),
  align(center, image("./media/bert_pretrain.png", height: 45%))
)

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  align(center, image("./media/gpt_pretrain.png", height: 45%)),
  align(horizon, [GPT (Generative Pre-trained Transformer) 的预训练模式，通过固定删除文本末尾的token，让模型尝试续写还原原文本，以此建立#hl([语言运用能力])]),
)

== BERT与预训练-微调范式

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  [
  Google BERT模型带来了前GPT时代的第一波统一浪潮。

  初代BERT通过在当时最大的数据集BooksCorpus和Wikipedia上进行广泛的预训练，由此获得了大量语言领域的先验知识。在BERT基座模型上经过微调后即可在11项NLP任务上取得sota的表现。

  NLP领域从此正式确立了预训练-微调的训练范式，并且一路沿用到了现在。

  右图即是DeepSeek-V3所公布的各阶段训练成本。
],
  [
  #set align(center + horizon)
  #image("./media/bert_intro.jpeg", height: 70%)
  #image("./media/dsv3_cost.png")
]
)

== GPT与自回归生成

相比较BERT侧重理解任务而言，GPT是一种原生的生成式模型。由于其训练目标是不断预测结尾的下一个token，只需要将其采样后的token迭代添加到输入当中，即可连续地生成一长段文本。这一过程就被称为「#hl([自回归 (Autoregressive) 生成])」。

GPT-1 (117M) 早在2018年就已发布，随后GPT-2对其进行了参数与训练数据的扩展，并首次观察到了「零样本迁移」的能力。但在当时GPT类模型的性能表现尚且不足。

#image("./media/gpt_roadmap.png")

= GPT一统江湖

== 零样本迁移

在2019年，GPT-2的首发论文_"Language Models are Unsupervised Multitask Learners"_当中，OpenAI就有着重提到当时发现的「零样本迁移」能力。

原论文当中提到的一个例子是……假设你在像GPT-2一样执行自回归生成任务，并且需要接着下面的这一段话继续生成——

“I hate the word ‘perfume,”’ Burr says. ‘It’s somewhat better in French: #hl([‘parfum.’])

是的！你已经成功在不知不觉间完成了经典的「机器翻译」任务。除此之外的各种语言任务也都可以通过类似的方法执行。这一现象在当时已经引起了人们的普遍兴趣，但它仍然受限于

- 性能表现不足
- 高度依赖prompt，稳定性不佳

但在实验当中已经能够观察到随着模型尺寸增加，GPT零样本迁移的能力有显著提升。

== In-Context Learning

#grid(
  columns: (1fr, 0.6fr),
  [
  时间来到2020年5月，此时OpenAI破天荒地将GPT-2的模型参数量扩张了100倍有余，达到了空前的1750亿。这使得GPT-3具有了显著更强的零样本迁移能力，能够在不经过专门训练的情况下，仅通过prompt指导即可在42项经典的NLP任务上取得sota的性能表现。

  OpenAI此时首次将这种能力定名为#hl([In-Context Learning])，并区分了zero-shot, one-shot, few-shot三种不同的使用方式 (如右图)。

  但此时的GPT模型仍有几项局限：

  - 使用麻烦，不稳定且成本高昂
  - 在经典NLP任务上缺少明显优势
  - 缺少独有的killer feature
],
  align(center, image("./media/icl_settings_gpt3.png"))
)

== 后训练：从GPT到ChatGPT

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  [
  从GPT-3发布后的两年内不断研究演进的后训练技术是引导NLP走向统一的最后一块拼图。

  大模型事实上已在预训练阶段获取了足够丰富的通用知识和语言能力，#hl([后训练则是通过少量训练引导它充分发挥自己的潜力])。GPT时代的后训练方法主要包括：

  - #hl([监督训练 (SFT)])：在特定格式的指令遵循、对话等数据上微调，使其适应特定的输入和输出格式。
  - #hl([基于人类反馈的强化学习 (RLHF)])：应用PPO等算法，基于人类反馈评价来调控LLM的输出。
],
  figure(
    caption: [SFT数据示例],
    image("./media/instructions.png"))
)

== 从量变到质变的涌现

经过完善后训练的GPT-3模型具备了理解用户指令、高效执行zero shot任务以及多轮对话等等能力。它的输出稳定性得到了显著的提高，同时其在#hl([指令理解上的泛化能力])更是极大地拓展了语言模型的应用空间。

最终成品ChatGPT于2022年11月30日正式发布，并从此开启了NLP大一统的时代 (一个模型胜任所有NLP任务)。

#figure(
  caption: [在GPT之后，所有模型都是GPT],
  image("./media/gpt_5_dialog.png"))

== 最终回顾 - 大道至简

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  [
  在1000行代码的GPT-2实现当中，主要就包括如下内容：

  - 各种网络层的前向与后向计算
    - 位置编码层
    - LayerNorm层
    - 矩阵乘法
    - 注意力层
    - GeLU激活函数
    - ...
  - 交叉熵损失函数
  - 模型结构定义
  - 训练循环、采样代码等

  这便是数十年NLP研究的精髓所在。
],
  figure(
    caption: [llm.c代码符号表],
    image("./media/symbols_llm.c.png", height: 90%)
  )
)

#image-slide(img: image("./media/nlp_united.png"), body: [一个统一而强盛的NLP帝国，下一步的目标将会是…… ？])

== 数学 - 是否会是通向真正智能的道路？

#image("./media/math_progress.png")

#image-slide(img: image("./media/gtm_series.jpeg"), body: [当人类自身成为瓶颈，AI是否有能力再次改变一切？])

#focus-slide([
  敬请期待下集 - 深度思考革命！
])
