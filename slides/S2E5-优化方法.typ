#import "@preview/may:0.1.1": *
#show: may-pre.with(
  config-info(
    title: [优化方法],
    subtitle: [AI4Math讨论班第二季-05],
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

#image-slide(img: image("./media/s2e5.png"))

#title-slide()

= 前言

== Kimi K2 Thinking

#tcslide([
  #link("https://moonshotai.github.io/Kimi-K2/thinking.html", "Kimi K2 Thinking")模型于2025年11月7日正式发布，标志着国内开源模型在性能上再次达到匹敌GPT-5以及Claude Sonnet 4.5等闭源sota的表现。

  Kimi将这一模型称为thinking agent，因其最主推的能力就在于结合工具调用的多步长推理，并且能够在无人为干预的情况下实现数百步的连续工具调用。

  而在此前的Kimi K2基础模型当中，他们就率先开创性地使用了MuonClip优化器来进行训练，结合数据合成与通用RL算法取得了相当好的性能表现。
], figure(image("./media/k2-gpt5-reasoning.png")))

== Kimi K2 Thinking Benchmarks

#align(center, image("./media/kimi-k2t-eval.png"))

== 人类的最终测试

#tcslide([
  #link("https://huggingface.co/datasets/cais/hle", "HLE (Humanity’s Last Exam)") 的目标是成为同类当中最后一个仍有区分度的考试项目。它从2024-09-16开始面向全球征集问题，经过多轮复核和筛选后最终于2025-04-03发布第一个定稿版本，总计2,500题。

  HLE当中有约四成为数学问题，其余则覆盖理化、生物、人文社科等各种领域。它同时保留有一份私有数据集用于防范针对性训练以及过拟合等情况。

  在它刚刚发布时最强大的o1也只能达到8%的准确率，而如今的Kimi-K2 Thinking已经能够达到51%的准确率上限了。
], image("./media/hle_samples.png"))

#focus-slide([
  #set page(margin: 2em)
  自ChatGPT之后，各个厂商的基础模型结构已经不再是影响性能的关键因素。决定性的差距在于后续的训练以及寻找最优参数的过程当中。
])

= 训练算法

== 训练目标

#align(horizon, tcslide([
  在机器学习时代之后我们通常会将一个模型的训练视作一个优化过程。更进一步地在深度学习时代，针对神经网络的优化算法基本都使用类梯度下降和反向传播算法。

  右侧列举了一些常见的损失函数，通常而言损失函数所计算的数值并不直接等同于我们的最终训练目标，也因此单纯的loss值通常并不作为性能指标使用。RL算法的一大意义就在于通过引入额外的reward计算损失函数，使得优化方向更加接近我们的最终需求。
], [
  当前的LLM在训练开始前通常直接将所有参数按照$cal(N)(0, 0.02)$的正态分布随机初始化，此后再通过训练调整具体的参数值。

  #table(
    columns: 2,
    [训练方案], [损失函数主项],
    [图像分类], [$EE sum y_i log p_i$],
    [LLM预训练], [$EE sum y_i log p_i$],
    [LLM指令微调], [$EE sum y_i log p_i$],
    [推理强化学习], [$EE_t (pi_theta (a_t | s_t))/(pi_(theta_"old") (a_t | s_t)) A_t$],
    [Flow Matching], [$EE ||v_theta - u||^2$],
    [AlphaZero], [$EE (z-v)^2 + pi^TT log p$]
)]))

== 随机梯度下降与反向传播

#tcslide([
  最基础的神经网络训练方法就是针对损失函数的随机梯度下降 (Stochastic Gradient Descent, SGD)。

  对于给定的损失函数 $L(theta, d)$，此处 $theta$ 是模型的参数，$d$ 是给定的样本，我们的优化目标就是寻找使得 $EE_d L(theta, d)$ 最小的 $theta$ 值。由于这个期望通常无法直接计算，SGD算法的思路就是通过随机选取 $m$ 个样本去计算出经验损失，并针对它进行优化，也就是：

  $ theta <- theta - eta 1/m sum_(i=1)^m gradient_theta L(theta, d_i) $
  // 那么当m发生变化时学习率$eta$又应该如何相应地改变呢？
], box([
  #set align(left)
  反向传播算法根据复合函数的梯度运算规则逐层计算梯度，例如对于线性层而言其核心计算公式就是：

  $
  delta^l = ((W^(l+1))^TT delta^(l+1)) dot.o sigma'^l (z^l)
  $

  这里 $delta^l$ 就是第 $l$ 个隐藏层变量的偏导数，$sigma^l$ 就是激活函数，$W^(l+1)$ 是上一层的参数。

  #align(center, image("./media/output_sgd_baca77_39_1.svg"))
]))

== 梯度下降当中的三类问题

#tcslide([
  在这种常规的梯度下降计算过程当中可能会遇到几类问题：

  - 陷入局部最优 (实际上基本不存在)
  - 鞍点附近梯度过小难以优化
  - 由于激活函数或者多层反向传播，导致深层梯度消失，难以优化
    - 与之对应的梯度爆炸现象则只需通过clip机制即可很好地解决

  后续的各种改良优化算法很大程度上就是为了解决后两类问题，提升优化效率。
], box([
    #image("./media/output_optimization-intro_70d214_39_0.svg")
    #grid(
      columns: 2,
      image("./media/output_optimization-intro_70d214_51_0.svg"),
      image("./media/output_optimization-intro_70d214_75_0.svg"),
    )
  ]))

== 动量优化

#tcslide([
  一项针对SGD优化器的经典改良就是引入动量 (momentum) 机制。引入动量机制后的SGD会使用多步累积的梯度值动态修正参数更新的速度。

  具体而言对于衰减超参数 $beta$ 和学习率 $eta$ 而言，动量SGD的参数更新公式就是

  $
  vb(v)_t &<- beta vb(v)_(t-1) + vb(g)_t\
  vb(theta)_t &<- vb(theta)_(t-1) - eta vb(v)_t
  $

  注意此时我们需要额外存储一个与参数量同样大小的动量 $vb(v)_t$，这会消耗显著更多的显存，但确实能有效提高稳定性和优化速度。
], [
  #align(horizon, figure(image("./media/ming_momentum.png"), caption: [动量机制演示，其最初的灵感就可能来源于物理世界]))
])

== Who's Adam?

#tcslide([
  Adam (Adaptive Moment Estimation, 目前更多使用的是AdamW这一版本) 是目前使用最为广泛的优化器。它在momentum机制的基础上额外引入了二阶动量来动态调控每一个参数位置上的学习率，以此实现了更快的收敛速度。

  $vb(v)_t$ 这一项就会定向放大小梯度方向的学习率，同时缩小大梯度方向的学习率。

  公式当中最后 $lambda vb(theta)_t$ 这一项的作用就是额外引入weight decay正则化。
], [
  $
  vb(m)_t &<- beta_1 vb(m)_(t-1) + (1 - beta_1) vb(g)_t\
  vb(v)_t &<- beta_2 vb(v)_(t-1) + (1 - beta_2) vb(g)_t^2\
  hat(vb(m))_t &<- (vb(m)_t)/(1 - beta^t_1),quad hat(vb(v))_t <- (vb(v)_t)/(1 - beta_2^t)\
  vb(theta)_(t+1) &<- vb(theta)_t - eta ((hat(vb(m)_t))/(sqrt(hat(vb(v))_t) + epsilon) + lambda vb(theta)_t)
  $
  #image("./media/whos-adam.webp", height: 40%)
])

== 各种优化器的实验验证

#tcslide([
  这里我们也提供了一个用于测试上述三种经典优化器的实验代码。

  在"vit_optim.py"当中我们实现了一个小型的Vision Transformer来执行cifar100数据集上的图像分类任务。我们在实验当中使用的三种优化器设定分别是

  ```py
  optimizer = optim.SGD(model.parameters(), lr=0.01)
  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  optimizer = optim.AdamW(model.parameters(), lr=3e-4)
  ```
], figure(image("./media/cifar100.jpg"), caption: [cifar100数据集示例]))

== 各种优化器的实验结果

#figure(image("./media/vit_cifar100_optim_compare.png"))

== Muon - 从向量到矩阵的本质跨越

#tcslide([
  此部分内容建议阅读#link("https://zhuanlan.zhihu.com/p/13401683661", "苏剑林的知乎回答")。

  Muon优化器是在2024年底才首次提出并逐渐走向应用的。如果我们仔细考察前述AdamW等优化器的话，可以看出它们都将线性映射当中的参数矩阵视作向量来独立地进行优化。而Muon优化器则是真正地将它们作为矩阵处理，并取得了相比较AdamW更强的性能表现。

  简而言之，SGD方法是在F范数约束下的最速梯度下降，Muon所走的则是在谱范数约束下的最优优化方向。
], [
  #figure(image("./media/muon-adamw.jpg"), caption: [Muon优化器和AdamW在语言模型上的性能对比])
])

= 训练与正则化

== 稳定性与泛化

#align(horizon, tcslide([
  神经网络的泛化性 (Generalization) 是深度学习当中最重要的课题之一。所谓泛化指的就是神经网络将自身所学到的知识推广应用到训练范畴之外的能力。

  强大的泛化能力依赖于AI模型各方面的优化，例如数据、损失函数、模型结构等等，当然其中也有一部分能力需要依赖于训练算法的选择。

  通常而言我们期望神经网络能够学习到更具*普适性*且*通用*的知识，而这样的结果通常都是不依赖于特定样本的。防范过拟合的方法就被称为正则化 (Regularization)。
], [
  #set align(left)
  模型训练的三种状态：
  
  - 欠拟合：训练集表现与泛化能力均不佳
  - 适当拟合：训练集表现与泛化能力均较好
  - 过拟合：训练集表现极好而泛化能力不佳

  #figure(image("./media/overfitting_21-2.png"), caption: [三种拟合状态的示意图])
]))

== Lipschitz约束与Weight Decay

#tcslide([
  或者数学上说，我们希望训练得到的模型能够满足一个较好的#link("https://kexue.fm/archives/6051", "Lipschitz稳定性")，由此对于输入的微小扰动可以保持稳定，也就是：

  $
  ||f_theta (x_1) - f_theta (x_2)|| <= C(theta) ||x_1 - x_2||
  $

  这里 $f_theta$ 表示带参数的模型，而 $x_1, x_2$ 则是模型的输入。此处使用的范数可以有不同可能。由于现代神经网络基本上都是由多个线性层 (或卷积) 的复合，因此也许我们可以仅考虑单层线性层的情形。

  此时如果参数矩阵为 $W$，那么目标就变成了期望在参数矩阵的谱范数

], [
  #set align(left)
  $
  ||W|| = (||W (x_1 - x_2)||)/(||x_1 - x_2||)
  $

  它尽可能小的同时保持足够的拟合效果。但由于谱范数通常难以计算，因此实践当中常常可以将约束条件修改为矩阵的Frobenius范数 (将矩阵当作向量计算的范数)，而我们总有 $||W||_F >= ||W||$，因此它也能发挥效果。将二范数这一项加到损失函数上，就得到了经典的Weight Decay优化。
  
  「所谓的正则化也就是正常化」

  函数在接近无穷时会变得病态，那就不让它走到无穷。
])

== 归一化层

#tcslide([
  归一化层 (Normalization Layer) 最早在2015年左右提出，目前在各种神经网络当中都得到了非常广泛的应用。而在当前的大语言模型当中最常使用的是LayerNorm和RMSNorm两种。

  简而言之LayerNorm的计算方式就是将每层的输入减去均值再除以方差，写成公式就是

  $
  "LN"(vb(x)) = (vb(x) - mu)/(sigma + epsilon) dot gamma + beta
  $

  这里的 $gamma$ 和 $beta$ 都是可学习的参数。
], [
  #set align(left)
  RMSNorm (Root Mean Square Normalization) 则是在LayerNorm的基础上进行了进一步简化，去掉了均值的计算而直接采用 $L^2$ 范数进行归一化，也就是

  $
  "RMSNorm"(vb(x)) = (vb(x))/sqrt(||vb(x)||_(L^2) + epsilon) dot gamma
  $

  归一化层在模型结构当中的意义就在于稳定数值，由此可以实现：

  - 前向计算时提供一定的正则化
  - 反向计算时缓解梯度消失和梯度爆炸
])

== 现代的大语言模型的常规组成

#tcslide([
  - 模型结构
    - Tokenizer + 词嵌入 + 位置编码
    - Self-Attention + 线性层 + 类ReLU
    - 残差连接
    - RMSNorm归一化层
  - 训练数据
    - 大量无标注文本
    - 指令遵循样本
    - 可验证推理问题
  - 损失函数 (训练目标)
    - Next Token Prediction ($EE sum y_i log p_i$)
    - 强化学习 (未完待续)
  - 优化算法
    - AdamW (或者Muon)
], image("./media/transformer.png"))

#focus-slide([
  #set page(margin: 1.4em)
  在下一次讨论班当中我们就会讲解智能体相关的概念。它后续强化学习的概念基础，同时基于语言模型制作的智能体系统也已经取得了相当多的成果。
])
