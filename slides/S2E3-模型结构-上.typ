#import "@preview/may:0.1.1": *
#show: may-pre.with(
  config-info(
    title: [模型结构-上],
    subtitle: [AI4Math讨论班第二季-03],
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

#image-slide(img: image("./media/s2e3.png"))

#title-slide()

= 前言

== 新闻环节 - Sora 2

#tcslide([
  OpenAI Sora 2视频生成模型发布于2025年9月30日，其在前代的基础上实现了物理一致性、动作幅度、多样性等等维度的大幅进步。Sora 2同时提供免费版本给各家使用，由此开启了一个#link("https://www.bilibili.com/video/BV1VVxiz6E5u", "抽象创作的时代")。

  OpenAI基本没有透露Sora 2实现方式的各种细节，不过根据此前的信息推测，其模型结构应该仍然是DiT (Diffusion Transformer)。
], box([#image("./media/sora2_daqingxi.png", height: 45%) #image("./media/sora2_hk_rail.png", height: 45%)]))

== Sora初代

#tcslide([
  Sora初代于2024年2月首次公布，由此引起了视频生成模型的浪潮。(尽管它直到2024年底才正式公开，并且效果远不如早期宣传或者同期的其他视频生成模型)

  Sora所使用的模型结构来自2023年3月的一篇论文，_Scalable Diffusion Models with Transformers_。这篇论文的主要贡献在于使用标准的transformer架构替换了此前图像生成领域最常用的U-Net架构，并取得了不错的效果。它一开始被质疑创新性不足并被CVPR拒稿。
], box([#image("./media/dit_structure.png", height: 45%) #image("./media/sora_sample.png", height: 45%)]))

== 主要参考书 - D2L

#tcslide([
  后续科普部分主要使用的参考书目是:

  #link("https://d2l.ai/", "Dive into Deep Learning")

  这本书主要由李沐等一众Amazon大佬牵头编写，质量较高、简单易学且配有丰富的代码示例。也非常推荐用于自学使用。

  不过书中内容基本截止于2022年，ChatGPT流行之前的时间段。后续内容在我们的讨论班当中也会做一些补充。
], image("./media/d2l-front-cup.jpg"))

= ImageNet

== 深度学习浪潮的起源

#grid(
  columns: (auto, auto),
  gutter: 0.4em,
  [
  在 2012 年，来自多伦多大学的三位研究者提出了一个名为「AlexNet」的深度神经网络，赢得了 2012 年大规模视觉识别挑战赛 ImageNet 的冠军。

  三位作者如今都是AI领域里响当当的人物。Geoffrey Hinton被誉为「深度学习之父」，Ilya Sutskever是OpenAI的联合创始人及前首席科学家，Alex同时也是CIFAR-10和 CIFAR-100数据集的创建者。

  在描述当年的 AlexNet 项目时，Geoffrey Hinton 总结道：「Ilya 认为我们应该做这件事，Alex 让它成功了，而我获得了诺贝尔奖。」
],
  align(horizon, figure(caption:[AlexNet三位作者合影], image("./media/alexnet.jpg")))
)

== ImageNet的诞生

#tcslide([
  ImageNet最早由斯坦福大学李飞飞团队带头，调动全球167个国家近五万名标注者在网络上实现众包标注，耗时三年完成了1500万张图像以及22000个类别的标注工作，成为了当时世界上最大的视觉数据集。ImageNet数据集最早于2009年投递到了计算机视觉顶会CVPR当中，并作为poster论文中稿，受到的关注并不多。

  后来为了增强ImageNet的影响力，李飞飞团队举办了ImageNet大规模视觉识别挑战赛，需要参赛选手在一个ImageNet的子集140万张图像当中完成1000个类别的分类任务。
], align(horizon, figure(
    caption: [李飞飞及她的自传宣传图],
    image("./media/lifwfw.jpg")
  )))

== AlexNet首次夺魁

在开始的两年里 (2010 - 2011)，ImageNet赛事并没有收获预期的效果，参赛选手基本上都直接使用支持向量机算法参赛，效果进步有限且没有带来什么新的技术突破。因此ImageNet早期参赛人数逐年下降，逐渐遇冷。直到AlexNet之后取得了显著的性能提升，这一赛事才再次受到关注。

#align(center, image("./media/imgnet_samples.png", height: 65%))

= 经典神经网络

== MNIST图像分类任务

#tcslide([
  MNIST是在1998年由Yann LeCun发布的经典图像识别测试集，主要测试的是手写数字的识别任务。整个MNIST由6万张训练集和1万张测试集组成，每一个图像都是28x28分辨率的灰度图像，共分为数字0-9十个类别。

  我们通常采用one-hot编码以及概率观点来处理图像分类问题。具体而言我们期望的图像分类器是一个函数 $phi: RR^(28 times 28) -> RR^10$，输入图像后直接输出图像属于每一个类别的概率。在深度学习当中我们就可以通过训练一个神经网络来拟合构造这一函数。
],
  align(horizon, figure(caption: [MNIST图像分类数据集示例], image("./media/mnist.png")))
)

== 线性神经网络

#tcslide([
  为实现这一目标最简单的就是使用线性神经网络。我们可以直接通过将图像展平成一个$28 times 28$尺寸的向量，然后引入一个形状为$M(28 times 28, 10)$的参数矩阵并使用矩阵乘法计算得到结果，如有需要也可以额外引入一个偏置等等。最终我们通过一个softmax函数就可以将输出数值固定转换为一个概率分布。

$
vb(O) = vb(X) vb(W) + vb(b),\
vb(hat(Y)) = "softmax"(vb(O))
$

在这其中$"softmax": RR^n -> [0, 1]^n, vb(o) |-> vb(y) "where" y_i = (exp(o_i)) \/ (sum_j exp(o_j))$。
], figure(caption: [pytorch部分代码示例], [
  ```py
  # 数据加载
  transform = transforms.Compose([transforms.ToTensor()])
  train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
  test_data = datasets.MNIST('./data', train=False, transform=transform)
  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_data, batch_size=batch_size)

  # 定义模型（单层线性网络）
  model = nn.Sequential(
      nn.Flatten(),
      nn.Linear(28*28, 10)
  )
  ```
]))

== 交叉熵损失函数 / SGD优化器

#tcslide([
  交叉熵 (Cross Entropy) 的概念来自于信息论，对于概率分布$p, q$而言，一个概率分布的熵就是

  $
  H(p) = -integral log p(x) dif F_p (x)
  $

  它用于衡量一个随机变量的不确定性。同时两个随机变量的交叉熵就是

  $
  H(p, q) = -integral log q(x) dif F_p (x)
  $

  交叉熵度量的是，如果使用分布$q$来近似分布$p$，我们所需要的额外信息。只有当$p = q$时交叉熵才会取得最小值，刚好等于$H(p)$。
], [
  #set align(left)

  ```py
  # 构造损失函数和优化器
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=learning_rate)
  ```

  优化器上我们就选用最直接的SGD (随机梯度下降) 优化器即可。我们将最小化模型与真实分布的交叉熵作为训练目标，其参数$vb(theta)$的更新规则就是简单的

  $
  vb(theta) <- vb(theta) - eta nabla tilde(H)(p, q_vb(theta))
  $

  这里的$tilde(H)$是根据随机采样数据估算的交叉熵，$q_vb(theta)$就是我们的神经网络。
])

== 训练结果

#tcslide(
  [
  完整训练代码参见#link("https://github.com/Carlos-Mero/AI4Math-Seminar", "GitHub仓库")当中的"src/linear_mnist.py"文件，在完成环境配置之后可以直接在家用电脑的CPU上快速运行。

  在经过多轮训练之后，这样的神经网络实际上已经能在这个任务上达成90%的准确率了。(此时总参数量7,840)

  ```md
  Epoch [1/5], 准确率: 86.68%
  Epoch [2/5], 准确率: 88.35%
  Epoch [3/5], 准确率: 89.18%
  Epoch [4/5], 准确率: 89.70%
  Epoch [5/5], 准确率: 90.01%
  ```
],
  [
  ```py
  # 训练循环
  for epoch in range(epochs):
      # 训练阶段
      model.train()
      print(f"开始第{epoch+1}轮训练")
      for images, labels in tqdm(train_loader):
          outputs = model(images)
          loss = criterion(outputs, labels)
          
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      # 测试阶段
      model.eval()
      # ...
  ```
]
)

== 多层感知机

#tcslide([
  这里我们同样可以实现一个多层感知机 (Multi-Layer Perceptron, MLP) 作为一个更强的模型来执行这一任务。只需要在数个线性层中间叠加上合适的激活函数，即可实现一个经典的多层感知机。

  可以看出通常情况下更大的模型训练速度更慢，但最终能够达到更强的性能上限。(总参数量535,818)

  ```md
  Epoch [1/5], 准确率: 83.61%
  Epoch [2/5], 准确率: 89.04%
  Epoch [3/5], 准确率: 90.32%
  Epoch [5/5], 准确率: 91.98%
  Epoch [10/10], 准确率: 94.15%
  ```
], [
  #set align(left)

  具体实现可参考文件"src/mlp_mnist.py"，此处相比较前文仅仅修改了神经网络结构，关键代码为

  ```py
  model = nn.Sequential(
      nn.Flatten(),
      nn.Linear(28*28, 512),    # 第一层：输入层→隐藏层1
      nn.ReLU(),                # 激活函数
      nn.Linear(512, 256),      # 第二层：隐藏层1→隐藏层2
      nn.ReLU(),                # 激活函数
      nn.Linear(256, 10)        # 第三层：隐藏层2→输出层
  )
  ```
])

== 卷积神经网络

线性神经网络和多层感知机本质上第一步都是将一个二维图像展平转换成一个一维的向量再进行处理，但是这样一来很显然会忽略掉图像这一模态当中的很多特性，最终效果就相对欠佳。此时如果能够利用上我们关于图像的一些先验认知，就有可能使用大幅提高神经网络在图像任务上的表现。

图像模态有两个重要的先验特性可以使用:

/ 局部性: 图像当中的物体特征更多依赖于局部的信息
/ 平移不变性: 同一个物体位置移动之后其内容不会改变

而我们所使用的线性神经网络恰恰是位置敏感的，在有限的参数量和训练数据当中难以有效地处理这两种特性。

在1998年时Yann LeCun首先做出了第一个深度卷积神经网络，并在MNIST任务上取得了极好的性能表现。

== 卷积层的构造

#tcslide(
  [
  在数学上有一种运算方式可以很好的契合这些特性，这就是卷积运算 (Convolution)。
  一般在连续情形下卷积运算的定义是对于两个函数$f,g: RR^d -> RR$，它们之间的卷积就是一个函数

  $ (f * g)(vb(x)) = integral f(vb(z)) g(vb(x) - vb(z)) dif vb(z) $
  通常来说计算机当中显示的图像都不是一个简单的二维矩阵，而是一个三阶张量的形式。除去后两个维度分别代表长宽之外，第一个维度通常是RGB或RGBA等不同的色彩通道。
],
  [
  #set align(left)

  在我们神经网络当中则基本上都使用离散的卷积运算，处理各个不同位置上像素之间的对应关系。我们可以直观地想象这种卷积运算，它实际上是原本输入每一个位置的局部信息经过卷积核的加权求和，如下图所示。

  #align(center, image("./media/convolution.svg"))

  使用卷积运算的深度神经网络就被称为卷积神经网络 (Convolutional Neural Network, CNN)。
]
)

== CNN的实现

#tcslide([
  卷积层同样具有线性性质，因此需要使用ReLU等激活函数来防止坍缩。与此同时CNN也常常搭配各种池化层以及线性层来调整输出张量的形状。

  可以看到在训练后CNN在图像分类上的性能表现明显强出很多 (参见"cnn_mnist.py")，而2012年震惊世界的AlexNet本质上也是一个多层的CNN。(总参数量421,642)

  ```md
  Epoch [1/5], 准确率: 90.59%
  Epoch [2/5], 准确率: 93.79%
  Epoch [3/5], 准确率: 95.26%
  Epoch [4/5], 准确率: 95.62%
  Epoch [5/5], 准确率: 96.72%
  ```
], [
  ```py
  model = nn.Sequential(
    # in: (28, 28)，out: (32, 28, 28)
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),  # 14x14
    # in: (32, 14, 14)，out: (64, 14, 14)
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),  # 7x7
    # in: (64, 7, 7)，out: (64 * 7 * 7)
    nn.Flatten(),
    nn.Linear(64*7*7, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
  )
  ```
])

== CNN的持续发展

#tcslide([
  从2012年到2014年三届ImageNet竞赛当中获胜的模型基本上都采用了这类CNN模型结构。其中比较有代表性的进展有
  
  - AlexNet: 第一个使用GPU训练的CNN，效果拔群
  - VGG: 使用单纯重复简单的3x3卷积层取得效果，并在简单的模型结构下观察到了扩展潜力
  - Inception: 结合使用多尺度卷积分支提取信息，取得了更好的性能表现
], box([
  #image("./media/vgg.svg", height: 70%)
  #image("./media/inception.svg", height: 20%)
]))

== 残差神经网络的背景

#tcslide([
  时间来到2015年，这时人们发现了一种灵异现象，随着VGG类卷积神经网络深度以及参数量的扩大，模型的性能表现居然最终呈现出了下降的趋势，即使训练集上也同样如此。

  此时最关键的突破来自于2016年何凯明的残差神经网络 (Residual Network)，它通过创新性的残差连接结构基本确保了随着模型结构的扩展神经网络的性能总可以持续提升下去。时至今日几乎所有的深度神经网络都会采用残差连接这一结构。ResNet也成为整个21世纪引用量最高的一篇论文。
], [
  #image("./media/resnet_bkg.png")
  #image("./media/residual-block.svg")
])

== 残差连接 - 简单的直觉

#tcslide([
  ResNet最初的灵感来自于函数系，如 @resnet 所示。如果我们把一个参数待定的神经网络看作能够覆盖一个函数系，那么如果能够确保扩大模型尺寸以及添加更多层参数之后其能够表达的函数系范围完整覆盖之前的一系列小尺寸模型，那么完整训练之后就能够实现模型性能的稳定提高了。

  残差连接的设计就是为了实现这一点。它通过引入额外的路径，将神经网络某些层计算的输入张量直接加到输出张量上，由此就可以形成一个个残差块。
], [
  #set align(left)

  #figure(
    image("./media/functionclasses.svg"),
    caption: [ResNet灵感示意图，通过残差连接可以确保模型的表达范围稳定扩大],
    placement: auto
  )<resnet>

  这就相当于输入张量跳过了神经网络的很多层计算而直接进入后续的计算流程，由此就可以确保使用各种残差块搭建起来的ResNet必定可以覆盖任何一个更小尺寸模型的表达能力。
])

// == 真的是这样吗？
//
// #tcslide([
//   其实后续的很多研究表明Residual Connection发挥作用的原理并非维持了模型表达范围一致扩大，而是在于数值稳定性等原因上。
// ], [])

== ResNet实验表现

#tcslide([
  ResNet在2015年推出后几乎结束了ImageNet的比赛，其以3.57%的错误率直接超越了人类的平均水平，比前一年的冠军GoogLeNet近乎减半。后续其他人引入残差连接之后也都复现出了相当可观的性能进步，为我们今天的神经网络模型结构奠定了基础。

  也正因为ImageNet分类任务的性能接近上限，这一赛事也最终于2017年正式结束，完成了其历史使命。
], [
  #set align(left)

  模型结构可以参考代码"resnet.py"，这里实现了一个18层的ResNet来实现MNIST分类任务，可以看到它的性能表现相当之好。

  实际上此处的ResNet的能力已经足够胜任相当复杂的任务了，在MNIST数据集上多少有些大材小用，最终已经存在一些过拟合的现象。(总参数量701,178)

  ```md
  Epoch [1/5], 准确率: 96.47%
  Epoch [2/5], 准确率: 98.54%
  Epoch [3/5], 准确率: 98.82%
  Epoch [4/5], 准确率: 98.88%
  Epoch [5/5], 准确率: 98.73%
  ```
])

== CNN的后续应用

#tcslide([
  卷积神经网络在计算机视觉领域有着非常广泛的应用，可以应用于图像分类、目标检测、图像分割等等经典任务上面。近期发布的DeepSeek-OCR也应用卷积提取特征实现了效率的显著提高。

  #image("./media/ldm.png")
], [
  #set align(left)
  #image("./media/unet.jpg")

  2022年时广泛流行的Stable Diffusion模型的主体部分同样是一个大型的U-Net卷积神经网络。
])

#focus-slide([敬请期待下集: 模型结构-下

下集当中会主要讲解NLP领域最常使用的RNN与Transformer架构，也会涉及到ViT以及许多后续改良])
