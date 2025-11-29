#import "@preview/may:0.1.1": *
#show: may-pre.with(
  config-info(
    title: [视觉生成模型],
    subtitle: [AI4Math讨论班第二季-08],
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

#image-slide(img: image("./media/s2e8.png"))

#title-slide()

= 新闻环节

== 从Stable Diffusion到Flux

#tcslide([
  Stable Diffusion 1.x系列发布于2022年8月，成为了AI绘画模型流行的起点。它基本上延续了此前Latent Diffusion的技术路线并进行了大量的尺度扩展，其出图效果良好且完全开源，掀起了民间微调训练的热潮。

  后续2022年底到2023年SD系列2.x和xl系列陆续发布，直到2024年SD3/3.5发布之后正式切换为Diffusion Transformer架构，Stability AI也在不久之后倒闭。

  后续的Flux系列基本由SD原班人马打造，它们使用了新的flow matching方案，并在此后持续受到开源社区的欢迎。
], image("./media/diffusion_timeline.png"))

== Civitai

#tcslide([
  Civitai成立于2022年11月，目前已经成长为了全球最大的开源AI绘画模型社区。它起初的定位是一个模型仓库，允许用户上传或下载各种模型或者训练lora。不过后续Civitai在商业化发展当中仍然存在困难，并且其开放的运营模式也引发了许多版权与法律问题。

  早期AI绘画模型大多在单个消费级显卡上就可以完成推理和训练任务，但最新的Flux.2系列模型参数量也已达到了32B，个人想要完成训练微调已经相当困难。而与此同时它自身的性能表现也已经落后闭源模型许多了。
], figure(image("./media/civitai.png"), caption: [Civitai模型分享页面]))

== Nano Banana Pro

#tcslide([
  Google nano banana pro发布于2025年11月20日，它在图像生成的综合能力上显著更强，甚至几乎能够通过明日方舟测试。
  #align(center, image("./media/sikadi_nano_banana.png", height: 70%))
], figure(image("./media/nano_banana_challenge.png"), caption: [摘录自#link("https://mp.weixin.qq.com/s/hPJ7UljlknSToMGI1Z70zQ", "这篇报道")]))

= 早期的图像生成算法

== 三种经典的图像生成方案

在2012年深度学习革命之后，整个人工智能领域都迎来了一轮全新的研究热潮。其中一部分人逐渐开始进行生成式模型的探索，随后很快出现了各种极具创新性的经典工作。其中就包括作为现代图像生成模型基石的#link("https://arxiv.org/abs/1312.6114#:~:text=,datasets%20with%20continuous%20latent", "(变分自编码器, VAE, 2013)"), #link("https://arxiv.org/abs/1406.2661", "(对抗生成模型, GAN, 2014)")和#link("https://arxiv.org/abs/1503.03585", "(扩散模型, Diffusion Model, 2015)")。事实上在三者刚刚发布时，它们均不能取得很好的性能表现。如 @early_gmodels 就是它们原论文当中的的生成效果图：

可以看到在早期实验当中，各种生成式模型实际上都在差不多时间提出，并且早期效果均不算很突出。尽管如此，它们已然为我们揭示出了一条未来可期的路线，而它们的思想与影响力也一直延续至今。事实上从2022年的Latent Diffusion Model到其继任者Stable Diffusion在算法上都可以视作三者的结合。

#figure(
  grid(
    columns: 3,
    align: center,
    image("./media/mnist_vae.png"),
    [#image("./media/mnist_gan.png", height: 30%)
    #image("./media/cifar10_gan.png", height: 30%)],
    image("./media/cifar10_diffusion.png")
  ),
  caption: [从左至右依次是VAE，GAN和Diffusion的早期生成效果，测试当中使用的数据集分别是MNIST和cifar10。]
) <early_gmodels>

== 变分自编码器

经典的自编码器 (Auto Encoder, AE) 由两段卷积神经网络构成，它的encoder部分在对输入图像进行计算的同时会不断缩减中间的激活值尺寸，第二部分decoder则会不断地扩增激活值的尺寸，直到最终输出时与原图像一致。为实现这一目标它们直接使用的损失函数就是 $||vb(x) - hat(vb(x))||_2$。由于计算过程中信息存在压缩，因此这一过程必然是非平凡的。

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  [
  但是如此训练得到的自编码器仅能够实现压缩数据，而无法采样生成新的数据。VAE (Variational AE) 对此做出的改进是引入了一些新的限制，要求编码器部分输出的中间激活值分布尽量接近正态分布。为此它在单纯的重构损失函数之外额外加入了一项其与标准正态分布之间的KL散度项。此处的「变分」一词就来源于统计学当中的变分推断。
],
  figure(
  image("./media/vae_mechanism.jpg"),
  caption: [VAE的工作原理示意图]
  )
)

#pagebreak()

VAE在单纯的重构损失函数之外额外加入了一项其与标准正态分布之间的KL散度项。最终得到的训练目标就是

$ cal(J) = EE_(q_phi (vb(z) | vb(x))) [log p_theta (vb(x) | vb(z))] - D_"KL" (q_phi (vb(z) | vb(x)) || n(vb(z))) $

公式当中$vb(x)$是模型输入，$vb(z)$是中间激活值，$phi, theta$分别是encoder和decoder的参数，$n$表示标准正态分布。其中第一项重构损失可以证明和二范数作用相同。由此训练之后我们就可以直接从标准正态分布当中随机采样一个数据点，然后直接输入给decoder部分实现图像生成了。

这一损失函数也被称为证据下界 (Evidence Lower Bound, ELBO)。在实践操作当中多数情况下最大化对数似然 $EE [log p_theta (vb(x) | vb(z))]$ 其实就等价于最小化均方误差 $EE ||vb(x) - hat(vb(x))||_2$。

尽管能够取得一定的效果，VAE直接使用的图像生成质量始终较差。其在随机采样的输入的情况下生成图像往往相当模糊，缺乏更明确的「语义」信息。

== 向量量化变分自编码器

#tcslide([
  Google Deepmind提出了VAE的新改进版本，称为#link("https://arxiv.org/abs/1711.00937", "向量量化变分自编码器") (Vector Quantization VAE, VQ-VAE)。VQ-VAE在encoder和decoder中间额外引入了一个量化层。它会首先随机初始化一个有限的向量列表作为codebook，然后在推理过程当中会将encoder输入的每一个向量都约化到codebook当中距离最近的向量。(codebook向量也会随着学习进程而更新)

  如此一来VQ-VAE当中的隐空间内容就会受到更多的限制，但实验当中VQ-VAE却可以取得比经典VAE显著更好的效果。
],
  align(horizon, figure(
  image("./media/vq_vae.png"),
  caption: [VQ-VAE工作原理的示意图。人们通常认为离散的隐变量更有助于学习连续模态当中隐藏的离散结构。]
)))

== 对抗生成网络

对抗生成模型 (Generative Adversarial Networks, GAN) 是第一个明星生成式模型，并且在图像领域取得了相当令人振奋的结果。GAN的训练过程同样由两个神经网络构成，一个生成器 (Generator) 网络和一个判别器 (Discriminator) 网络。生成器直接接收一个随机生成的噪声图像，并尝试从中合成出以假乱真的图像。判别器则始终尝试将真实图像和生成器合成出的图像加以区分。两者始终进行着对抗训练，这也是GAN名称的由来。最终经过充分训练的生成器网络就可以直接用于图像生成任务。

GAN训练过程当中所使用的分类器相当于可以为模型提供一个智能的损失函数，引导模型去匹配数据的高阶统计而非像素级误差，以此确保图像语义与质量。

#figure(
  image("./media/ok_gan.svg"),
  caption: [GAN模型工作原理示意图]
)

#pagebreak()

#figure(
  image("./media/G_architecture.jpg"),
  caption: [现代的#link("https://mingukkang.github.io/GigaGAN/", "GigaGAN") (CVPR 2023) 生成器模型结构示意图]
)

#pagebreak()

#tcslide([
  GAN自身也面临着很多新的挑战。其主要问题就来自于其对抗训练所带来的不稳定性。一方面生成器与判别器训练目标对立，就需要非常精心的设计与微调训练流程和各种超参数，否则很容易出现崩坏，训练震荡或者无法收敛等情况。

  GAN面临的另一项挑战是*模式崩溃* (mode collapse)，这一现象主要来自于判别器仅判断真伪，使得生成器倾向于仅生成少数几种类别的图像，生成多样性就可能会出现问题。也就是针对这一现象当时出现了针对生成式模型的通用评估指标#link("https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance", "Frechet Inception Distance (FID)")

], figure(image("./media/D_architecture.jpg"), caption: [现代的#link("https://mingukkang.github.io/GigaGAN/", "GigaGAN") (CVPR 2023) 鉴别器模型结构示意图]))

= Diffusion与后续工作

== 最初的Diffusion Model

#tcslide([
  Diffusion式的生成模型在2015年提出之后就进入了长期沉寂，因为它的生成效果并不突出的同时生成速度极慢。一直到2020年#link("https://arxiv.org/abs/2006.11239", "DDPM")这篇论文发布之后才重新回到人们的视野。Diffusion Model有效解决了GAN当中常见的模式坍缩问题，生成质量和多样性显著提高。

  Diffusion Model的基本思想来自于物理上的粒子扩散运动，即原本有序的结构在经过长时间的随机运动之后逐渐变得混乱，而我们也许可以通过神经网络逐步反推重建，由此得到新的初始值。
], figure(
    image("./media/diffusion_process.jpg"),
    caption: [扩散过程的图示与相关公式，$w_t$为布朗运动，$g(t)$是时间相关的噪声强度函数]
  ))

== Denoising Diffusion Probabilistic Model

应用到图像生成上，前向扩散过程就相当于不断向图像当中添加噪声，反向过程则使用神经网络不断去除噪声实现生成。也因此diffusion model的全名实际上是*降噪扩散概率模型* (Denoising Diffusion Probabilistic Model, DDPM)。此时在具体使用上DDPM就产生了图像预测 ($x$-prediction) 和噪声预测 ($epsilon$-prediction) 两种路线。DDPM实验当中发现噪声预测的效果最好，因此在实验当中的前向和后向过程就是：

$
vb(x)_(t+1) &= sqrt(macron(alpha)_t) vb(x)_0 + sqrt(1 - macron(alpha)_t) vb(epsilon) quad &"Forward"\
vb(x)_(t-1) &= 1/sqrt(1 - beta_t) (vb(x)_t - (beta_t)/sqrt(1 - macron(alpha)_t) vb(epsilon)_theta (vb(x)_t, t)) + sqrt(beta_t) vb(epsilon) quad &"Backward"
$

在上面的公式当中，$vb(x)_t$是第t时刻扩散的数据点，$vb(epsilon)$是随机生成的噪声项，$vb(epsilon)_theta$就是我们神经网络所拟合的噪声项，$beta_t$表示t时刻的噪声强度，是一系列预先指定的超参数，$macron(alpha)_t = product_(i=0)^t (1 - beta_t)$。训练过程当中只需要调整$theta$使得$epsilon_theta (vb(x)_t, t)$与$epsilon$之差的二范数尽可能小即可。(对应ELBO的正态假设)

== Denoising Diffusion Implicit Model

到此时，diffusion model仍然没有解决其最关键的问题之一，也就是生成速度。相比较GAN的单步生成，ddpm采样的1000步生成就需要1000倍的时间消耗，这在很多时候都是无法接受的。随后很快学术界就出现了许多针对diffusion model的加速采样算法，实际上证明了ddpm采样算法当中所使用到的生成步数存在相当多的冗余

*#link("https://arxiv.org/abs/2010.02502", "DDIM算法")*就是其中一个经典的例子，它不再严格按照前向过程的逆过程进行推理而是转而使用确定性的方法，由此可以减少推理步骤数到50步至100步左右，实现10-20倍的加速。

DDIM所采用的推理迭代过程如下

$
vb(x)_(t-1) &= sqrt(macron(alpha)_(t-1)) hat(vb(x))_0 + sqrt(1 - macron(alpha)_(t-1)) vb(epsilon)_theta (vb(x)_t, t) quad "where"
hat(vb(x))_0 = (vb(x)_t - sqrt(1 - macron(alpha)_t) vb(epsilon)_theta (x_t, t))/sqrt(macron(alpha)_t)
$

DDIM另外也提出了将扩散过程转换为ODE进行处理的思想。

== DPM-Solver

DDIM算法实质上是使用Euler法来近似求解扩散过程的ODE，而沿着这一思路，2022年提出的#link("https://arxiv.org/abs/2206.00927", "DPM-Solver")就进一步将数值分析当中的Runge-Kutta应用到了扩散模型上，并再次实现了采样速度的显著进步。它只需简单的10-20次迭代即可生成出高质量的图像。在此处DPM-Solver实质上是将扩散模型当作了预测连续扩散方向的黑盒函数。

在连续情形下扩散过程对应的概率流ODE就是

$
dif vb(x) = [vb(f)(t) vb(x) - 1/2 g^2 (t) gradient_vb(x) log p_t (vb(x))] dif t
$

此处的 $gradient_vb(x) log p_t (vb(x))$ 也被称为score function，在正态假设下它在每一点处的数值实际上就等于预测的噪声值加上一点缩放。而对于上述这种半线性结构的ODE而言，直接应用Runge-Kutta法就可以非常高效地完成求解。

== Latent Diffusion Model

#tcslide([
  同样在2022年，为了解决diffusion model生成高清图像上的挑战，#link("https://arxiv.org/abs/2112.10752", "Latent Diffusion Model") (LDM) 应运而生。

  LDM在扩散训练之前首先训练了一个强大的自编码器 (通常是VQ-GAN或KL-VAE等变体)，通过将图像压缩到潜空间后再进行扩散过程，极大地提升了生成图像的质量和效率。

  它成为了后来Stable Diffusion及Flux系列的基础，配合上DPM-Solver这样的高效采样算法，直接引导了一次文生图模型的浪潮。
], [
  #align(horizon, figure(image("./media/ldm.png"), caption: [Latent Diffusion Model的模型结构图]))
])

== Flow Matching

#tcslide([
  流匹配 (Flow Matching) 模型是对经典Diffusion过程的一次大幅简化，并以更高的效率和生成质量成为了新的主流范式。

  它直接放弃了扩散过程的物理背景，将图像到噪声的演变路径直接拉直成直线，因此其前向和后向过程都变得很简单：

  $
  vb(x)_t = (1 - t) vb(x)_0 + t vb(epsilon)
  $

  而模型直接预测的就是 $vb(epsilon)$。由于此时ODE当中的速度值已经被基本拉成直线，只需要简单的Euler法就可以实现极为高效且高质量的采样了，4-8步迭代足矣。
], align(horizon, figure(image("./media/flux_samples.webp"), caption: [SD3和Flux系列都采用flow matching方案进行训练。])))

== Just Image Transformer

#tcslide([
  这是最近2025年11月17日何凯明的一篇有趣的#link("https://arxiv.org/pdf/2511.13720", "新论文")，可能是对此前各种扩散模型研究的拨乱反正。

  此前的很多工作都关注使用扩散模型进行噪声预测，因为它直接符合正态假设，优化目标明确。但是正态噪声分布于整个高维空间当中，将它作为拟合目标反而可能会造成困难。相比之下经典的图像流形假设就认为自然界中合理的图像都可以认为处在一个很小的子流形上，以它作为预测目标就可能实现更好的效果。JIT就首先实践并肯定了这一点。

  // 它甚至放弃了使用VAE的标准做法，直接在像素空间扩散但效果同样好。
], align(horizon, figure(image("./media/image_manifold_assumption.webp"), caption: [图像流形假设示意图])))

== GPT-image后的黑暗时代

#tcslide([
  2025年3月25日发布的GPT Image Generation功能可以认为是经典图像生成领域的一个分水岭，它展示了极为强大的精确语义理解、文字渲染能力以及惊人的人物和风格一致性。在此后只有谷歌的Nano Banana系列能够抗衡。

  但是此时OpenAI与谷歌均没有公开实现方案当中的任何细节，相关的技术研究和开源社区似乎仍然处在黑暗时代 (至今仍然不知道如何实现)。

  目前在强语义理解与可控生成上有几个探索方向，但都还没有产生出足够强的产品。
], [
  #set align(left)
  - 原生多模态统一：DeepSeek Janus，Show-o，Seed-X；相信多模态统一将产生质变，只是尚未实现足够强的进步。
  - 增强扩散模型：Flux系列，Qwen-Image等；原本文生图模型已有很强的指令遵循能力，仍有量变提升空间。
  #image("./media/janus_samples.png")
])

#focus-slide([
  #set page(margin: 1.4em)
  至此我们讨论班的AI基础部分就已结束。在下周的第九次讨论班当中我们就会开始讲解自动定理证明的相关内容，包括形式化数学与自然语言证明两种方案。之后我们也会针对开展研讨会。
])
