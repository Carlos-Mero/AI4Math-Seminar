#import "@preview/may:0.1.1": *
#show: may-pre.with(
  config-info(
    title: [模型结构-下],
    subtitle: [AI4Math讨论班第二季-04],
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

#image-slide(img: image("./media/s2e4.png"))

#title-slide()

= 前言

== 图像分类 - 但是大语言模型

#tcslide(align(center, image("./media/gpt_5_imgnet_classification.png")), image("./media/gpt_5_kaihe.png"))

== 大一统的道路

#tcslide([
  #image("./media/qwen3vl_obj_detection.png")

  如此这一切能力的实现都依赖于如今的多模态大语言模型技术 (Multimodal LLM, MLLM)，而它背后的模型结构无一例外都是各种*Transformer*或其变体。
], [

  #set align(left)

  不过在如今，绝大多数的MLLM都仅支持图像等多模态输入的能力，而在多模态输出层面上则始终存在一定的困难。输入层面的通常做法是额外训练一个Encoder将各种模态的输入统一映射到语义空间。

  #align(center, image("./media/qwen3vl_arc.jpg"))
])

== VAR与多模态生成的探索

#tcslide([
  2024年的CVPR best paper，VAR (Vision AutoRegressive model) 这篇论文则首次使用Transformer架构加上类GPT的自回归生成模式实现了图像生成任务，并且取得了远超同期diffusion model的效率和性能表现。

  这为CV和NLP的最终统一提供了一条可能路线，但真正的未来还有待探索。

  #image("./media/var.png")
], [
  #set align(left)

  #align(center, image("./media/ghibli.png"))

  于2025年3月25日发布的GPT image generation功能则将视觉生成也整合进入了ChatGPT当中，并且实现了图像生成质量与可控性的显著提升。

  在此之后我们也见到了Bagel, gemini nano banana等类似竞品诞生。
])

#focus-slide([而我们今天的主题就是

Transformer模型架构])

= 语言模型

== 序列建模与语言模型

#tcslide([
  与我们此前所介绍的图像识别等模型相比，处理语言任务时会存在两个关键的不同点

  - 语言符号*离散*，无法直接由神经网络处理
  - 语言文本是*变长*的序列

  在这其中，语言的离散性目前主要依靠tokenizer以及token embedding处理。而在这之后则通常会采用循环神经网络 (Recurrent Neural Network, RNN) 或者Transformer这样针对设计的神经网络进行处理。

  广义上针对序列建模的模型都可以称为语言模型。
], align(horizon, figure(caption: [AI4Science当中就常常能看到Protein Language Model这一概念], image("./media/protein_lm.png"))))

== 序列数据的一般处理

#tcslide([
  针对序列数据的一般处理方式就是将其拆分成数个单独、大小相同的基本单元进行处理。如此一来我们就只需要处理定长的数据即可。

  而作为序列数据而言，另一项最关键的目标则是要处理各个单元之间的相关性。对于文本而言，一个基本处理单元就是一个token。

  当前的AI天气预报通常同样会采用自回归方式进行，如右图所示。
], align(horizon, image("./media/whether_report.webp")))

== BPE Tokenization

#tcslide([
  当前GPT类的大模型最常使用的Tokenizer方案是Byte Pair Encoding (BPE) 算法，它直接基于统计给出可能的token词表。

  - BPE首先会初始化所有可能的Byte作为初始的token列表
  - 统计给定的文本数据集当中出现频率最高的token组合
  - 合并频率最高的token构造一个新的token添加到列表当中
  - 直到词表token数量达到预设值

  这样的token切分算法并不保证语义合理性，但是能够较好地确保工作效率。
], align(horizon, figure(image("./media/xc_tokenize.png"), caption: [BPE算法的演示])))

= 循环神经网络

== 循环神经网络结构

#tcslide([
  循环神经网络最早于2011年左右提出，它的大致实现方式就是通过线性层逐一处理每个token，并将中间信息以一个数值张量 (也即隐藏层$vb(H)_t$) 的形式传递到下一次计算当中。

  在右侧公式当中，$vb(X)_t$表示对应位置输出的词向量，$vb(H)_t$就是隐藏层数据，$W_(x h), W_(h h), W_(h q)$分别是三个参数矩阵，$vb(b)_h, vb(b)_q$是两个偏置项。

  我们通过训练使得输出$vb(O)_t$近似拟合$t+1$时刻的token出现的概率，由此即可实现自回归生成。(当然实际上也存在很多其他用法)
], [
  #set align(left + horizon)

  RNN的两个主要计算公式:

  $
  vb(H)_t &= phi(vb(X)_t vb(W)_(x h) + vb(H)_(t-1) vb(W)_(h h) + vb(b)_h)\
  vb(O)_t &= vb(H)_t vb(W)_(h q) + vb(b)_q
  $

  #figure(image("./media/rnn.svg"), caption: [经典RNN计算流程示意图])
])

== 循环神经网络的实现

#tcslide([
  我们在示例代码"rnn_demo.py"当中就实现了一个最简单的RNN模型，它使用的是直接的单层RNN作为语言模型，并且使用了最简单的字符级tokenizer (并没有使用BPE算法，而是直接将单个字符作为token)。

  其模型结构在PyTorch当中的主要实现方式如右所示，这里的"input_size"所指的就是词表大小。我们随后就在莎士比亚全集上就训练这个模型来实现文本生成任务。
], [
  #set align(horizon + left)
  ```py
  # class CharRNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
      super(CharRNN, self).__init__()
      self.hidden_size = hidden_size
      self.embedding = nn.Embedding(input_size, hidden_size)
      self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
      self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x, hidden):
      x = self.embedding(x)
      out, hidden = self.rnn(x, hidden)
      out = self.fc(out)
      return out, hidden
  ```
])

== 简单RNN的成效

#tcslide([
  此处我们每次生成时均使用相同的前缀：

  "ROMEO:"

  并要求我们的语言模型在它之后补全生成100个token (字符)。可以看到随着训练的进行，生成文本的质量有了显著提高 (虽然还是在胡说八道x)。

  在未经训练时RNN的输出基本为纯乱码，训练一轮后似乎开始有了英文的感觉，到20轮后RNN就已经可以学会很多词汇的正确拼写了。

  // (补充一句，现代LLM基本都采用一个特殊token <|eos|>来控制输出，预测出这个token即代表输出终止)
], [
  #set align(left + horizon)

  *Epoch 0*:

  e Isz.raJEAqRhRFW;H-ni!Vahgytpnha,KKUhtGh3kjb3 f.?LcU'3dqCthF
  fbym.
  ?jjgs.YNrr$.$
  Z?&!U.-HtK-MURLZBT

  *Epoch 1*:

  Sill, we heart, of but to excold is him. Lided and for
  Mare do be shang againg Sor the prist, and to

  *Epoch 20*:

  If he twenty more man, I crown:
  Say no hation, therefore to the good some mayst that tears the death
])

== RNN的改进型 - LSTM与GRU

#image("./media/xc_lstm_gru.png")

== Attention机制的起源

#tcslide([
  Attention机制最早提出于2014年，它来源于早期NLP的核心领域 - 机器翻译。

  经典的RNN机器翻译模型使用一种seq2seq的工作模式，将输入的文本序列经过RNN编码与解码后翻译为另一种语言的文本序列。

  而在直觉上，当我们在执行翻译任务时，实际上我们总需要选择性关注一小部分相关的token，但RNN却始终平等地处理每一个token。这或许有很大的优化空间。
], [
  #set align(center)
  #image("./media/seq2seq-details.svg")
  #image("./media/seq2seq-details-attention.svg")
  RNN seq2seq模型与attention机制
])

== Attention输出的计算方式

#tcslide([
  但是我们怎么知道输出每一个token时应当关注哪一个token呢？没关系，因为在AI的世界里，一切都是可以学习或拟合的。

  这里我们以目前最常用的dot-product attention为例，它的注意力计算公式就是:

  $ "Attention"(Q, K, V) = "softmax"((Q K^TT)/sqrt(d_k)) V $

  在其中$Q in RR^(n times d_k), K in RR^(n times d_k), V in RR^(n times d_v)$，其中$n$是输入序列的长度，$d_k, d_v$是两个超参数。$Q, K, V$三个矩阵都由输入的向量序列经过线性映射得到。
], [
  #image("./media/attention-output.svg")

  #set align(left + horizon)
  这里的$Q, K, V$分别代表query, key, value三个量，它们都来自数据库术语，人们输入query进行查询，通过与key进行比对后获取对应的value。
])

= Transformer的时代

== Self-Attention

#tcslide([
  在约2017年时，人们开始在很多任务上探索一种自注意力 (self attention) 机制。如果将序列当中的每一个token都输入给原序列做attention计算的话，我们得到的输出就会是另一个相同长度的序列。在经过适当的训练之后，它就能完成很多我们期望的信息处理任务。

  更进一步地，我们也可以并行地使用多套参数计算self-attention，并将得到的结果叠加在一起，这就形成了多头注意力 (multi-head attention)。使用self-attention生成更好的序列表示之后，就可以更好地服务于RNN去做生成任务了。
], [
  #set align(horizon)
  #figure(image("./media/cnn-rnn-self-attention.svg"), caption: [CNN, RNN和self-attention三种序列计算模式的对比])
])

== Attention is All You Need!

#tcslide([
  既然单独使用Attention机制即可以处理序列数据，那么是否有可能抛弃RNN而只使用Self-Attention单飞呢？

  于2017年发布的重量级论文#link("https://arxiv.org/abs/1706.03762", [_Attention is All You Need_])就做出了这样的尝试，并且取得了相对RNN显著更优的表现，最终成为AI领域引用量第二高的论文。删去RNN之后纯attention堆叠得到的模型结构就是Transformer，其优势具体体现在：

  - 自然捕获全局依赖关系
  - 能够大规模并行计算
  - 实际展开深度显著更低，易于训练
], image("./media/transformer.png"))

== Encoder-Decoder与因果掩码

#tcslide([
  最早的Transformer同样应用于机器翻译任务，并采用Encoder-Decoder架构实现。后续它的两个影响力最大的衍生分支，BERT和GPT则分别仅使用了它的Encoder和Decoder部分。

  Transformer的Encoder和Decoder两块结构最主要的区别就在于Attention计算时是否具有因果掩码 (Causal Mask)。

  因果掩码通过固定删除attention的上三角部分，确保了每一个token在计算时无法利用到后续序列的信息，如此才能正确执行自回归生成。
], [
  #figure(image("./media/causal_mask.png"), caption: [Transformer Decoder部分计算示意图，参考自链接#link("https://poloclub.github.io/transformer-explainer/", "Transformer Explainer")，非常推荐阅读。])
])

#image-slide(img: image("./media/transformer_overview.png"))

== Transformer相关实验

#tcslide([
  我们也在代码示例“little_lm_demo.py”当中实现了一个具有三个attention block的decoder-only语言模型，并同样使用最小的字符级tokenizer，并在莎士比亚全集上进行训练。最终得到的效果如下页PPT所示。

  如果希望语言模型发挥更好的效果，我们仍需要使用更多的训练数据以及更完善的训练流程。
], [
  #figure(image("./media/MiniMind-structure.png"), caption: [#link("https://github.com/jingyaogong/minimind", "MiniMind")模型结构示意图。MiniMind是一个完整且现代化的小型GPT类语言模型的开源实现，如有兴趣非常推荐拿来玩一玩！])
])

== little_lm_demo生成效果

#tcslide([
  *Epoch 0*:

  HUFwwnBDk wICnw'XrH UYxyo3RwRF. PTVIzOPVL?sCg'Q Wf&NnwQBdh!:RF-nN:sgoqedX,\$LIGsXLbBOw'oAgkk,ghurtEcf?c:fYejfN\$wq.Qe-3dV-xn?'DcVg?GwC!TQ\$!P&qM!F3bwwI?A:bAI&qMjXsFlxY\$vxLbalOPSRtL3ryY!BDleq:IIawFFG?szhX

  *Epoch 20*:

  IUSINHA:
  Will fow frre couck.

  VOLAUKE:
  Ry thou prom of ant us, ind wer ca moorge?
], [
  #set align(left + horizon)

  ORUKEN ETHARD
  TINRY RENT:
  Whours:
  Whe sply mus buresh, Rome your of maaicest.
  Firch murthants andeow the our ard, o

  *Epoch 50*:

  When thee I shalt, it prossived; in the
  the roubled hold.

  DUKE OF NEDYBEY:
  Nors and truck beauter all our not a ofence,
  I'll the are way this weath, when these now.

  Cleto it the is becouranter, you
])

== 对图像使用Attention吧 - Vision Transformer

#tcslide([
  注意力机制在图像相关的任务当中同样具有重要的意义，因此当Transformer架构在NLP领域取得成功之后，将它迁移到CV领域就显得理所当然了。

  于是在2020年，_“An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale”_这篇论文当中就首次提出的Vision Transformer (ViT)架构，它首先将图像切分为数个固定大小的小patch，此后再通过线性层或卷积层直接映射得到语义向量。之后就可以直接将图像当作token序列进行处理了。
], [
  #image("./media/vit.svg")
])

== 寻找二维的自然序 - VAR

#align(horizon, tcslide([
  由于在二维图像当中似乎并不存在自然的顺序关系，类似GPT的Transformer自回归生成范式就难以应用到图像生成领域。此前也存在使用扫描顺序逐块生成的工作，但是效果并不很好。

  直到2024年年中时，由北京大学一位硕士生所做的#link("https://arxiv.org/abs/2404.02905", [_Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction_]) (VAR) 首次提出了Scale顺序预测的概念，并应用GPT-2模型取得了极好的效果。
], [
  #figure(image("./media/var.png"), caption: [VAR主页配图，其中也对比了彼时存在的三种Transformer生成框架的工作原理，其中VAR在生成效率和质量上均显著更高。])
]))

== Transformer的继续优化

#tcslide([
  在Transformer架构取得显著成功的同时，其自身的缺点也逐渐凸显。当前Transformer架构明显的缺陷基本都在运行效率上:

  - 在训练或推理的prefill阶段，Transformer的计算复杂度与输入序列长度$n$之间呈平方关系 (每两个token之间需要计算一次相关性)
  - 在推理生成阶段，使用KV Cache可以保留前期token计算的结果，将复杂度削减到线性的同时却显著提高了显存压力

  这其中存在很多代表性的工程优化。
], [
  #set align(horizon)
  #figure(image("./media/GQA.png"), caption: [Grouped Query Attention (GQA) 技术，通过多头注意力共享KV值的方法显著削减了显存消耗，并保持性能较少下降。])
])

#pagebreak()

#tcslide([
  #figure(image("./media/dsmoe.png"), caption: [DeepSeek-V3的模型架构，MLA，MoE和FP8训练带来了成本的显著降低。])
], [
  #figure(image("./media/v3_2_cost_compare_en.png"), caption: [DSv3.2采用NSA后成本大幅下降])
  // 此处DSv3.2的实现方法是首先使用小型的lightning indexer根据query迅速扫过所有输入tokens，此后挑出少量相关性较强的tokens再做完全attention计算
  // 实际上很像一个内置的RAG
  #figure(image("./media/kimi_kda.png"), caption: [Kimi KDA线性注意力，类RNN层与full attention层混合同样有效])
])

#focus-slide([
  #set page(margin: 1em)
  敬请期待下集 - 模型训练

  接下来两次讨论班我们会主要讲解AI模型的训练流程，同时也会介绍模型结构当中的各种正则化层。大致介绍当今的AI模型完整的构建方法。
])
