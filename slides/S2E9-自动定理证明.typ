#import "@preview/may:0.1.1": *
#show: may-pre.with(
  config-info(
    title: [自动定理证明],
    subtitle: [AI4Math讨论班第二季-09],
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

#image-slide(img: image("./media/s2e9.png"))

#title-slide()

= 新闻环节

== DeepSeek-V3.2

#tcslide([
  #link("https://arxiv.org/abs/2512.02556", "DeepSeek-V3.2")于2025年12月1日正式发布，其中包含日常使用的V3.2和推理特化的V3.2-Speciale两个版本。两者已经能够达到GPT-5和Gemini 3 Pro级别的通用或推理能力，当然实际上仍稍弱一些。

  DeepSeek-V3.2系列的一大特色是DeepSeek Sparse Attention机制带来的成本大幅下降，同时整体参数仍是从DeepSeek-V3续训得到。能够在如此低成本且开源的情况下达到匹敌顶尖闭源模型的性能表现，只能说不愧是DeepSeek。
], image("./media/dsv3.2_influence.jpeg"))

== DeepSeek-V3.2的性能表现

#align(center, image("./media/v3.2_251201_benchmark.webp"))

== DeepSeek-V3.2技术细节

DeepSeek-V3.2最难能可贵的一点就是，作为工业级高成本的前沿大模型，它仍然愿意以技术报告的形式公开其训练过程的各种细节。其中信息弥足珍贵，非常值得一读。

它的训练过程当中最重要的技术方案包括

- DeepSeek Sparse Attention大幅优化了长上下文下的attention计算
- 在DeepSeek-V3.1的基础上继续训练，在有效控制成本的同时仍然取得了极强的性能
- 在后训练当中应用专家蒸馏 (Specialist Distillation) 的方式获得各种专业能力
- 多任务上的混合强化学习训练
  - 算法为大量改良后的GRPO
  - 应用了改良的无偏KL散度估计
  - 负例遮盖、一致路由和采样遮罩等一系列维持稳定性的方案
- 针对智能体任务的大规模数据合成
- 应用DeepSeekMath-V2的方案续训得到Speciale模型

== DeepSeekMath-V2

在DeepSeek-V3.2发布前几天的时间里，另一个重量级的工作，#link("https://arxiv.org/abs/2511.22570v1", "DeepSeekMath-V2")也正式发布了。其标题就是_Towards Self-Verifiable Mathematical Reasoning_，强调通过自我验证不断训练取得了极强的性能表现，是第一个开源的IMO金牌级别的推理模型。

其主要原理就是通过训练一个强大的Verifier来对自身的证明进行检验与评价，以此实现自身证明能力与严谨性的不断提高。

一个值得关注的演变是，自2025年年初的DeepSeek-R1之后，应用可验证奖励激发深度推理能力的思路就开始广泛流行，近期又回到了此前reward model的路线上去。但一切却都变得有所不同，我认为转变的关键就在于reward scaling上面。与此前RL算法轨迹探索占据主要成本不同，现今的DeepSeekMath-V2当中奖励验证部分才是成本的主导，这也与我们新近发布的论文#link("https://arxiv.org/abs/2511.21522", [_Pessimistic Verification for Open Ended Math Questions_])当中的观点高度相关。

#focus-slide([
  #set page(margin: 1.4em)
  自动定理证明曾是人工智能的核心课题，而在推理能力成为大模型主导方向的今天，自动定理证明也再一次成为了广受关注的热点话题。
])

= 形式化定理证明

== 形式化思想的起源

有一个比较反直觉的事情是，基于形式化方法构建的自动定理证明系统的历史要远长于如今使用自然语言处理定理证明的工作。

上世纪逻辑学和数学基础的研究一直在尝试给所有的数学理论建立一个更加严格的基础，其中对于逻辑/形式系统的研究包括三个核心问题：

- *一致性* (系统内部无矛盾)
- *完备性* (所有真命题均可以证明)
- *可判定性* (存在一套算法可以自动验证每一个命题的正确性)

在某些特定的情形下这三个性质都可以得到很好的满足 (今晚要开讨论班公理系统，Presburger算术，弱二阶逻辑等等)，但是实际上绝大多数情况下我们都无法很好地验证这三条性质。例如如果某一个形式系统当中可以构造出皮亚诺算术，那么它的完备性和一致性就都无法在内部证明。而图灵-丘奇定理则确定了具有一阶逻辑的系统必定是不可判定的。(往好处想是确定了数学家 / AI4Math永远有事情可做)

== 形式化自动定理证明

推荐参考阅读 #link("https://zhuanlan.zhihu.com/p/627072208", "深度学习时代的自动定理证明简介")

最早的自动定理证明程序可以被追溯到1954年，Martin Davis开发了第一个可以证明“两个偶数的和是偶数”的程序。后面自动定理证明的概念与人工智能一同提出，受到炒作以及遇冷。由于符号主义命题数量组合爆炸的现象存在，通过枚举实现的自动定理证明方法一直难以推广。(现今有很多形式化语言仍然在通过枚举等方案尝试实现局部的自动化证明)

事实上当今数学形式化工作与形式语言的设计更多来自于直觉，也就是所谓的:

/ Curry-Howard同构: 命题即类型，证明即程序；写数学证明和写代码是同一回事

直观想象，对于一个软件工程项目而言，我们期望的命题是「存在一个可以将LaTeX代码转换为pdf文档的程序」，那么它的证明方式就是「通过代码写出一个符合要求的程序，就能证明存在性」。对于数学定理而言，我们期望的是「构造一个从条件到结论的映射」，而它的实现方式是「应用定理组合形成一个符合需求的大定理」，或者说证明就是定理的一个实例。如果从公理化的角度考虑，软件工程使用的基础是计算机指令集，而数学定理证明最终则都来源于公理。

正是有这样同构性质的存在，各类形式化语言的设计就成为了可能。它们的基本思路都是使用计算机指令集实现符号计算以及模拟公理化的数学证明。第一个专门面向数学的形式化语言是1973年的Mizar，现今最常使用的则是#link("https://lean-lang.org/", [*Lean*]) (2013, Microsoft)，Coq，Isabelle这三者。形式化语言Lean实际上也是一个全功能的计算机语言，可以编写出任意可能的程序，不过在语法上针对数学任务进行了很多特化的设计。(例如"theorem"在Lean当中就是一个关键字)

此时，通过Lean编译器的类型检验(此时类型系统就是数学公理系统的一个「虚拟机」)，我们就可以得到一个最直接的应用，即可以使用它去直接一个形式化证明步骤的正确性。目前关于数学形式化的工作基本上都主要利用了这一项能力。与此同时我们可以给这个编译器添加一些额外的能力，例如输出证明的中间状态以及提供错误信息等等。这样一来我们就实现了一个弱化版的ATP (Automatic Theorem Proving, 自动定理证明)的工具，通常称为ITP (Interactive Theorem Proving, 交互式定理证明)。

如下是Mathlib4当中的一段使用Lean 4语言证明推广的Sylow第二定理的代码示例

```lean
/-- A generalization of **Sylow's second theorem**.
  If the number of Sylow `p`-subgroups is finite, then all Sylow `p`-subgroups are conjugate. -/
instance Sylow.isPretransitive_of_finite [hp : Fact p.Prime] [Finite (Sylow p G)] :
    IsPretransitive G (Sylow p G) :=
  ⟨fun P Q => by
    classical
      have H := fun {R : Sylow p G} {S : orbit G P} =>
        calc
          S ∈ fixedPoints R (orbit G P) ↔ S.1 ∈ fixedPoints R (Sylow p G) :=
            forall_congr' fun a => Subtype.ext_iff
          _ ↔ R.1 ≤ S := R.2.sylow_mem_fixedPoints_iff
          _ ↔ S.1.1 = R := ⟨fun h => R.3 S.1.2 h, ge_of_eq⟩
      suffices Set.Nonempty (fixedPoints Q (orbit G P)) by
        exact Exists.elim this fun R hR => by
          rw [← Sylow.ext (H.mp hR)]
          exact R.2
      apply Q.2.nonempty_fixed_point_of_prime_not_dvd_card
      refine fun h => hp.out.not_dvd_one (Nat.modEq_zero_iff_dvd.mp ?_)
      calc
        1 = Nat.card (fixedPoints P (orbit G P)) := ?_
        _ ≡ Nat.card (orbit G P) [MOD p] := (P.2.card_modEq_card_fixedPoints (orbit G P)).symm
        _ ≡ 0 [MOD p] := Nat.modEq_zero_iff_dvd.mpr h
      rw [← Nat.card_unique (α := ({⟨P, mem_orbit_self P⟩} : Set (orbit G P))), eq_comm]
      congr
      rw [Set.eq_singleton_iff_unique_mem]
      exact ⟨H.mpr rfl, fun R h => Subtype.ext (Sylow.ext (H.mp h))⟩⟩
```

== AlphaGeometry

Google DeepMind团队此前凭借AlphaGo和AlphaFold两项成果给世界带来了相当大的震撼。而当时间来到2024年1月17日，DeepMind再次发挥自己刁钻的选题能力以及强大执行力，再次贡献了AlphaGeometry (AG) 这项成果。

#grid(
  columns: 2,
  gutter: 0.4em,
  [
  它通过AI与形式化证明方法的结合，几乎直接征服了平面几何竞赛题。在近年来的30道IMO平面几何证明题当中，AlphaGeometry成功证明了25道，已经接近IMO金牌的平均水平。AlphaGeometry的模型和代码都完全开源。

  而今随着AlphaGeometry路线的发展，AlphaGeometry2等新系统都已经可以在平面几何上完杀人类选手了。
  ], align(center, image("./media/ag.jpeg", height: 70%))
)

#pagebreak()

平面几何当中涉及到的公理较少，且逻辑自洽，很少依赖外部知识。这使得DeepMind可以很容易地针对平面几何设计一套专用的形式化语言，并且可以通过遍历公理的方式实现逻辑推理。谷歌同时也设计了一套快速合成数据的方案来提供大量的形式化题目。

AlphaGeometry使用了一套neuro-symbolic方法，其中推理部分基本上依靠符号推理引擎完成，而神经网络部分着重负责完成创造性、构造性的任务。

#align(center, image("./media/agapproch.jpeg", height: 50%))

#pagebreak()

AG使用了一个相当小的语言模型作为其核心组件，其参数量仅仅是150M (运行占用内存/显存约300MB)，可以在个人电脑上轻松地运行。其唯一的作用就是在符号推理引擎推断出所有结论之后仍未得到最终结论时构造出一个新的辅助点 (辅助线)，直到搜寻到最终答案。

它通过枚举各种条件组合，随机构造辅助线/点，以及推理完毕删除的方法构造出了1亿道平面几何题作为训练数据，依靠这些来训练语言模型构造辅助点的能力。

到2024年7月25日之后，Google再次放出了AlphaGeometry 2，性能比起前代又有了大幅提升，已经可以在平面几何证明问题上超过所有人类的表现。在准确率更高的同时，解决一道IMO级别的平面几何问题却只需要十几秒的时间。

AlphaGeometry向我们展示了AI4Math，或者说自动定理证明在现代的可能性。尽管这些成果对于数学研究或者应用的实际意义仍然有限。

== AlphaGeometry合成数据示例

#align(center, image("./media/agdata.jpeg"))

== AlphaGeometry解题过程示例

#align(center, image("./media/agex.jpeg"))

== 通用的形式化证明器

构建一个普遍适用于多个数学领域的形式化证明器的难度比起AlphaGeometry而言难度要高许多，但同时也会有明显更大的现实意义。国内外的多个研究团队都有针对这个方向做出过努力，而目前主流的路线都是使用Lean 4作为形式化语言来处理证明问题，其中可能包括数个关键的子任务：

- 自动形式化：将自然语言书写的命题或者证明翻译到Lean等形式化语言的过程
- 证明生成：自动生成出形式化数学证明
- 证明搜索：利用形式化语言的精确验证和反馈构建搜索工作流，直到找到正确的证明

围绕着这些任务产生出了一系列代表性的工作，我们就挑一部分讲解。

== DeepSeek-Prover

DeepSeek很早就开始关注到了数学问题与形式化证明任务，很多评价也指出数学推理团队和基础架构团队就是DeepSeek的两大王牌。

#grid(
  columns: 2,
  gutter: 0.4em,
  [
  初代的DeepSeek-Prover发布于2024年5月23日，它通过在DeepSeekMath 7B这一小模型上继续训练以取得性能提升，具体流程包括迭代优化的自动形式化训练以及证明训练。

  整体看来初代DeepSeek-Prover在方法上新意不多，不过确实在miniF2F等测试基准上取得了不错的性能进步。

  DeepSeek Prover直接采用全证明生成的工作方案，这与人类相当不同。
],
  figure(image("./media/deepseek_prover_approach.png"), caption: [DeepSeek-Prover初代的训练方法概览])
)

== AlphaProof

Google DeepMind在2024年7月25日同时发布了新的AlphaProof系统和AG2，并且声称两者合作之后可以在IMO竞赛当中取得银牌高分水平，距离金牌仅差一分。不过这一次谷歌并没有开源任何代码和模型权重，公开的技术细节也较少。只知道它在训练当中使用了自动形式化方法构建Lean语言数据以及强化学习方法来训练证明器。

#align(center, image("./media/apformalization.jpeg", height: 60%))

== DeepSeek-Prover-V1.5 / V2

#tcslide([
  DeepSeek-Prover-V1.5随后发布于2024年8月15日，它在前代的基础上增加了训练量，增加了强化学习方案并且引入了错误修正与证明搜索的能力，由此取得了进一步的性能提升。

  DS-Prover-V2发布于2025年4月30日，它直接将模型参数增加到了671B，在DeepSeek-V3的基础上进行训练。在方法上则是进一步扩展了定理搜索以及与Lean编译器交互的能力。它最终在miniF2F上取得了88.9%的准确率 (但是pass\@8192)，与它同期的还有BFS-Prover和Kimina-Prover等类似的工作。
], figure(image("./media/dsp_v1.5_overview.png"), caption: [DeepSeek-Prover-V1.5概览]))

== Seed-Prover

字节跳动的Seed-Prover是今年形式化自动定理证明的一个大工程，期望取得AlphaProof级别的性能表现，资源投入相当之多。

#grid(
  columns: (1fr, 1fr),
  gutter: 0.4em,
  [
  Seed-Prover是第一个能够在形式化高中数学竞赛数据集miniF2F上取得100%准确率的形式化自动定理证明系统 (当然，并不是单步证明)。其最关键的技术就在于引理式证明与自我迭代验证工作流。这种思路其实并不难想到，但难点仍在于大规模的工程实现。

  Seed-Prover同样使用了自动形式化合成数据以及强化学习训练，不过截至撰稿时它们仍然没有开源模型或相应代码。
],
  figure(image("./media/miniF2F_trend.png", height: 60%), caption: [高中竞赛miniF2F的性能演进])
)

== Seed-Prover的工作流

#tcslide([
  Seed-Prover性能进步的另一大来源就是其针对引理式证明所设计的工作流。它们采用验证、迭代、自我修正的方式充分利用上了测试时算力扩展，通过消耗显著更多的算力资源取得了性能的明显进步。

  Seed-Prover工作当中所涉及到的四种工作流如右图所示，其中包括：

  - 全证明生成：直接生成完整证明，中间不接收验证反馈
  - 轻设定：全证明生成加反馈迭代
  - 中设定：逐引理验证与修正
  - 重设定：并行迭代证明多个猜想逐步推进证明
], [
  #image("./media/seed_prover_workflow_1.png", height: 45%)
  #image("./media/seed_prover_workflow_2.png", height: 45%)
])

= 自然语言证明

== 自然语言证明的背景

自从三年前ChatGPT发布之后，以它为代表的大语言模型开始拥有了通用的语言能力。事实上早在GPT-2的时代，人们就已经观察到了语言模型具有一定的处理数学问题的能力，但这种能力还非常孱弱，几乎无法解决稍微复杂一点点的计算问题。它的数学能力很多时候更像文本拟合带来的错觉，而不是真正拥有解题能力。而就在2021年时，两个经典的数学数据集被提出，用于衡量语言模型真正的推理能力。

- GSM8K (OpenAI, 2021.10): 包含8,500道小学数学练习题，发布之初最强的语言模型准确率也只有不到20%
- MATH (UC Berkeley, 2021.3): 包含12,500道高中数学题，当时GPT-3在其上的准确率仅5%左右，曾被视为不可逾越的高峰

而如今随着思维链技术普及、训练数据与方法的不断优化，当今的推理模型数学解题能力已经取得了极大的进步。自然语言证明的性能表现已经明显超越了此前的形式化证明。这或许说明了世界知识本身对于数学工作也同样至关重要。

== 从DeepSeekMath初代到如今

#tcslide([
  DeepSeekMath初代是DeepSeek早期针对数学任务的一个杰出工作，同时也是我做AI4Math的入坑作。它的主要方法包括：

  - 通过大量的数据工程取得数学能力的进步
  - 首次提出GRPO强化学习算法，解锁了更大的进步空间

  直到目前的DeepSeekMath-V2，大语言模型的基础数学能力进步同样依赖于类似的方案，但时代却已完全不同。
], [#image("./media/math_progress.png") #image("./media/grpo_formula.png")])

== AI Mathematician的展望

#tcslide([
  随着大语言模型数学证明能力的快速进步，我们也开始探索应用大模型与智能体技术解决前沿数学问题的可能 (并非像炒作新闻一样随便跑一点小结果出来就开始放卫星)。

  我们在2025年5月28日首次放出了#link("https://github.com/Carlos-Mero/AIM", "AIM")的原型，其中有几项关键要点，可能对于未来的研究工作会有所启发：

  - 纯自然语言推理路线
  - 针对复杂问题的引理式多步探索
  - 迭代式自我验证与调优
], figure(image("./media/yaqin_aim.jpeg"), caption: [张亚勤院士在人文清华论坛上对AIM的报告]))

== 自然语言与形式化

在近两个月当中另外也有一些很知名的自动定理证明智能体工作诞生，其中的代表就是Hilbert (UC San Diego, 2025.9.26) 和Aristotle (The Harmonic Team, 2025.10.1)，两者都采用了自然语言与形式化混合的技术方案，并取得了显著的成效。

它们两者在设计上的共同特点都是使用自然语言的推理引擎首先解决并拆解问题，而形式化部分则仅负责最终验证。事实上在Seed-Prover当中也使用了自然语言的不严谨思维链作为中间过程，这或许也就是未来一段时间当中正确的发展路线。最终两者分别低成本地实现了：

- Hilbert: 可解决99.2%miniF2F数据集当中的证明题和70%的PutnamBench竞赛题 (超过Seed-Prover的50%)
- Aristotle: 声称可以在IMO 2025上取得人类金牌的水平

最近也有听闻一些内部消息称DeepSeek内部也在针对V3.2设计类似Hilbert的自动定理证明智能体，并且效果极好，几乎已经完全战胜了PutnamBench。

== Hilbert架构设计

#image("./media/hilbert_workflow.png")
#grid(
  columns: 2,
  gutter: 0.4em,
  [
  Hilbert整体仍针对数学竞赛问题设计，应用gemini 2.5 pro等大模型作为reasoner生成自然语言证明，并配合RAG生成出Lean语言的证明梗概。后使用Goedel-Prover-V2 32B这样的prover模型去尝试解决各个子问题，之后再根据Lean编译器的反馈不断调整目标。其中Subgoal Decomposition的具体流程如右图所示。
],
  image("./media/hilbert_sample.png", height: 50%)
)

#focus-slide([
  #set page(margin: 1.4em)
  AI4Math讨论班第二季正片部分到此结束，感谢各位的陪伴！

  后续我们也会组织一些新系列的活动，敬请期待！
])
