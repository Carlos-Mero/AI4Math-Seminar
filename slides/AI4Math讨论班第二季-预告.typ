#import "@preview/may:0.1.1": *
#show: may-pre.with(
  config-info(
    title: [AI4Math讨论班第二季],
    subtitle: [即将开启！],
    author: [黄砚星],
    institution: [求真书院],
    date: datetime.today()
  )
)

#title-slide()

= 先行预告

== 讨论班整体情况

#text(fill: sea, weight: "bold", [大致内容])

自动定理证明的概念自19世纪50年代提出开始便始终是人工智能研究的首要目标之一，直到近年以来随着大语言模型和推理模型的发展，这一夙愿才终于渐渐成为可能。时至今日，数学推理能力已经成为大语言模型最重要的性能指标之一，针对数学问题的训练则有效提升了大模型的能力边界。与此同时我们也相信，强大的数学推理能力会是通向AGI的必要条件。

AI4Math讨论班第二季的内容整体采用总分结构，会首先快速讲解大语言模型与推理模型的大致情况，随后再分别讲解与讨论各部分的细节内容。我们会更加关注各种技术背后的直觉而非实现细节，同时配有丰富的 (故事) 以及代码示例。在内容安排上本次讨论班与上一季大体一致，但是会加入更多新兴成果、新闻，同时会有许多优化。

#text(fill: sea, weight: "bold", [先修要求]):
少许微积分、线性代数、概率论，以及一定的python编程基础即可。(如果都没有其实也可以单纯来听故事x)

#pagebreak()

#grid(
  columns: (1fr, 1fr),
  column-gutter: 2em,
  row-gutter: 0em,
  [
/ 时间安排:
  - 从2025年秋季学期第四周开始
  - 截止于第十三周，共计十次
  - 每周六晚19:20-21:20
/ 地点安排:
  - 线下地点根据参与人数确定
  - 每周会提前在微信群内通知
  - 每次活动同时配有腾讯会议直播
  - 对于不便实时参与的同学也会提供每次讨论班的录屏
    - (录屏内容请勿外传！)
  - 群聊二维码见下文
/ 资源链接: #link("https://github.com/Carlos-Mero/AI4Math-Seminar", "讲义源码") #link("https://github.com/Carlos-Mero/may", "文档模板")
],
  [
  #set align(right)
  #grid(
    columns: 2,
    align: (left + bottom, left),
    [
    *主讲人*
    ],
    image("media/hyx.png", height: 40%)
  )
  #v(-0.9em)
  #line(length: 80%)
  #v(-0.9em)
  黄砚星，求真书院一年级博士生 THUNLP-MT成员
  #v(-1em)
  #grid(
    columns: 2,
    align: (right + bottom, left),
    [
    *助教*

    林心诚

    沈意明
  ],
    image("media/tas.jpeg", height: 40%),
  )
  #v(-0.9em)
  #line(length: 80%)
  #v(-0.9em)
]
)

== 内容目录 (暂定)

第二季AI4Math讨论班计划从2025年秋季学期第四周周六 (2025年10月11日) 开始，此后每周固定开展一次，各位可以根据自己感兴趣的课题选择参与！大致内容目录如下所示：

#grid(
  columns: (1fr, 1fr),
  [
1. 第四周 - 大语言模型
2. 第五周 - 深度思考革命
3. 第六周 - 模型结构-上
4. 第七周 - 模型结构-下
5. 第八周 - 训练流程
6. 第九周 - 强化学习
7. 第十周 - 智能体系统
8. 第十一周 - 形式化数学证明
9. 第十二周 - 视觉生成模型
10. 第十三周 - 前沿成果选讲
],
  [
  #set text(size: 0.8em)
  #figure(
    caption: [讨论班微信群！(有效期7天)],
    image("media/group_qrcode.JPG", height: 65%)
  )
]
)

#focus-slide([感谢关注！])
