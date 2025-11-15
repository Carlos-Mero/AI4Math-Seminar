#import "@preview/may:0.1.1": *
#show: may-pre.with(
  config-info(
    title: [智能体],
    subtitle: [AI4Math讨论班第二季-06],
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

#image-slide(img: image("./media/s2e6.png"))

#title-slide()

= 前言

== 编码智能体 - Coding Agent

#tcslide([
  编写代码是当今语言模型最成功的应用之一，而这一任务本身在近几年来也取得了明显的进步。整体而言它的发展趋势是

  - 补全几行代码
  - 完成一个任务/issue
  - 端到端的完整开发流程

  当今的智能体系统能够自动执行的任务呈现出从简单到复杂，任务由短到长的趋势。

  时至今日，vibe coding已成AI家必争之地，而我在自己的科研代码当中已经有90%以上都由codex等编码智能体进行编写了。
], [
  #align(horizon, figure(
    image("./media/claudecode.jpeg"),
    caption: [Claude Code是第一个基于命令行的coding agent工具]
  ))
])

== OpenAI Codex

#tcslide([
  相比较手动输入prompt调用大模型完成工作而言，类似Claude Code或者OpenAI Codex这样的编码智能体可以：

  - 进行任务规划与多步执行
  - 自主搜索并查阅相关代码
  - 直接操作编辑代码文件
  - 主动执行各种shell指令
  - (还可以有网络搜索、图片阅览等等拓展功能)

  而当前除去这两家之外，还有如GitHub copilot，Cursor等等智能体应用。市场正处于勃勃生机玩物竞发的状态。
], image("./media/codex_examing.png"))

== 智能体技术已无处不在

#tcslide([
  时至今日，几乎所有面向用户的大模型前端都已经不再是单纯的自回归生成，而是已经融入了包括网络搜索、工具调用等等能力的某种「智能体」。这些能力的存在有效拓展了大模型的能力边界。

  #image("./media/hilbert_request.png")
  #image("./media/daiyu_request.png")
], image("./media/codex_completion.png"))

== GPT-5.1的工作流示例

#tcslide(image("./media/hilbert_response.png"), image("./media/daiyu_response.png"))

= 经典的智能体

#image-slide(img: image("./media/ym_agent.png"), body: [我们的助教老师正在执行一项特工任务])

== 智能体的概念

#tcslide([
  智能体 (Agent) 的概念就可以当作「具有智慧的个体」来进行理解。像ChatGPT这样的chatbot必定不会是智能的终极形态，而智能体这一概念则更加贴近人们对于人工智能的早期想象。智能体通常会涉及到如下四个关键词

  - 感知：观测获取环境信息
  - 推理/规划：处理信息并为目标策划行动
  - 动作：做出能够改变环境的行为
  - 记忆：记录环境信息服务前述各项行为
  
  这些概念都很抽象且普适，而早期智能体研究当中最重要的场景就是「游戏」。
], image("./media/agent_ex.png"))

== AlphaGO - 就是一种智能体

#tcslide([
  对于围棋类任务而言，我们可以据此理解AlphaGo系统的构建：

  - 感知：获取棋子位置与胜负信息
  - 推理：由神经网络预测落子位置以及局势，配合MCTS算法进行多步探索确定最优落点
  - 动作：落子
  - 记忆：无 (或者认为棋局属于记忆)

  因此这种下围棋的智能体唯一能做的事情就是下围棋，但是可以通过强化学习算法不断学习最终取得远胜人类的表现。
], image("./media/alphago.jpeg"))

== OpenAI Five

#tcslide([
  在大语言模型之前OpenAI研究的一个主流方向就是游戏智能体 (跟当时的DeepMind相当相似)。#link("https://openai.com/index/openai-five/", "OpenAI Five")完成于2018-2019年，是首个能够在Dota2当中击败职业冠军队伍的智能体系统。

  它的核心是一个大型的LSTM循环神经网络，网络输入是一个约20,000维的游戏信息向量，输出包括各种动作的概率分布，价值预测和一些辅助性信息。

  OpenAI Five不通过图像UI获取信息，而是直接通过游戏后端的API读取各种游戏状态数据。输出动作概率之后会采样确定具体动作。
], [
  #align(horizon, figure(image("./media/openai-five.webp"), caption: [OpenAI Five 2018年的宣传图]))
])

#pagebreak()

#tcslide([
  伴随着PPO强化学习算法的训练，OpenAI Five团队在实战当中自发地涌现出了

  - 团战占位合作
  - 经济与推塔节奏
  - 反打策略
  - Roshan时机
  - 拉野补刀
  - 闪烁先手，BKB timing

  等等能力。在OpenAI Five公测之后在线对阵人类玩家的胜率可以达到99.4%。

  不过须知由于方法限制，OpenAI Five的能力几乎无法泛化到Dota2之外的任何环境。
], align(horizon, figure(image("./media/group-laptop.webp"), caption: [OpenAI Five团队结算画面])))

== Dreamer - 吾好梦中玩游戏

#tcslide([
  DreamerV1-V4系列创建于2020年到2025年，相比较OpenAI Five应用复杂的特征工程作为输入的方案而言，Dreamer系列则直接将游戏图像编码进隐空间，并在隐空间上展开训练。这一方案就能够普遍适用于各种游戏项目。

  它的另一项创新在于训练隐空间当中的世界模型，从当前状态的隐变量与动作出发去预测下一步的隐变量，并在此基础上去训练动作网络。由此它就可以高效地学会游玩各种游戏的技术。#link("https://danijar.com/project/dreamerv3/", "DreamerV3")的论文发表在Nature上，其中称它为「通用的、跨领域的决策引擎」。
], align(horizon, box([#image("./media/dreamerv3.gif", height: 50%) #image("./media/dreamer_arch.png")])))

= LLM智能体

== 为什么使用LLM？

LLM通过大量的文本预训练获取了极为丰富的先验知识，并且可以通过不同的文本指令执行各种任务。这使得无训练仅通过prompt搭建通用智能体成为可能。而这种技术也将大大拓展智能体系统的应用空间。LLM智能体发展当中的代表性成果包括：

/ WebGPT: (OpenAI, 2021) 可以在纯文本浏览器当中使用搜索引擎，浏览网页并进行问答的智能体。
/ ReAct: (Princeton, 2022) 确定了先推理后选择工具执行，后接收工具返回的prompt范式，并为后续各种工作广泛采纳。
/ Reflexion: (2023) 建立了一个让agent从动作反馈当中总结积累文本经验，并将经验加入后续的prompt当中，以此就能实现不错的性能提升。
/ Function Calling / Model Context Protocol: (OpenAI / Anthropic) 确定了智能体通过函数调用工具的标准规范。

后来LLM基础模型的训练上也越来越重视提升基础的Agentic能力，如GPT-4.1，kimi k2，Anthropic系列模型都是如此，它们与智能体应用呈现双向奔赴的状态。

== Voyager - 早期LLM Agent的代表性成果

#align(center, image("./media/voyager.png"))

#pagebreak()

#tcslide([
  #link("https://arxiv.org/abs/2305.16291", "Voyager")诞生于GPT-4刚刚发布的大模型时代早期，但已经成功通过精心设计的智能体工作流，在Minecraft游戏任务上取得了令人惊叹的成果。它也由此成为了LLM智能体的经典工作。

  此时的GPT-4还完全没有多模态能力，因此Voyager通过#link("https://github.com/PrismarineJS/mineflayer", "Mineflayer API")直接获取Minecraft后台的游戏信息，加上此前的探索记忆，以纯文本的形式输入给大语言模型。随后GPT-4再以code as actions的方式输出代码交由Mineflayer执行。

  其关键技术在于自主规划，自我评估和技能库迭代三个方面上。
], box([
  #image("./media/voyager_components.png")
  #image("./media/voyager_curriculum.png")
]))

== Voyager - 关键的技术实现

#tcslide([
  Voyager当中的各个模块实际上都由GPT-4模型加上不同的prompt实现，同时由于当时的上下文长度与成本限制，技能库部分采用了RAG (检索增强) 方案。在一轮迭代当中Voyager会：

  - 根据当前进度提出下一步任务
  - 寻找已有技能组合成新的动作
  - 提交Mineflayer执行并获取反馈
  - 根据后续信息总结，并更新游戏经验以及技能库内容

  如此保持长期运行，voyager就可以在minecraft当中取得大量的成就。
], box([
    #image("./media/voyager_skill_library.png")
    #image("./media/voyager_self_verification.png")
]))

== Lumine - 该原神启动了

字节跳动Seed团队于2025.11.12公布了他们的#link("https://www.lumine-ai.org/", "Lumine")项目，这是一个完全模拟人类输入输出方式的视觉语言模型智能体。它在原神当中进行针对性训练，并成功完成了五小时的蒙德主线。这种游戏能力还可以直接迁移到崩坏-星穹铁道和鸣潮等等游戏当中。

#align(center, image("./media/lumine.jpeg", height: 70%))

== 原神是一款开放世界游戏

#align(center, image("./media/genshin.png"))

== Lumine的实现方案

Lumine建立在Qwen2-VL-7B-Base这一基础模型之上，它采用的是直接的Vision-Language模式。Lumine直接通过特定格式的文本输出来操作鼠标和键盘。这些文本也会进入到Transformer后续的自回归生成当中，并影响到它对于后续操作的判断。

#image("./media/lumine_model.jpeg")

== Lumine训练方案

#tcslide([
  Lumine接受输入的游戏画面是720P5Hz的超低游戏画面，但是对于VL模型而言已经足够了。

  Lumine的训练当中整体使用的是数据工程+SFT方案，并没有采用强化学习或工作流路线。它首先收集了2424h带有键盘/鼠标动作的人类游玩录像，经过不同的标注之后分隔成了三块数据，并在VLM上进行微调训练，使其具有了自主游玩游戏的能力。
], [
  #image("./media/lumine_training.png")
  #image("./media/lumine_data.png")
])

#focus-slide([
  #set page(margin: 1.4em)
  在游戏领域之外，基于大模型的智能体也取得了越来越多实用性的进展。其中的例子可能包括
])

== Cursor - 氛围编程带来工作效率大涨

#tcslide([
  #link("https://cursor.com/", "Cursor")可能是最知名的AI辅助编程工具，同时也是商业上最成功的智能体创业项目。主创团队创业两年多公司估值已经达到了接近300亿美元。

  #image("./media/cursor_sample.png")
], [
  #figure(image("./media/cursor_founders.png"), caption: [初创公司Cursor的四位创始人，他们都是MIT的本科辍学生。])
])

== Hilbert - 简易而有效的形式化证明工具

#tcslide([
  #link("https://arxiv.org/abs/2509.22819", "Hilbert")通过混合编排自然语言推理大模型与形式化证明器、验证器等形成大型的智能体工作流，以极低的成本实现了远胜针对性训练的专用prover的性能表现，令人汗颜。
], [
  #image("./media/hilbert_workflow.png")
  #image("./media/hilbert_sample.png")
])

== Agent Hospital - 大模型驱动的疾病诊断

通过让语言模型智能体在模拟医院环境当中学习并积累经验，最终在疾病诊断能力上超越了人类专家医师的水准。

#image("./media/agent_hospital.png", height: 80%)

#focus-slide([
  #set page(margin: 1.0em)
  或许我们不应该使用智能体元年这一说法

  而是智能体的这十年
])

#focus-slide([
  #set page(margin: 1.4em)
  在下周第七次讨论班当中，我们将会着重讲解强化学习算法，它们是一大类针对智能体任务设计的神经网络训练算法，并且被认为具有相当强大的潜力。
])
