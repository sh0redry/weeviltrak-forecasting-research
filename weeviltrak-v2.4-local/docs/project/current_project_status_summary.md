# Historical Project Status Summary (v2.2-era)

> **Historical document.** This captures an earlier v2.2 monotonic/zero-day
> investigation. Current v2.4 state is documented in
> [`2026_project_handoff.md`](2026_project_handoff.md).

## 我们这个项目的目标是什么

这个项目的目标，是用天气和地点信息去预测 boll weevil 不同 life stage 什么时候会发生。

更具体地说：

- 输入是某个地点在某一天之前的天气累计情况和相关特征
- 输出是这个地点距离某个虫态发生还剩多少天，也就是 `days_to_event`

最后业务上希望得到的是：

- 某一天、某个地点、某个 stage 的预测剩余天数
- 以及对应的预测发生日期 `predicted_stage_date`

## 我们现在想解决的问题是什么

当前最主要的问题叫做 **bounce back**。

用人话说，就是：

- 模型在接近事件的时候，预测有时会变小
- 但到了后面又会重新变大
- 特别是事件已经发生之后，模型有时还会继续预测“还剩几天”

这和业务直觉是不一致的，因为正常应该是：

- 越接近事件，剩余天数应该总体下降
- 一旦事件已经发生，剩余天数应该停在 `0`

所以我们现在真正想解决的，是两个相关但不完全一样的问题：

1. **post-event 问题**
   事件已经发生以后，为什么模型还在给正值，而不是 `0`

2. **pre-event bounce back 问题**
   在事件发生前几天，为什么预测有时会来回波动，而不是稳定下降

## 我们已经尝试了什么

### 1. 加 LightGBM monotonic constraint

我们先在 LightGBM 路径上加了 monotonic constraint：

- 目前只对 `cumu_gdd_air` 加了 `-1`
- 意思是：累计 GDD 越高，预测的 `days_to_event` 不应该反而变大

这个改动的出发点是：

- 用模型结构约束来减少明显不合理的回弹

### 2. 做了轻量 business check

为了不一上来就做完整 backtesting，我们先做了一个 lightweight business check：

- 用真实 `weevil_data`
- 固定 `stage_id = 2`
- 选每个真实事件前 `7` 天到后 `2` 天的窗口
- 比较 `v2.1` Random Forest 和 `v2.2` LightGBM

这个检查主要看两个指标：

- `bounce_back_count / rate`
- `post_event_positive_count / rate`

### 3. 加 post-event zero samples

后面我们发现，单靠 monotonic constraint 不够，因为模型虽然学会了“接近事件”，但没有被明确教会“事件发生以后就应该是 0”。

所以我们改了训练数据：

- 在每个真实 `stage_date` 之后，再补一个短窗口
- 使用这些天**真实存在的天气数据**
- 把这些 post-event 样本的目标值统一设成 `0`

目前这个窗口参数就是：

- `post_event_zero_days`

它的意思是：

- 在真实事件发生之后，再额外加入多少天的真实天气样本，并把目标都设成 `0`

### 4. 加 zero-day termination

除了训练数据增强，我们还加了一个后处理 safety net：

- 如果连续两天预测都已经很接近 `0`
- 就把后面的预测锁定为 `0`

这个规则的作用是：

- 防止模型在已经接近事件后，又继续往上弹

## 我们目前得到的结果是什么

### 第一阶段结果：只加 monotonic constraint

当时的结论是：

- 有帮助
- 但不够

它确实减少了一部分 bounce back，但并没有真正解决“事件之后应该归零”这个问题。

### 第二阶段结果：加 post-event zero samples + zero-day termination

这个阶段的结果明显更好。

旧版 `v2.2` 的结果大致是：

- `bounce_back_count = 449`
- `post_event_positive_count = 516`
- `post_event_positive_rate = 0.9181`

加入 post-event zero samples 和 zero-day termination 之后，新版 `v2.2` 变成：

- `bounce_back_count = 347`
- `post_event_positive_count = 357`
- `post_event_positive_rate = 0.6330`

这说明：

- bounce back 明显减少了
- 事件之后还预测为正值的问题也明显减少了

所以目前可以比较明确地说：

- 这个方向是有效的
- 真正起关键作用的，不只是 monotonic constraint
- 更重要的是补上了“event 后应该为 0”的训练监督信号

## 我们现在的理解是什么

到目前为止，我们对问题的理解大概是这样：

### 1. post-event 问题的核心

核心原因不是单纯参数没调好，而是：

- 训练集中过去主要是 pre-event 样本
- 模型没有系统学到“事件已经发生后应该停在 0”

所以：

- `post-event zero samples` 是很关键的修复

### 2. pre-event bounce back 的核心

pre-event bounce back 还没有完全解决。

我们现在判断，它更可能来自这些因素：

- 特征和标签之间并不是非常干净的一一对应关系
- 当前特征虽然能描述天气累积，但不一定足够直接描述“已经快发生了”
- 除了 `cumu_gdd_air` 之外，其他特征仍然会把预测往上拉
- 观测日期本身也有噪声

所以：

- monotonic constraint 对这个问题有帮助
- 但不一定能完全解决

## 我们现在在考虑的下一步办法是什么

当前在考虑的办法主要有三类。

### 1. 继续调 `post_event_zero_days`

这个参数不是越大越好。

如果它太小：

- 模型学到的 post-event zero 信号可能不够强

如果它太大：

- 训练集里 `0` 标签会占更多比例
- 模型可能会更容易过早地把 event 前的预测也往 `0` 压

所以我们现在已经在考虑做 sweep，自动比较不同 `post_event_zero_days` 的结果。

### 2. 继续优化 pre-event 特征

如果未来还要继续改善 pre-event bounce back，我们可能会考虑增加更接近“临近事件”的特征，比如：

- 更直接的 seasonal progress features
- 距离某类阈值的 proxy 特征
- location-stage 的历史典型 timing 先验

这些特征的作用，都是让模型更清楚地知道：

- 现在离事件到底有多近

### 3. 评估是否要做两阶段模型

我们也讨论过另一种更结构化的方案：

- 一个分类模型判断“是否已经达到该 stage”
- 一个回归模型只负责预测“如果还没达到，还差多少天”

这个思路在逻辑上是合理的，因为：

- post-event 的“是否已经发生”本质上更像分类问题
- pre-event 的“还差几天”才是连续回归问题

但目前还没有进入实现阶段，因为：

- 当前单回归路线在加了 post-event zero samples 之后已经明显变好了
- 所以两阶段模型还属于潜在下一阶段方案，而不是当前主线

## 当前可以怎么一句话总结

如果用最简单的人话总结目前的项目状态，就是：

- 我们的目标是预测 boll weevil 的 stage 发生时间；
- 当前最主要的问题是模型在接近或超过事件后会出现 bounce back；
- 我们先试了 monotonic constraint，发现有帮助但不够；
- 后来又加了 post-event zero samples 和 zero-day termination，效果明显变好；
- 现在已经确认这个方向有效，下一步重点是继续平衡“event 后归零”和“event 前不要被压得太早”之间的关系。
