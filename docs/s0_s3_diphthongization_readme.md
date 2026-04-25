# S0-S3 裂化分析 README

这个 README 是 [s0_s3_diphthongization_analysis.md](/Users/kleinchan/wuyu_vowel_chain/docs/s0_s3_diphthongization_analysis.md) 的配套导航文档，用来快速说明这份分析在做什么、依赖哪些文件、主要输出在哪里，以及结果应该怎样读。

## 1. 这份分析回答什么问题

这份分析聚焦吴语 `S0` 到 `S3` 四个链位，目标不是泛泛讨论所有复合元音，而是回答下面几类问题：

1. 哪些现代音值可以算“后元音链上的清晰裂化”？
2. 裂化主要集中在哪些链位？
3. 哪些声母条件更容易推动裂化？
4. `S2` 和 `S3` 之间有没有稳定的同现关系？
5. 如果再结合 ACO 结果，`S3` 的高后元音区是不是还存在 `u > o` 的局部降低支线？

一句话概括：

> 这份分析是在给 `S0-S3` 的裂化现象做口径清理、分布统计和结构解释。

## 2. 分析范围

本轮只处理：

- `chain_slot = S0 / S1 / S2 / S3`
- 现代读音列 `phonetic`
- 声母条件 `onset_class`

暂不讨论：

- `S4`
- 入声尾、非元音性读音
- `yo / yɑ / øy / uo / ua / uᴇ` 这类待判介音形式
- `ai / ɛi / æi / ɔi` 这类 i 尾复合音
- `au / ao / aɤ` 这类非本轮核心链条复合音

## 3. 核心分析口径

这份分析最重要的前提，是把“清晰裂化”从其他复合音现象里分出来。

主统计只使用两类 token：

- `monophthong`
- `clear_split`

核心分母定义为：

`eligible_tokens = monophthong + clear_split`

核心裂化率定义为：

`clear_split_rate = clear_split / eligible_tokens`

这意味着：

- `ia` 这类介音形式不算裂化
- `ai` 这类 i 尾复合音不算裂化
- `uo` 这类待判圆唇介音形式先不算裂化

所以这份结果讨论的是“后元音链相关裂化”，不是“所有看起来像双元音的形式”。

## 4. 主要文件

主说明文档：

- [s0_s3_diphthongization_analysis.md](/Users/kleinchan/wuyu_vowel_chain/docs/s0_s3_diphthongization_analysis.md)

分析脚本：

- [analyze_s0_s3_diphthongization.py](/Users/kleinchan/wuyu_vowel_chain/scripts/analyze_s0_s3_diphthongization.py)

主要输出表：

- `data_clean/value_type/s0_s3_split_value_classification.csv`
- `data_clean/value_type/s0_s3_medial_pending.csv`
- `data_clean/value_type/s0_s3_i_offglide_excluded.csv`
- `data_clean/value_type/s0_s3_non_core_diphthong_excluded.csv`
- `data_clean/value_type/s0_s3_split_onset_stats.csv`
- `data_clean/value_type/s0_s3_split_onset_sequences.csv`
- `data_clean/value_type/s0_s3_split_slot_stats.csv`
- `data_clean/value_type/s0_s3_split_slot_onset_stats.csv`
- `data_clean/value_type/s0_s3_split_point_onset_stats.csv`
- `data_clean/value_type/s0_s3_split_point_slot_stats.csv`
- `data_clean/value_type/s0_s3_split_implication_rules.csv`

主要图像：

- `figs/s0_s3_split_onset_rates.png`
- `figs/s0_s3_split_slot_rates.png`

与 ACO 衔接时会用到：

- [vowel_aco_small_results.md](/Users/kleinchan/wuyu_vowel_chain/docs/vowel_aco_small_results.md)
- `data_clean/value_type/aco_conditioned_group_summary.csv`
- `data_clean/value_type/aco_conditioned_path_summary.csv`

## 5. 怎么运行

在项目根目录运行：

```bash
env/bin/python scripts/analyze_s0_s3_diphthongization.py
```

如果你后面要把裂化结果和条件化 ACO 一起看，再运行：

```bash
env/bin/python scripts/run_vowel_aco_conditioned.py
```

## 6. 这份结果要先看哪几项

如果你只想快速抓住主结论，建议按这个顺序读：

1. 链位裂化率
2. 声母条件序列
3. 链位蕴涵关系
4. 声母蕴涵关系
5. 最后再看和 ACO 的衔接解释

最值得先记住的三点是：

1. `S2` 是裂化主场，`S3` 次之，`S0/S1` 很弱。
2. 如果讨论“裂化推动力”，优先看“方言点 + 链位等权 rate”。
3. `S2` 和 `S3` 在方言点层面完全同现，但强度不一样。

## 7. 声母序列应该怎么读

文档里给了多种序列，它们不是互相矛盾，而是在回答不同问题。

如果你问：

> 哪些声母条件下，裂化例子堆得最多？

看：

`token 贡献份额`

如果你问：

> 哪些声母更容易推动裂化？

看：

`方言点 + 链位等权 rate`

这也是当前最推荐的序列：

`N > T > TS > K > M > Ø > P`

这里的意思不是 `N` 的裂化例子绝对最多，而是说在控制链位和方言点样本量后，`N`、`T` 对裂化更敏感。

## 8. 链位关系应该怎么读

这份分析里，链位比声母更基础。

主结论是：

- `S2` 裂化率最高
- `S3` 也明显裂化，但低于 `S2`
- `S0/S1` 基本不是裂化主场

在方言点层面，目前观察到：

`S2 有裂化 <=> S3 有裂化`

但这不表示二者强度一样。比较合理的理解是：

> `S2` 更像裂化充分实现的位置，`S3` 则是同一区域里的较弱或分流位置。

## 9. 和 ACO 结果怎样接起来

如果只看这份统计文档，可以得到：

- 裂化核心在 `S2/S3`
- `S2` 强于 `S3`

如果再结合条件化 ACO，可以把高后元音区解释成两条后续机制：

```text
主链 / 裂化：
a > ɑ > ɔ > o > ɷ > u > əu

S3 局部降低：
u > ɷ > o
或
u > o
```

这样就能解释为什么：

- `S2/S3` 总是同现
- 但 `S2` 裂化更强
- `S3` 内部又会出现一部分 `u > o`

也就是说，`S3` 不只是“比 S2 更晚一步的裂化位”，它更像是高后元音区内部发生分流的链位。

## 10. 目前最稳妥的结论

可以先把这份分析压缩成下面这段话：

> S0-S3 的裂化不是整条链平均发生，而是主要集中在 `S2`，其次是 `S3`；`S0/S1` 基本不是核心裂化区。声母差异是在这个链位结构之内出现的，其中 `N/T` 的裂化敏感性较高，`K` 的覆盖面和材料贡献较大，`P` 最弱。若结合条件化 ACO，`S3` 还存在一条局部的 `u > o` 降低支线，因此 `S3` 更适合理解为高后元音区的分流位置，而不只是较弱的裂化位。

## 11. 后续最值得继续做什么

如果要把这份工作继续往前推，最建议做三件事：

1. 单独做 `S3` 的三分比较：`保持 u / 裂化为 əu / 降低为 o`
2. 在 `S3` 内部比较声母条件，检验 `M` 是否显著偏向 `u > o`
3. 按小片比较 `S3` 分流，检验 lowering 支线是否主要集中在苏沪嘉、临绍等区域

这三步会把现在的描述统计推进成更明确的机制分析。
