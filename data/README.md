# 数据集

这些数据由本仓库的模拟器和求解器自动生成。面片顺序为 `URFDLB`，每面从外向内看按行排列。18 种面转动作均计为一步（HTM）。

## 浅层监督数据

| 文件 | 状态数 | 精确距离 1 / 2 / 3 |
|---|---:|---|
| `cube_shallow_256.jsonl` | 256 | 18 / 100 / 138 |
| `cube_trajectories_expanded.jsonl` | 323 | 18 / 167 / 138 |
| `cube_trajectories_1000.jsonl` | 1000 | 18 / 167 / 815 |

字段包括 `state`（54 字符状态）、`input`（六面输入）、`solution`、`optimal_distance`、`target_action` 和 `target_index`。后四项是标签，不能放进模型输入。`trajectory_sources` 记录展开来源。每个状态保留一个经验证的最短下一步；其他等价最优动作未必包含在标签中。

固定验证起点由 323 状态文件和种子 17 重建，共 76 个二步状态、256 个三步状态。1000 状态扩充避免训练轨迹经过这些起点，但不是对所有三步状态均匀抽样。验证中间状态可以与训练状态重合；未排除旋转等价状态。该验证集已经用于多轮开发。

在干净的仓库副本中重新生成（脚本会拒绝覆盖部分已有文件，因此先将现有对应数据移到其他目录）：

```bash
npm --prefix cube_lab ci
node cube_lab/generate_shallow.mjs
python cube_expand.py
python cube_expand_1000.py
```

## 随机打乱的 1000 条完整轨迹

`cube_random_1000/all.jsonl`：种子 20260925，每个魔方随机转动 30 次，相邻动作不转同一面，起点去重。这是随机游走，不是均匀随机状态采样。

cubejs 1.3.2 通用两阶段求解器返回的解法长度分布：

| 返回步数 | 数量 |
|---|---:|
| 18 | 2 |
| 19 | 7 |
| 20 | 36 |
| 21 | 202 |
| 22 | 753 |

平均 21.697 步。`by_length/` 按返回长度存放同一批记录。`solution_length` 是可行解长度，不是最短距离；`optimal_distance` 为 null。`states` 包含起点、逐步状态和最终还原状态，因此长度为 `solution_length + 1`。所有打乱、动作和中间状态都经过回放验证。

```bash
# 新目录不能已存在，父目录必须存在。
node cube_lab/generate_random.mjs data/cube_random_regenerated
```

该批轨迹尚未划分训练／验证集，也没有排除浅层验证状态。不能直接作为当前浅层训练脚本的输入。课程学习使用前需要先划分数据、检查状态重叠并展开后缀。通用求解器的后缀分布也不能视为同距离所有魔方状态的均匀样本。

## 在线 GRPO 数据

GRPO 从排除固定验证起点后的二、三步状态池采样，模型实际执行生成 `episodes.jsonl`。这是在线交互数据，与上述专家轨迹不同。运行目录里的 `training_data.jsonl` 是监督阶段快照；不是 RL 的动作标签来源。运行日志默认不纳入版本控制。
