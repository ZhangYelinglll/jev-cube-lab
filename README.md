# Jev Cube Lab · 魔方决策与学习实验

探索模型能否根据三阶魔方的当前状态选择动作，并通过监督学习和强化学习提高还原能力。

项目包含两个独立部分：**Jev API 网页实验**展示动作概率、轨迹和旋转动画；**Qwen3.5-0.8B 本地训练实验**使用 18 类动作头，支持监督学习、闭环评估和 GRPO 小实验。目前本地训练模型尚未接入网页。

> 这是研究原型，不是稳定还原任意魔方的产品，也不是 Jev RLCD 或原版 RLCR 的复现。已完成浅层状态实验；渐进式课程训练仍在计划中。

## 已实现

- 网页逐步决策、动画、回放、实验记录导出；API 密钥仅在服务端读取。
- 54 面片状态输入，18 种动作输出：`U U' U2 R R' R2 F F' F2 D D' D2 L L' L2 B B' B2`。
- Qwen3.5 文本主干全参数监督训练及独立动作头保存。
- 无搜索、无动作过滤的贪心闭环评估。
- 在线 GRPO：终局成功奖励、冻结参考策略 KL、辅助 Brier 成功概率头。
- 固定模型的采样温度与轨迹多样性探测。
- 浅层精确解数据，以及 1000 条随机打乱魔方的求解器轨迹。

所有动作（包括 `U2`）计为一步。打乱次数、求解器返回长度、最短距离是三个不同概念。

## 快速运行网页

需要 Python 3.11+、Node.js 22+、npm 和 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/ZhangYelinglll/jev-cube-lab.git
cd jev-cube-lab
cp .env.example .env
# 编辑 .env，填写自己的 TYPESAFE_API_KEY
npm --prefix web ci
npm --prefix web run build
uv run --env-file .env python -m cube.server
```

打开 <http://127.0.0.1:8765>。远程服务器需要监听所有网卡时：

```bash
uv run --env-file .env python -m cube.server --host 0.0.0.0 --port 8765
```

该服务没有用户认证，不要直接作为公开 API 部署；访问者可消耗服务端的 Jev API 用量。远程个人使用可选择端口转发或限制网络访问。公开源代码不要求把运行服务暴露到公网。

Jev 网页会过滤立即逆操作和部分重复状态；本地 Qwen 评估没有这些过滤，因此两者成绩不能直接比较。网页显示的置信度不是还原成功率。使用细节见 [网页说明](web/README.md)。

## 训练环境与复现

已在独立 H800 服务器验证的环境：Python 3.11、PyTorch 2.6.0+cu124、Transformers 5.3.0。网页不需要安装训练依赖。请使用独立环境，并根据 GPU 驱动安装 PyTorch；以下对应 CUDA 12.4：

```bash
python -m venv .venv-train
source .venv-train/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-training.txt
python -m cube.smoke --self-test
```

自行下载 `Qwen/Qwen3.5-0.8B`，将 `CUBE_MODEL` 指向完整的本地模型目录。实验使用非 Base 版本，模型权重不在本仓库分发，使用时遵守上游模型条款。

```bash
export CUBE_MODEL=/path/to/Qwen3.5-0.8B
CUDA_VISIBLE_DEVICES=0 python -m cube.smoke \
  --model "$CUBE_MODEL" --batch-size 2 --steps 1

CUDA_VISIBLE_DEVICES=0 python -m cube.train \
  --model "$CUBE_MODEL" \
  --data data/cube_trajectories_1000.jsonl \
  --output "$HOME/cube-runs/overfit-1000" --batch-size 8 --epochs 100

CUDA_VISIBLE_DEVICES=0 python -m cube.eval \
  --checkpoint "$HOME/cube-runs/overfit-1000/checkpoint" \
  --data data/cube_trajectories_expanded.jsonl --seed 17 --max-steps 10
```

注意评估命令刻意使用 **323 状态数据文件**来重建固定的 332 个验证起点，不要改为 1000 状态文件后直接比较结果。所有训练命令要求新的输出目录。

GRPO、探测实验和各阶段详细流程见 [TRAINING.md](docs/TRAINING.md)。目前 `cube/grpo.py` 的训练采样温度固定为 1.0；`python -m cube.probe --temperatures ...` 只做诊断，不改变训练策略。新的课程学习、有效组补采样、关闭置信度主干梯度等方案尚未实现。

## 数据与结果

仓库提供自动生成的魔方数据，不包含模型权重、API 响应日志或个人运行目录：

| 数据 | 用途 |
|---|---|
| `cube_shallow_256.jsonl` | 256 个精确距离 1～3 的状态 |
| `cube_trajectories_expanded.jsonl` | 展开专家路径并去重后的 323 状态 |
| `cube_trajectories_1000.jsonl` | 保留验证起点后扩充的 1000 状态 |
| `cube_random_1000/all.jsonl` | 1000 个随机打乱 30 次的状态与完整可行解 |

生成方法、字段与数据边界见 [数据说明](data/README.md)。最后一批解法长度为 18～22，**不是最短距离**，尚未拆分训练／验证集，不能直接交给当前浅层 `cube/train.py`。

固定 332 个浅层验证起点上的单次实验记录：

| 模型 | 成功数 | 还原率 |
|---|---:|---:|
| 323 状态监督模型 | 185/332 | 55.72% |
| 1000 状态监督模型 | 222/332 | 66.87% |
| 1000 状态监督模型＋20 轮 GRPO | 224/332 | 67.47% |

这些结果由训练服务器运行后记录，未附权重；不是多随机种子结论，也不证明完整随机魔方的还原能力。验证集已反复使用，中间状态允许与训练集重叠。完整实验口径见 [EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 检查

不需要 API 密钥或 GPU 的基础检查：

```bash
uv run python -m tests.check_lab
python -m tests.check_eval
python -m tests.check_expand
python -m tests.check_data
npm --prefix web run check
npm --prefix web run build
```

安装训练依赖后还可以运行真实微型 Qwen3.5 的 CPU 检查：

```bash
OMP_NUM_THREADS=2 python -m tests.check_train
OMP_NUM_THREADS=2 python -m tests.check_grpo
```

浏览器检查见 [网页说明](web/README.md)，使用离线替身，不消耗 API。`web/evaluate.mjs` 则会真实调用 Jev，应主动运行并承担用量。

## 项目结构

```text
jev-cube-lab/
├── cube/                 # Python 服务、训练、评估与数据处理
│   ├── server.py         # Jev API 服务
│   ├── smoke.py          # 模型环境验证与共享动作定义
│   ├── train.py          # 监督训练
│   ├── eval.py           # 魔方模拟与闭环评估
│   ├── grpo.py           # GRPO 训练
│   ├── probe.py          # 探索信号诊断
│   ├── expand.py         # 专家轨迹逐步展开
│   └── expand_1000.py    # 浅层数据扩充
├── web/                  # 页面、动画、npm 依赖和前端检查
│   └── tools/            # JavaScript 数据生成工具
├── tests/                # Python 离线检查
├── data/                 # 已发布数据与字段说明
├── docs/                 # 训练指南与实验记录
└── .github/workflows/    # 自动检查
```

所有 Python 命令在仓库根目录使用模块方式运行。例如，旧命令 `python cube_train.py` 改为 `python -m cube.train`，旧命令 `python cube_eval.py` 改为 `python -m cube.eval`。前端命令改为 `npm --prefix web ...`。拉取新版本时保持完整目录结构；已有模型检查点和数据格式不变。

## 参考

- [TypeSafe / Jev](https://docs.typesafe.ai/)：网页决策接口；本项目为独立实验。
- [cubejs](https://github.com/ldez/cubejs)、[cubing.js](https://github.com/cubing/cubing.js)：状态、求解及动画。
- [DeepCubeA](https://github.com/forestagostinelli/DeepCubeA)：强化学习与搜索方向参考。
- [RLCR](https://arxiv.org/abs/2507.16806)：校准奖励研究；本项目的辅助 Brier 设计是适配实验。

## 许可

目前仅公开源代码，暂未授予开源许可证。公开可见不等于获得通用的复制、修改或再分发授权。第三方依赖及预训练模型遵循各自许可证，派生材料说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
