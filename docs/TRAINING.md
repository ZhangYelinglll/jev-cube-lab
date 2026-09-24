# Training and evaluation guide

See [README.md](../README.md) for environment setup and set `CUBE_MODEL` to your local Qwen3.5-0.8B directory. Run commands from the repository root. The generated shallow datasets are included; see [data/README.md](../data/README.md) to regenerate them.

## 256-state supervised warm-up

Clone or pull the full repository on the training server and run commands from its root; preserve the `cube/`, `tests/`, and `data/` directories. Use the same Python 3.11 environment that passed the GPU smoke check: torch 2.6.0 CUDA 12.4 and transformers 5.3.0. No additional training dependencies are required.

```bash
CUDA_VISIBLE_DEVICES=0 python -m cube.train \
  --model "$CUBE_MODEL" \
  --data data/cube_shallow_256.jsonl \
  --output "$HOME/cube-runs/overfit-256" \
  --batch-size 8 --epochs 100
```

The downloaded model is the non-Base version. This experiment trains the full text backbone and a randomly initialized 18-action head using single-label cross entropy. It does not train a success-probability head or perform reinforcement learning yet.

Only the 54-facelet state enters the model. Each state has one verified shortest-path next-action label; another equally optimal action currently counts as a label mismatch. All 256 rows are used for overfitting, without a validation split. Accuracy is agreement with these training labels, NOT held-out accuracy or full-cube solving success.

Epoch 0 records initial metrics. Stop at 99% training-label accuracy or 100 epochs, whichever occurs first. Reaching 99% is an experimental target, not guaranteed. If needed, lower batch size to 2 for memory. Existing output directories are refused; select a new output path for each run.

Outputs:
- `metrics.jsonl`: epoch loss, label accuracy, accuracy by depth, elapsed time, peak allocated VRAM.
- `run.json`, `training_data.jsonl`: settings and exact samples used.
- `summary.json`: final metrics and whether the accuracy target was reached.
- `checkpoint/backbone/`: text backbone, reload with `Qwen3_5TextModel.from_pretrained(...)`.
- `checkpoint/action_head.pt`: FP32 `Linear(hidden_size, 18)` state dictionary, load with `torch.load(..., weights_only=True)`.
- `checkpoint/tokenizer/`, `checkpoint/actions.json`: tokenizer and fixed action ordering.

The checkpoint stores final weights, not a standard chat model or an exact optimizer-resume checkpoint. Save occurs after normal completion; leave sufficient disk space for FP32 weights. Original downloaded weights are unchanged. Training outputs are git-ignored.

Optional CPU regression check:

```bash
OMP_NUM_THREADS=2 python -m tests.check_train
```

This tests a tiny random Qwen3.5 with both linear and full attention: data checks, loss decrease, backbone updates, and saved/reloaded prediction parity. Full 0.8B CUDA training must be run on the training server.

## Closed-loop evaluation

The evaluator is available as `python -m cube.eval`. It uses pure Python facelet permutations verified against cubejs, so the training server does not need Node.js. Run:

```bash
CUDA_VISIBLE_DEVICES=1 python -m cube.eval \
  --checkpoint "$HOME/cube-runs/overfit-256/checkpoint" \
  --batch-size 16 --max-steps 10
```

The default case source is the exact `training_data.jsonl` saved in the training run. All 256 training starts are evaluated, plus the 143 remaining exact-depth-2 starts and a seeded sample of 256 unseen exact-depth-3 starts. There are no held-out depth-1 starts because all 18 were used for training. Held-out means unseen **initial state**; later states in an episode can overlap training states. No claim of rotation-equivalent state exclusion is made.

Decisions use greedy argmax with no inverse-move mask, cycle filter, solver hints or search. Repeated states are recorded but do not stop a run. Each run ends at solved or the 10-move budget. Solving on the final allowed move counts as success. All moves count as one, including half turns. BFS is used only to construct and label the evaluation cases; the policy only receives current facelets.

Results go in a new timestamped `eval-*` directory beside the checkpoint: frozen `cases.jsonl`, replayable `episodes.jsonl`, and `summary.json`. Reported metrics include success rate, mean moves among successful episodes (null when none succeed), shortest-solve rate and fraction of episodes revisiting a state. Training and held-out metrics are separate, including by starting depth.

Optional local evaluator check: `python -m tests.check_eval`. It verifies case exclusion, known shallow shell counts, all 3501 shallow solutions under an oracle, cycle behavior and final-step success. It does not run the trained checkpoint.

## Full-trajectory supervision (323 states)

`python -m cube.expand` expands the original 256 verified shortest solutions into 632 step occurrences and 323 unique states (18 depth-1, 167 depth-2, 138 depth-3). All 67 new states are depth-2 intermediates. Original first-action labels remain unchanged. Each nonterminal intermediate state has a next-action label; terminal solved states are omitted. This remains single-label cross entropy, with each unique state weighted once per epoch.

Generated data: `data/cube_trajectories_expanded.jsonl`. Its `.meta.json` explains counts; `.eval_cases.jsonl` freezes the seed-17 comparison cases. The current evaluator deterministically regenerates those same cases with the expanded file passed to `--data` and seed 17.

Train again from the SAME original Qwen weights and the same default learning rates/seed, so the first comparison isolates expanded data rather than a changed initialization:

```bash
CUDA_VISIBLE_DEVICES=1 python -m cube.train \
  --model "$CUBE_MODEL" \
  --data data/cube_trajectories_expanded.jsonl \
  --output "$HOME/cube-runs/overfit-323" --batch-size 8 --epochs 100
```

The 67 new training states were previously held-out depth-2 starts. Do not compare the new held-out aggregate to the old 399-case score. Re-evaluate both checkpoints on the same expanded exclusion set (76 held-out depth-2 starts + 256 depth-3 starts):

```bash
CUDA_VISIBLE_DEVICES=1 python -m cube.eval \
  --checkpoint "$HOME/cube-runs/overfit-256/checkpoint" \
  --data data/cube_trajectories_expanded.jsonl --seed 17 --max-steps 10

CUDA_VISIBLE_DEVICES=1 python -m cube.eval \
  --checkpoint "$HOME/cube-runs/overfit-323/checkpoint" \
  --data data/cube_trajectories_expanded.jsonl --seed 17 --max-steps 10
```

For the old checkpoint, the report's `train` group now means the expanded comparison pool, including 67 states it did not train on. Compare `heldout` and its per-depth groups across checkpoints. This is a diagnostic overfit experiment; different stopping epochs/update counts must be reported, and the reused test panel is not a fresh final benchmark.

Run `python -m tests.check_expand` for CPU-only checks: preservation of original labels, coverage of every intermediate state and chosen successor, no solved-state labels, rejection of invalid solutions.

## 1000-state experiment with the same held-out starts

`python -m cube.expand_1000` keeps the original 323 rows unchanged and adds 677 exact-depth-3 states. Total: 18 depth-1, 167 depth-2, 815 depth-3. Each added state has a shortest continuation through existing training states; its entire path avoids the frozen 332 held-out starts. Sampling is therefore conditioned on this path coverage, not uniform over all depth-3 states. All solutions and intermediate coverage are verified.

Copy `data/cube_trajectories_1000.jsonl` to the training server, then train from the original Qwen model:

```bash
CUDA_VISIBLE_DEVICES=1 python -m cube.train \
  --model "$CUBE_MODEL" \
  --data data/cube_trajectories_1000.jsonl \
  --output "$HOME/cube-runs/overfit-1000" --batch-size 8 --epochs 100
```

For direct comparison with the recorded 323-state model's 185/332 held-out successes, explicitly retain the OLD evaluation data argument:

```bash
CUDA_VISIBLE_DEVICES=1 python -m cube.eval \
  --checkpoint "$HOME/cube-runs/overfit-1000/checkpoint" \
  --data data/cube_trajectories_expanded.jsonl --seed 17 --max-steps 10
```

This uses exactly the same 332 held-out starts and reports `train` on the original 323-state diagnostic subset, not all 1000 training states. Omitting `--data` would regenerate a different test set from the 1000-state training snapshot and invalidate that direct comparison. Epoch/update counts also differ by training-set size; report them with results.

## Online GRPO pilot after 1000-state SFT

`cube/grpo.py` starts from the saved SFT actor and performs online environment interaction. No solver action labels enter the RL loss. This is GRPO with an auxiliary Brier confidence head, NOT a reproduction of Jev RLCD or the original RLCR generated-confidence method. The RLCR paper itself uses GRPO: https://arxiv.org/html/2507.16806v2 .

For each sampled start, generate 8 independent trajectories at temperature 1, with up to 10 actions. Terminal reward Y is 1 exactly when solved within the budget, otherwise 0. Normalize rewards within each group, and update the actor using the clipped probability ratio (clip 0.2). All-success/all-failure groups have zero reward advantage. Add exact categorical KL to the frozen SFT actor (weight 0.02) and differentiable `(q-Y)^2` (weight 0.5). Each episode has equal loss weight; its steps are averaged. There is no critic or expert-label auxiliary loss.

The new sigmoid confidence head reads the last hidden state plus remaining budget divided by the maximum budget. Actor inputs remain unchanged. Confidence targets refer to the temperature-1 sampling policy, not the greedy evaluator. Brier gradients update both confidence head and shared backbone. Training Brier is a fit statistic, not proof of calibration; the changing policy and data selection also affect it.

The original RLCR reward is `Y-(q-Y)^2`, jointly optimizing answer and generated confidence. Simply inserting a deterministic state-only q into GRPO rewards does not reproduce it: within a start group that reward is `(1+2q)Y-q^2`, whose shift/scale cancels under group normalization (apart from epsilon). A detached reward also cannot train the confidence head directly. Hence this pilot uses an explicit differentiable auxiliary loss and records the adaptation in `run.json`.

The training start pool includes all exact-depth-2/3 states except the frozen 332 test starts. This adds new training starts beyond the 1000 SFT examples; results therefore measure online RL with expanded experience, not optimizer differences alone. Entire groups are discarded if any rollout visits a reserved start. This prevents training on those test states, but conditions the retained distribution: monitor `discarded_groups`. Intermediate evaluation states may overlap training as before. The repeatedly used panel is a development comparison, not a fresh final test.

Pull the full repository on the training server. No new package is required. The current implementation starts from an SFT checkpoint; it does not resume RL optimizer/confidence state.

```bash
python -m tests.check_grpo

CUDA_VISIBLE_DEVICES=1 python -m cube.grpo \
  --checkpoint "$HOME/cube-runs/overfit-1000/checkpoint" \
  --eval-data data/cube_trajectories_expanded.jsonl \
  --output "$HOME/cube-runs/grpo-1000-pilot" \
  --iterations 20 --groups 4 --group-size 8 --max-steps 10

CUDA_VISIBLE_DEVICES=1 python -m cube.eval \
  --checkpoint "$HOME/cube-runs/grpo-1000-pilot/checkpoint" \
  --data data/cube_trajectories_expanded.jsonl --seed 17 --max-steps 10
```

Use a fresh output directory. First compare held-out greedy success to SFT's 222/332 (66.87%), depth-3 success to 149/256 (58.20%), and repeat rate to 32.53%. Twenty iterations are a pipeline pilot, not a convergence guarantee. Inspect `informative_groups` (mixed outcomes), discarded groups, KL, and Brier before extending training. Rollout success is sampled on retained training starts, so it cannot be compared directly to held-out greedy success. Saved episodes contain actions, predictions, and actual outcomes for inspection. The saved actor remains compatible with `cube/eval.py`; confidence weights and their semantics are saved separately. Independent confidence evaluation is not implemented in this pilot.

CPU check covers terminal rewards, group advantages, no-signal groups, reserved-state rejection, gradient direction, and an actual tiny hybrid Qwen3.5 forward/backward with gradient checkpointing. Full-size H800 training must be validated on the training server.

## Explore reward signal before another training run

`cube/probe.py` freezes the actor and compares temperatures 1.0/1.3/1.6 on the same 64 sampled non-test starts. For each start and temperature it samples 16 trajectories, then reports group sizes 8 (prefix) and 16 from those same draws. This paired comparison saves inference and isolates group-size effects. The first 8 retain their own validity even if later trajectories hit reserved states. No weights or training settings are changed. Confidence predictions are not evaluated (the loader initializes that head afresh).

```bash
python -m tests.check_grpo
CUDA_VISIBLE_DEVICES=1 python -m cube.probe \
  --checkpoint "$HOME/cube-runs/overfit-1000/checkpoint" \
  --eval-data data/cube_trajectories_expanded.jsonl \
  --output "$HOME/cube-runs/signal-probe-sft" \
  --starts 64 --temperatures 1.0 1.3 1.6 --group-sizes 8 16
```

Pull the full repository to update the probe and its shared modules. The original trainer still uses temperature 1; the probe's configurable temperature is for inference only. The collector stores the actual temperature-scaled sampling log-probability. Do not start training at a different temperature without also changing the objective/reference probabilities consistently.

Inspect `summary.json`: `informative_fraction_attempted` includes the cost of discarded groups; `informative_fraction_accepted` isolates reward diversity after exclusion; `informative_groups_per_1000_actions` accounts for sampling work. Also inspect `start_entropy_nats` (maximum ln(18)), `start_top1_probability`, unique first actions, and unique accepted action sequences. Diverse trajectories with uniform outcomes indicate a different issue from identical trajectories under a concentrated policy. Reserved trajectories terminate early, so their final success is unknown and excluded from success-rate statistics. Wall time covers collection of the maximum group size, not a separate timing benchmark for each prefix. The 64 starts are an initial diagnostic sample; confirm a promising setting with another seed before claiming a stable gain. Higher temperature may increase diversity but also failures and group rejection.

This diagnostic intentionally precedes adaptive resampling or separating confidence gradients, so changes in reward diversity are attributable to sampling settings alone. Model-specific results require running the saved checkpoint on the GPU server.
