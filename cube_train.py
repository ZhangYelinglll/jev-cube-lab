#!/usr/bin/env python3
"""Small-set supervised overfit experiment; NOT held-out cube evaluation.
Uses the same torch/transformers environment as cube_smoke.py.
Only state text enters the model; targets use one verified expert action.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path
import time

import torch
from cube_smoke import ACTIONS, action_logits


def read_data(path):
    rows, seen = [], set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        state = row['state']
        target = row['target_index']
        if (not isinstance(state, str) or Counter(state) != Counter({f: 9 for f in 'URFDLB'})
                or state in seen or type(target) is not int or not 0 <= target < 18
                or ACTIONS[target] != row['target_action']
                or type(row['optimal_distance']) is not int or not 1 <= row['optimal_distance'] <= 3
                or not row['solution'] or row['solution'][0] != row['target_action']
                or len(row['solution']) != row['optimal_distance']
                or any(a not in ACTIONS for a in row['solution'])):
            raise ValueError(f'Invalid/duplicate training row at line {line_number}')
        # Rebuild the input from state alone: metadata cannot leak into the prompt.
        row['input'] = '\n'.join(f'{f}: {state[i*9:i*9+9]}' for i, f in enumerate('URFDLB')) + '\nAction:'
        rows.append(row)
        seen.add(state)
    if not rows:
        raise ValueError('Empty training set')
    return rows


def train_epoch(body, head, tokens, labels, optimizer, batch_size, device):
    body.train()
    head.train()
    order = torch.randperm(len(labels))
    total = 0.0
    for indices in order.split(batch_size):
        inputs = {k: v[indices].to(device) for k, v in tokens.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            logits = action_logits(body, head, inputs)
            loss = torch.nn.functional.cross_entropy(logits.float(), labels[indices].to(device))
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(body.parameters()) + list(head.parameters()),
                                      1.0, error_if_nonfinite=True)
        optimizer.step()
        total += float(loss.detach()) * len(indices)
    return total / len(labels)


@torch.no_grad()
def evaluate(body, head, tokens, labels, depths, batch_size, device):
    body.eval()
    head.eval()
    total, predictions = 0.0, []
    for start in range(0, len(labels), batch_size):
        stop = start + batch_size
        inputs = {k: v[start:stop].to(device) for k, v in tokens.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            logits = action_logits(body, head, inputs).float()
        if not torch.isfinite(logits).all():
            raise RuntimeError('Nonfinite evaluation logits')
        total += float(torch.nn.functional.cross_entropy(logits, labels[start:stop].to(device), reduction='sum'))
        predictions.extend(logits.argmax(-1).cpu().tolist())
    correct = [a == b for a, b in zip(predictions, labels.tolist())]
    by_depth = {str(d): sum(c for c, x in zip(correct, depths) if x == d) / depths.count(d)
                for d in sorted(set(depths))}
    return {'train_loss': total / len(labels), 'train_label_accuracy': sum(correct) / len(labels),
            'train_label_accuracy_by_depth': by_depth}


def save_model(body, head, path):
    path.mkdir(parents=True, exist_ok=False)
    body.save_pretrained(path / 'backbone', safe_serialization=True)
    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, path / 'action_head.pt')
    (path / 'actions.json').write_text(json.dumps(list(ACTIONS), indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('data/cube_shallow_256.jsonl'))
    p.add_argument('--output', type=Path, default=Path('runs/cube-overfit-256'))
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--backbone-lr', type=float, default=1e-5)
    p.add_argument('--head-lr', type=float, default=1e-4)
    p.add_argument('--stop-accuracy', type=float, default=0.99)
    p.add_argument('--seed', type=int, default=17)
    args = p.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or any(not math.isfinite(x) or x <= 0 for x in (args.backbone_lr, args.head_lr)):
        p.error('Batch size, epochs and learning rates must be positive and finite')
    if not 0 < args.stop_accuracy <= 1:
        p.error('--stop-accuracy must be in (0, 1]')
    if args.output.exists():
        p.error('Output already exists; choose another --output to preserve previous results')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        p.error('Requires a CUDA GPU with BF16 support')
    import transformers
    from transformers import AutoTokenizer, Qwen3_5Model
    torch.manual_seed(args.seed)
    rows = read_data(args.data)
    cfg = json.loads((args.model / 'config.json').read_text())
    if cfg.get('model_type') != 'qwen3_5':
        p.error('Expected a full Qwen3.5 dense model directory')
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side='right')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer([r['input'] for r in rows], padding=True, truncation=False,
                       return_tensors='pt', return_token_type_ids=False)
    if tokens['input_ids'].shape[1] > min(512, cfg['text_config']['max_position_embeddings']):
        raise ValueError('Unexpected input length; refusing to truncate')
    labels = torch.tensor([r['target_index'] for r in rows])
    depths = [r['optimal_distance'] for r in rows]
    print(f'Loading {args.model}; samples={len(rows)}, max tokens={tokens["input_ids"].shape[1]}', flush=True)
    full, info = Qwen3_5Model.from_pretrained(args.model, local_files_only=True,
        dtype=torch.float32, attn_implementation='sdpa', output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise RuntimeError(f'Incomplete backbone loading: {info}')
    body = full.language_model
    del full
    body = body.to('cuda')
    body.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    head = torch.nn.Linear(body.config.hidden_size, 18).to('cuda')
    optimizer = torch.optim.AdamW([{'params': body.parameters(), 'lr': args.backbone_lr},
                                  {'params': head.parameters(), 'lr': args.head_lr}])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'run.json').write_text(json.dumps({**vars(args), 'actions': list(ACTIONS),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'gpu': torch.cuda.get_device_name(), 'objective': 'single-expert-label cross entropy',
        'scope': 'training-set fit only, NOT held-out accuracy or solve rate'}, default=str, indent=2) + '\n')
    # Save the exact input data used, so the run remains interpretable after data changes.
    (args.output / 'training_data.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    start = time.monotonic()
    with (args.output / 'metrics.jsonl').open('w') as log:
        for epoch in range(args.epochs + 1):
            if epoch:
                train_epoch(body, head, tokens, labels, optimizer, args.batch_size, 'cuda')
            metrics = {'epoch': epoch, **evaluate(body, head, tokens, labels, depths, args.batch_size, 'cuda'),
                       'elapsed_seconds': round(time.monotonic() - start, 2),
                       'peak_vram_gib': round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            line = json.dumps(metrics)
            print(line, flush=True)
            log.write(line + '\n')
            log.flush()
            if metrics['train_label_accuracy'] >= args.stop_accuracy:
                break
    save_model(body, head, args.output / 'checkpoint')
    tokenizer.save_pretrained(args.output / 'checkpoint/tokenizer')
    (args.output / 'summary.json').write_text(json.dumps({**metrics,
        'target_reached': metrics['train_label_accuracy'] >= args.stop_accuracy,
        'checkpoint': 'final epoch, inference/warm-start weights; no optimizer resume state'}, indent=2) + '\n')
    print(f'Saved {args.output}/checkpoint. Training label accuracy is NOT cube solve rate.', flush=True)


if __name__ == '__main__':
    main()
