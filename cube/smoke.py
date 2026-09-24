#!/usr/bin/env python3
"""Qwen3.5-0.8B-Base + 18-action head: one GPU training smoke test.

Python >=3.10; install CUDA-enabled torch and transformers==5.3.0.
python -m cube.smoke --model /path/to/Qwen3.5-0.8B-Base
python -m cube.smoke --self-test  # tiny random CPU model; no downloads
No weights are saved or uploaded. This is NOT a cube-solving benchmark.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

# Generated and inverse-verified using the web project's cubejs 1.3.2.
# Facelet order URFDLB; each face is row-major viewed from outside.
# Bundled fixtures keep this standalone check independent of JS or a solver.
STATES = {
    'U': 'UUUUUUUUUBBBRRRRRRRRRFFFFFFDDDDDDDDDFFFLLLLLLLLLBBBBBB',
    "U'": 'UUUUUUUUUFFFRRRRRRLLLFFFFFFDDDDDDDDDBBBLLLLLLRRRBBBBBB',
    'U2': 'UUUUUUUUULLLRRRRRRBBBFFFFFFDDDDDDDDDRRRLLLLLLFFFBBBBBB',
    'R': 'UUFUUFUUFRRRRRRRRRFFDFFDFFDDDBDDBDDBLLLLLLLLLUBBUBBUBB',
    "R'": 'UUBUUBUUBRRRRRRRRRFFUFFUFFUDDFDDFDDFLLLLLLLLLDBBDBBDBB',
    'R2': 'UUDUUDUUDRRRRRRRRRFFBFFBFFBDDUDDUDDULLLLLLLLLFBBFBBFBB',
    'F': 'UUUUUULLLURRURRURRFFFFFFFFFRRRDDDDDDLLDLLDLLDBBBBBBBBB',
    "F'": 'UUUUUURRRDRRDRRDRRFFFFFFFFFLLLDDDDDDLLULLULLUBBBBBBBBB',
    'F2': 'UUUUUUDDDLRRLRRLRRFFFFFFFFFUUUDDDDDDLLRLLRLLRBBBBBBBBB',
    'D': 'UUUUUUUUURRRRRRFFFFFFFFFLLLDDDDDDDDDLLLLLLBBBBBBBBBRRR',
    "D'": 'UUUUUUUUURRRRRRBBBFFFFFFRRRDDDDDDDDDLLLLLLFFFBBBBBBLLL',
    'D2': 'UUUUUUUUURRRRRRLLLFFFFFFBBBDDDDDDDDDLLLLLLRRRBBBBBBFFF',
    'L': 'BUUBUUBUURRRRRRRRRUFFUFFUFFFDDFDDFDDLLLLLLLLLBBDBBDBBD',
    "L'": 'FUUFUUFUURRRRRRRRRDFFDFFDFFBDDBDDBDDLLLLLLLLLBBUBBUBBU',
    'L2': 'DUUDUUDUURRRRRRRRRBFFBFFBFFUDDUDDUDDLLLLLLLLLBBFBBFBBF',
    'B': 'RRRUUUUUURRDRRDRRDFFFFFFFFFDDDDDDLLLULLULLULLBBBBBBBBB',
    "B'": 'LLLUUUUUURRURRURRUFFFFFFFFFDDDDDDRRRDLLDLLDLLBBBBBBBBB',
    'B2': 'DDDUUUUUURRLRRLRRLFFFFFFFFFDDDDDDUUURLLRLLRLLBBBBBBBBB',
}
ACTIONS = tuple(STATES)


def samples():
    texts, labels = [], []
    for move, state in STATES.items():
        if Counter(state) != Counter({f: 9 for f in 'URFDLB'}):
            raise ValueError(f'Invalid fixture: {move}')
        inverse = move if move.endswith('2') else move[0] if move.endswith("'") else move + "'"
        texts.append('\n'.join(f'{f}: {state[i*9:i*9+9]}' for i, f in enumerate('URFDLB')) + '\nAction:')
        labels.append(ACTIONS.index(inverse))
    return texts, labels


def action_logits(backbone, head, inputs):
    hidden = backbone(**inputs, use_cache=False).last_hidden_state
    # Right padding: pick the final real token, never a padding token.
    last = inputs['attention_mask'].sum(-1) - 1
    import torch
    return head(hidden[torch.arange(hidden.shape[0], device=hidden.device), last].float())


def checked_update(backbone, head, inputs, labels, optimizer, amp=False):
    import torch
    optimizer.zero_grad(set_to_none=True)
    before_head = head.weight.detach().clone()
    # Check a real transformer weight, not only the new output head.
    probe = next(backbone.layers[0].parameters())
    before_body = probe.detach().clone()
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
        logits = action_logits(backbone, head, inputs)
        loss = torch.nn.functional.cross_entropy(logits.float(), labels)
    if logits.shape != (len(labels), 18) or not torch.isfinite(loss):
        raise RuntimeError('Invalid output shape or nonfinite loss')
    loss.backward()
    norms = {}
    for name, module in [('backbone', backbone), ('head', head)]:
        grads = [p for p in module.parameters() if p.grad is not None]
        if not grads:
            raise RuntimeError(f'{name}: missing gradients')
        norm = torch.nn.utils.clip_grad_norm_(grads, 1.0, error_if_nonfinite=True)
        norms[name] = float(norm)
        if norms[name] <= 0:
            raise RuntimeError(f'{name}: zero gradients')
    optimizer.step()
    if torch.equal(before_head, head.weight) or torch.equal(before_body, probe):
        raise RuntimeError('Optimizer did not update both backbone and head')
    probs = logits.detach().float().softmax(-1)
    if not torch.allclose(probs.sum(-1), torch.ones(len(labels), device=labels.device), atol=1e-5):
        raise RuntimeError('Invalid probability normalization')
    return {'loss': float(loss.detach()), 'shape': list(logits.shape), 'gradient_norms': norms,
            'first_target': ACTIONS[int(labels[0])],
            'first_prediction_before_update': ACTIONS[int(probs[0].argmax())]}


def self_test():
    import torch
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
    torch.manual_seed(17)
    texts, labels = samples()
    assert len(set(texts)) == 18 and sorted(labels) == list(range(18))
    assert labels[:3] == [1, 0, 2]
    # Exercise BOTH DeltaNet and full attention, not a substitute architecture.
    config = Qwen3_5TextConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8,
        layer_types=['linear_attention', 'full_attention'],
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000,
                         'partial_rotary_factor': 1.0, 'mrope_section': [2, 3, 3]},
        pad_token_id=0)
    backbone = Qwen3_5TextModel(config).float().train()
    head = torch.nn.Linear(32, 18)
    inputs = {'input_ids': torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]]),
              'attention_mask': torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])}
    backbone.eval()
    with torch.no_grad():
        batched = action_logits(backbone, head, inputs)[1]
        single = action_logits(backbone, head, {k: v[1:2, :2] for k, v in inputs.items()})[0]
        torch.testing.assert_close(batched, single, atol=1e-5, rtol=1e-4)
    backbone.train()
    opt = torch.optim.AdamW([{'params': backbone.parameters(), 'lr': 1e-5},
                            {'params': head.parameters(), 'lr': 1e-4}])
    result = checked_update(backbone, head, inputs, torch.tensor([1, 0]), opt)
    print(json.dumps(result, indent=2))
    print('PASS: tiny CPU forward, padding, backward and optimizer update (not pretrained weights).')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, help='Local Qwen3.5-0.8B-Base directory')
    parser.add_argument('--batch-size', type=int, default=2, choices=range(1, 19), metavar='1..18')
    parser.add_argument('--steps', type=int, default=1, help='Smoke updates, not an accuracy target')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.model or not (args.model / 'config.json').is_file():
        parser.error('--model must be a local model directory containing config.json')
    if args.steps < 1:
        parser.error('--steps must be positive')
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5Model
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A CUDA GPU with BF16 support is required; CPU check: --self-test')
    config = json.loads((args.model / 'config.json').read_text())
    if config.get('model_type') != 'qwen3_5':
        raise ValueError('Expected a full Qwen3.5 dense checkpoint (model_type=qwen3_5)')
    torch.manual_seed(17)
    print(f'torch={torch.__version__}, transformers={transformers.__version__}, GPU={torch.cuda.get_device_name(0)}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side='right')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts, targets = samples()
    tokens = tokenizer(texts, padding=True, truncation=False, return_tensors='pt',
                       return_token_type_ids=False)
    length = tokens['input_ids'].shape[1]
    if length > min(512, config['text_config']['max_position_embeddings']):
        raise ValueError(f'Unexpected token length {length}; refusing to truncate')
    print(f'18 verified single-turn fixtures; max tokens={length}\nExample input:\n{texts[0]}\nLabel: {ACTIONS[targets[0]]}', flush=True)
    # Load the full body to preserve checkpoint key mapping; release vision immediately.
    # FP32 trainable parameters + BF16 autocast keep small optimizer updates reliable.
    body, info = Qwen3_5Model.from_pretrained(args.model, local_files_only=True,
        dtype=torch.float32, attn_implementation='sdpa', output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise RuntimeError(f'Incomplete backbone loading: {info}')
    backbone = body.language_model
    del body
    backbone = backbone.to('cuda').train()
    head = torch.nn.Linear(backbone.config.hidden_size, 18).to('cuda').train()
    optimizer = torch.optim.AdamW([{'params': backbone.parameters(), 'lr': 1e-5},
                                  {'params': head.parameters(), 'lr': 1e-4}])
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        idx = torch.tensor([(step * args.batch_size + i) % 18 for i in range(args.batch_size)])
        inputs = {k: v[idx].to('cuda') for k, v in tokens.items()}
        labels = torch.tensor([targets[i] for i in idx.tolist()], device='cuda')
        result = checked_update(backbone, head, inputs, labels, optimizer, amp=True)
        print(json.dumps({'step': step + 1, **result}), flush=True)
    torch.cuda.synchronize()
    print(f'Peak allocated VRAM: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB')
    print('PASS: local weights loaded; 18-action output, finite nonzero gradients, backbone/head updated.')
    print('This verifies the training pipeline only. No solving accuracy claim; no weights saved.')


if __name__ == '__main__':
    main()
