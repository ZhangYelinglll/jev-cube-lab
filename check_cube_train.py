"""CPU regression check; no downloaded weights required."""
import json
import tempfile
from pathlib import Path

import torch
from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
from cube_smoke import ACTIONS, action_logits
from cube_train import read_data, train_epoch, evaluate, save_model


def main():
    torch.manual_seed(17)
    rows = read_data(Path(__file__).parent / 'data/cube_shallow_256.jsonl')
    assert len(rows) == 256
    with tempfile.TemporaryDirectory() as tmp:
        bad = dict(rows[0], target_index=17)
        path = Path(tmp) / 'bad.jsonl'
        path.write_text(json.dumps(bad) + '\n')
        try:
            read_data(path)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid action mapping was accepted')
        cfg = Qwen3_5TextConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=8, linear_value_head_dim=8,
            layer_types=['linear_attention', 'full_attention'], pad_token_id=0,
            rope_parameters={'rope_type':'default','rope_theta':10000,
                             'partial_rotary_factor':1.0,'mrope_section':[2,3,3]})
        body = Qwen3_5TextModel(cfg).float()
        head = torch.nn.Linear(32, 18)
        tokens = {'input_ids':torch.tensor([[1,2,3],[4,5,0],[6,7,8]]),
                  'attention_mask':torch.tensor([[1,1,1],[1,1,0],[1,1,1]])}
        labels = torch.tensor([1,0,2])
        depths = [1,2,3]
        optimizer = torch.optim.AdamW(list(body.parameters()) + list(head.parameters()), lr=0.005)
        before = evaluate(body, head, tokens, labels, depths, 2, 'cpu')
        body_before = next(body.layers[0].parameters()).detach().clone()
        for _ in range(8):
            train_epoch(body, head, tokens, labels, optimizer, 2, 'cpu')
        after = evaluate(body, head, tokens, labels, depths, 2, 'cpu')
        assert after['train_loss'] < before['train_loss'], (before, after)
        assert not torch.equal(body_before, next(body.layers[0].parameters()))
        save_model(body, head, Path(tmp) / 'checkpoint')
        restored = Qwen3_5TextModel.from_pretrained(Path(tmp) / 'checkpoint/backbone').eval()
        restored_head = torch.nn.Linear(32, 18)
        restored_head.load_state_dict(torch.load(Path(tmp) / 'checkpoint/action_head.pt', weights_only=True))
        with torch.no_grad():
            torch.testing.assert_close(action_logits(body, head, tokens),
                                       action_logits(restored, restored_head, tokens))
        assert json.loads((Path(tmp) / 'checkpoint/actions.json').read_text()) == list(ACTIONS)
        print('PASS: data validation, actual hybrid-model training, loss decrease and checkpoint reload parity')
        print(before, after)


if __name__ == '__main__':
    main()
