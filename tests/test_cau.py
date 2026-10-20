import unittest
from types import SimpleNamespace
import torch
from cau.data import PairedCollator, encode_record
from cau.objective import ConflictObjective, masked_activation_mse, select_parameters
from cau.select import select_retain


class Tokenizer:
    chat_template = None
    eos_token_id = 1

    def __call__(self, text, **kwargs):
        return {"input_ids": [2 + ord(c) % 20 for c in text]}


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(32, 4)
        self.project = torch.nn.Linear(4, 4)
        self.head = torch.nn.Linear(4, 32)

    def forward(self, input_ids, attention_mask, labels):
        logits = self.head(self.project(self.embed(input_ids)))
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 32), labels[:, 1:].reshape(-1), ignore_index=-100
        )
        return SimpleNamespace(loss=loss, logits=logits)


class CAUTests(unittest.TestCase):
    def test_embedding_mapping_consumes_duplicate_rows(self):
        import numpy as np
        from cau.select_embeddings import select_by_embedding

        row = {"instruction": "q", "input": "x", "output": "0"}
        other = dict(row, input="far")
        kept = select_by_embedding(
            [row, row, other], [row], [row, other], np.array([[0.0], [1.0], [10.0]]), 50
        )
        self.assertEqual(kept, [other])
        with self.assertRaises(ValueError):
            select_by_embedding([row, other], [row], [row, other], np.array([[0.0], [10.0]]), 50)

    def test_masked_loss_has_no_cross_example_broadcast(self):
        activation = torch.tensor([[[1.0], [999.0]], [[3.0], [3.0]]])
        mask = torch.tensor([[True, False], [True, True]])
        self.assertEqual(masked_activation_mse(activation, torch.zeros(1, 1, 1), mask).item(), 5.0)
        with self.assertRaises(ValueError):
            masked_activation_mse(activation, torch.zeros_like(activation), torch.zeros_like(mask))

    def test_selection_removes_near_records(self):
        forget = [{"input": f"Patient information: age is {v}."} for v in [0, 2]]
        retain = [{"input": f"Patient information: age is {v}."} for v in [1, 10]]
        selected, _ = select_retain(forget, retain)
        self.assertEqual(selected, retain[1:])

    def test_feature_mask_preserves_other_fields(self):
        forget = [{"input": f"Patient information: age is {v}, score is {v}."} for v in [1, 3]]
        retain = [
            {"input": f"Patient information: age is {v + 100}, score is {v}.", "output": "yes"}
            for v in [1, 3]
        ]
        selected, report = select_retain(forget, retain, mode="feature")
        self.assertEqual(report["masked_columns"], ["age"])
        self.assertIn("age is unknown", selected[0]["input"])
        self.assertIn("score is 1", selected[0]["input"])
        self.assertEqual(selected[0]["output"], "yes")
        self.assertNotIn("unknown", retain[0]["input"])

    def test_truncation_retains_answer_and_padding_is_ignored(self):
        tok = Tokenizer()
        a = encode_record({"input": "long " * 100, "output": "yes"}, tok, 12)
        b = encode_record({"input": "q", "output": "n"}, tok, 8)
        self.assertEqual(len(a["input_ids"]), 12)
        self.assertEqual(a["labels"][-4:], [tok("yes")["input_ids"][i] for i in range(3)] + [1])
        batch = PairedCollator(0)([{"forget": a, "retain": b}, {"forget": b, "retain": a}])
        self.assertTrue(torch.all(batch["forget"]["labels"][1, 8:] == -100))

    def test_only_selected_parameters_update_and_hooks_removed(self):
        model = ToyModel()
        select_parameters(model, [r"project\.weight"])
        objective = ConflictObjective(model, "project")
        encoded = encode_record({"input": "q", "output": "yes"}, Tokenizer(), 16)
        batch = PairedCollator(0)([{"forget": encoded, "retain": encoded}])
        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
        loss, _ = objective(model, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        opt.step()
        changed = [k for k, v in model.named_parameters() if not torch.equal(v, before[k])]
        self.assertEqual(changed, ["project.weight"])
        self.assertEqual(len(model.project._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
