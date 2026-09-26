import sys
from pathlib import Path
import unittest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor"))
import train
from dassl.config import get_cfg_default

class ConfigurationTests(unittest.TestCase):
    def test_all_configs_merge(self):
        for path in (ROOT / "configs").rglob("*.yaml"):
            with self.subTest(config=str(path.relative_to(ROOT))):
                cfg = get_cfg_default()
                train.extend_cfg(cfg)
                cfg.merge_from_file(str(path))

    def test_dataset_cli_and_paper_parameters(self):
        args = train.build_parser().parse_args([
            "--dataset", "BTMRI", "--root", "/tmp/data", "--seed", "0",
            "--config-file", str(ROOT / "configs/base_to_novel.yaml"),
        ])
        cfg = train.setup_cfg(args)
        self.assertEqual(cfg.DATASET.NAME, "BTMRI")
        self.assertEqual(cfg.DATASET.SUBSAMPLE_CLASSES, "base")
        self.assertEqual(cfg.SEED, 0)
        self.assertEqual(cfg.OPTIM.LR, 0.0025)
        self.assertEqual(cfg.DATALOADER.TRAIN_X.BATCH_SIZE, 4)
        self.assertEqual(cfg.OPTIM.MAX_EPOCH, 50)
        self.assertFalse((ROOT / "configs/datasets").exists())
        self.assertEqual(cfg.TRAINER.SAMOE.N_CTX, 8)
        self.assertEqual(cfg.TRAINER.SAMOE.PROMPT_DEPTH, 9)
        self.assertEqual(cfg.TRAINER.SAMOE.LAMBDA_BALANCE, 0.1)

from trainers.samoe import StyleRouter, ExpertBank, routing_balance_loss

class RouterTests(unittest.TestCase):
    def test_batch_composition_does_not_change_routing(self):
        torch.manual_seed(7)
        router = StyleRouter(8, 4).eval()
        tokens = torch.randn(3, 6, 8)
        weights, ids, _, mass = router(tokens)
        w_single, i_single, _, _ = router(tokens[:1])
        torch.testing.assert_close(weights[:1], w_single)
        torch.testing.assert_close(ids[:1], i_single)
        torch.testing.assert_close(mass.sum(-1), torch.ones(3))
        self.assertTrue(((mass > 0).sum(-1) == 2).all())

    def test_sparse_expert_gradients(self):
        torch.manual_seed(2)
        bank = ExpertBank(4, 8, 4, torch.float32)
        prompt = torch.randn(2, 3, 4, requires_grad=True)
        weights = torch.tensor([[0.4, 0.6], [0.7, 0.3]], requires_grad=True)
        ids = torch.tensor([[0, 2], [0, 2]])
        output = bank(prompt, weights, ids)
        self.assertEqual(output.shape, (2, 3, 8))
        output.square().sum().backward()
        self.assertGreater(prompt.grad.abs().sum().item(), 0)
        self.assertEqual(bank.routed_experts.grad[[1, 3]].abs().sum().item(), 0)
        self.assertGreater(bank.routed_experts.grad[[0, 2]].abs().sum().item(), 0)

    def test_balanced_and_collapsed_routing(self):
        uniform = routing_balance_loss([torch.full((4, 4), 0.25)], 4)
        collapsed = routing_balance_loss([torch.tensor([[1., 0., 0., 0.]])], 4)
        self.assertEqual(uniform.item(), 0)
        self.assertGreater(collapsed.item(), uniform.item())

class PromptInjectionTests(unittest.TestCase):
    def test_layerwise_injection_and_removal(self):
        from clip_maple.model import CLIP
        from trainers.samoe import CustomCLIP
        cfg = get_cfg_default()
        train.extend_cfg(cfg)
        cfg.INPUT.SIZE = (32, 32)
        cfg.TRAINER.SAMOE.PROMPT_DEPTH = 2
        backbone = CLIP(
            embed_dim=32, image_resolution=32, vision_layers=3,
            vision_width=64, vision_patch_size=16, context_length=77,
            vocab_size=49408, transformer_width=64, transformer_heads=1,
            transformer_layers=3,
            design_details={"trainer": "MaPLe", "vision_depth": 0,
                            "language_depth": 0, "vision_ctx": 0,
                            "language_ctx": 0, "maple_length": 8},
        ).float()
        model = CustomCLIP(cfg, ["normal brain", "glioma tumor"], backbone)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("prompt_learner."))
        self.assertEqual(len(model.prompt_learner.expert_banks), 2)
        ids = train.trainers.samoe.clip.tokenize("a photo of a")
        expected = backbone.token_embedding(ids)[0, 1:5]
        torch.testing.assert_close(model.prompt_learner.ctx[:4], expected)
        text_lengths, visual_lengths = [], []
        handles = []
        for block in model.text_encoder.transformer.resblocks:
            handles.append(block.attn.register_forward_pre_hook(
                lambda module, args: text_lengths.append(args[0].shape[0])))
        for block in model.image_encoder.transformer.resblocks:
            handles.append(block.attn.register_forward_pre_hook(
                lambda module, args: visual_lengths.append(args[0].shape[0])))
        images = torch.randn(2, 3, 32, 32)
        model.train()
        loss, _, _ = model(images, torch.tensor([0, 1]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(text_lengths, [85, 85, 77])
        self.assertEqual(visual_lengths, [13, 13, 5])
        for bank in model.prompt_learner.expert_banks:
            self.assertGreater(bank.routed_experts.grad.abs().sum().item(), 0)
        self.assertGreater(model.prompt_learner.router.gate[0].weight.grad.abs().sum().item(), 0)
        for h in handles:
            h.remove()
        model.eval()
        with torch.no_grad():
            logits, _, _, routing = model(images, return_features=True)
        self.assertEqual(logits.shape, (2, 2))
        self.assertEqual(len(routing["masses"]), 2)

if __name__ == "__main__":
    unittest.main()
