"""Style-aware mixture-of-experts multimodal prompt learning."""
import os.path as osp

import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_checkpoint, load_pretrained_weights
import clip_maple.clip as clip

TOP_K = 2


def load_clip_to_cpu(cfg):
    """Load the MaPLe-compatible OpenAI CLIP backbone."""
    backbone_name = cfg.MODEL.BACKBONE.NAME
    model_path = clip._download(clip._MODELS[backbone_name])

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "trainer": "MaPLe",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
        "maple_length": cfg.TRAINER.SAMOE.N_CTX,
    }
    return clip.build_model(state_dict or model.state_dict(), design_details)


class StyleRouter(nn.Module):
    """A single router reused at every prompted visual layer."""

    def __init__(self, vis_dim, num_experts, hidden_dim=256):
        super().__init__()
        if num_experts < TOP_K:
            raise ValueError(
                f"SaMoE requires NUM_EXPERTS >= {TOP_K}, got {num_experts}"
            )

        self.num_experts = num_experts
        self.top_k = TOP_K
        self.gate = nn.Sequential(
            nn.Linear(3 * vis_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts),
        )
        self.routing_bias = nn.Parameter(torch.zeros(num_experts))

    @staticmethod
    def style_descriptor(visual_tokens, global_token=None):
        """Return per-image [patch mean, patch variance, global] statistics.

        OpenAI CLIP supplies the global feature as its first (CLS) token. A
        caller adapting an architecture without CLS, such as SigLIP, can pass
        its native pooled global feature explicitly.
        """
        if visual_tokens.ndim != 3:
            raise ValueError(
                "visual_tokens must have shape (batch, tokens, width)"
            )
        if global_token is None:
            if visual_tokens.shape[1] < 2:
                raise ValueError(
                    "CLIP visual_tokens must contain CLS and patch tokens"
                )
            patches = visual_tokens[:, 1:, :]
            global_token = visual_tokens[:, 0, :]
        else:
            patches = visual_tokens
            if global_token.ndim == 3 and global_token.shape[1] == 1:
                global_token = global_token[:, 0, :]
            if global_token.shape != patches.shape[:1] + patches.shape[2:]:
                raise ValueError(
                    "global_token must have shape (batch, visual_width)"
                )
        mean = patches.mean(dim=1)
        variance = patches.var(dim=1, unbiased=False)
        return torch.cat([mean, variance, global_token], dim=-1)

    def forward(self, visual_tokens, global_token=None):
        descriptor = self.style_descriptor(
            visual_tokens, global_token=global_token
        )
        descriptor = descriptor.to(dtype=self.gate[0].weight.dtype)
        logits = self.gate(descriptor)
        scores = torch.sigmoid(logits + self.routing_bias)
        selected_scores, selected_indices = scores.topk(TOP_K, dim=-1)
        selected_weights = selected_scores / selected_scores.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        # alpha_m^(l)(x): selected normalized mass, zero for unselected experts.
        routing_mass = scores.new_zeros(scores.shape)
        routing_mass.scatter_(1, selected_indices, selected_weights)
        return selected_weights, selected_indices, scores, routing_mass


class ExpertBank(nn.Module):
    """One shared projection and M independently routed projections."""

    def __init__(self, text_dim, vis_dim, num_experts, dtype):
        super().__init__()
        self.text_dim = text_dim
        self.vis_dim = vis_dim
        self.shared_expert = nn.Linear(
            text_dim, vis_dim, bias=False, dtype=dtype
        )
        self.routed_experts = nn.Parameter(
            torch.empty(num_experts, vis_dim, text_dim, dtype=dtype)
        )
        nn.init.normal_(self.routed_experts, std=0.02)

    def forward(self, text_prompt, selected_weights, selected_indices):
        """Compose a different visual prompt for every image in the batch."""
        batch_size, _, _ = text_prompt.shape
        selected = self.routed_experts.index_select(
            0, selected_indices.reshape(-1)
        )
        selected = selected.reshape(
            batch_size, TOP_K, self.vis_dim, self.text_dim
        )
        routed = torch.einsum("bkvt,bnt->bknv", selected, text_prompt)
        routed = (
            routed
            * selected_weights.to(routed.dtype)[:, :, None, None]
        ).sum(dim=1)
        return self.shared_expert(text_prompt) + routed


def routing_balance_loss(routing_masses, num_experts):
    r"""Layer-averaged balance penalty on normalized routing mass.

    L_bal = (1/|L|) sum_l M sum_m
            ( (1/B) sum_b alpha_m^(l)(x_b) - 1/M )^2.
    """
    if not routing_masses:
        raise ValueError("routing_masses must contain at least one layer")

    target = 1.0 / num_experts
    layer_losses = []
    for alpha in routing_masses:
        mean_mass = alpha.mean(dim=0)
        layer_losses.append(
            num_experts * ((mean_mass - target) ** 2).sum()
        )
    return torch.stack(layer_losses).mean()


def transformer_block(block, tokens, causal=False):
    """Apply a frozen CLIP block to an optionally extended token sequence."""
    mask = None
    if causal:
        mask = torch.full(
            (tokens.shape[0], tokens.shape[0]), float("-inf"),
            dtype=tokens.dtype, device=tokens.device,
        ).triu_(1)
    normalized = block.ln_1(tokens)
    attended = block.attn(
        normalized, normalized, normalized,
        need_weights=False, attn_mask=mask,
    )[0]
    tokens = tokens + attended
    return tokens + block.mlp(block.ln_2(tokens))


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, embeddings, token_ids, prompts):
        tokens = embeddings + self.positional_embedding.to(self.dtype)
        tokens = tokens.permute(1, 0, 2)
        for layer, block in enumerate(self.transformer.resblocks):
            if layer < len(prompts):
                context = prompts[layer].to(tokens.dtype)
                context = context[:, None, :].expand(-1, tokens.shape[1], -1)
                tokens = transformer_block(
                    block, torch.cat([context, tokens], dim=0), causal=True
                )[context.shape[0]:]
            else:
                tokens = transformer_block(block, tokens, causal=True)
        tokens = self.ln_final(tokens.permute(1, 0, 2)).to(self.dtype)
        eos = token_ids.argmax(dim=-1)
        return tokens[torch.arange(tokens.shape[0], device=tokens.device), eos] @ self.text_projection


class SaMoEPromptLearner(nn.Module):
    """Layer-specific prompts and expert banks with one shared style router."""
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        settings = cfg.TRAINER.SAMOE
        self.n_cls = len(classnames)
        self.n_ctx = settings.N_CTX
        self.prompt_depth = settings.PROMPT_DEPTH
        self.num_experts = settings.NUM_EXPERTS
        text_dim = clip_model.ln_final.weight.shape[0]
        visual_dim = clip_model.visual.conv1.out_channels
        dtype = clip_model.dtype
        max_depth = min(
            len(clip_model.transformer.resblocks),
            len(clip_model.visual.transformer.resblocks),
        )
        if not 1 <= self.prompt_depth <= max_depth:
            raise ValueError(f"PROMPT_DEPTH must lie in [1, {max_depth}]")
        if cfg.INPUT.SIZE[0] != clip_model.visual.input_resolution:
            raise ValueError("INPUT.SIZE must match CLIP input resolution")

        context = torch.empty(self.n_ctx, text_dim, dtype=dtype)
        nn.init.normal_(context, std=0.02)
        if settings.CTX_INIT:
            ids = clip.tokenize(settings.CTX_INIT.replace("_", " "))
            n_init = min(self.n_ctx, int(ids[0].argmax()) - 1)
            with torch.no_grad():
                context[:n_init] = clip_model.token_embedding(ids)[0, 1:1 + n_init].to(dtype)
        self.ctx = nn.Parameter(context)
        self.deep_text_prompts = nn.ParameterList([
            nn.Parameter(torch.empty(self.n_ctx, text_dim, dtype=dtype))
            for _ in range(self.prompt_depth - 1)
        ])
        for prompt in self.deep_text_prompts:
            nn.init.normal_(prompt, std=0.02)
        self.router = StyleRouter(visual_dim, self.num_experts, settings.ROUTER_HIDDEN)
        self.expert_banks = nn.ModuleList([
            ExpertBank(text_dim, visual_dim, self.num_experts, dtype)
            for _ in range(self.prompt_depth)
        ])

        token_ids = clip.tokenize([
            name.replace("_", " ") + "." for name in classnames
        ])
        with torch.no_grad():
            embeddings = clip_model.token_embedding(token_ids).to(dtype)
        self.register_buffer("tokenized_prompts", token_ids)
        self.register_buffer("token_embeddings", embeddings)

    def text_inputs(self):
        return self.token_embeddings, self.tokenized_prompts, [self.ctx, *self.deep_text_prompts]

    def compose_visual_prompt(self, layer, visual_tokens):
        weights, indices, scores, mass = self.router(visual_tokens)
        context = self.ctx if layer == 0 else self.deep_text_prompts[layer - 1]
        context = context.unsqueeze(0).expand(visual_tokens.shape[0], -1, -1)
        prompt = self.expert_banks[layer](context, weights, indices)
        return prompt, weights, indices, scores, mass


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = SaMoEPromptLearner(cfg, classnames, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.lambda_balance = cfg.TRAINER.SAMOE.LAMBDA_BALANCE

    def encode_image(self, image):
        encoder = self.image_encoder
        tokens = encoder.conv1(image.to(self.dtype))
        tokens = tokens.flatten(2).permute(0, 2, 1)
        cls = encoder.class_embedding.to(tokens.dtype)[None, None, :]
        cls = cls.expand(tokens.shape[0], 1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = encoder.ln_pre(tokens + encoder.positional_embedding.to(tokens.dtype))
        tokens = tokens.permute(1, 0, 2)
        routing = {key: [] for key in ["weights", "indices", "scores", "masses"]}
        for layer, block in enumerate(encoder.transformer.resblocks):
            if layer < self.prompt_learner.prompt_depth:
                prompt, weights, indices, scores, mass = self.prompt_learner.compose_visual_prompt(
                    layer, tokens.permute(1, 0, 2)
                )
                prompt = prompt.to(tokens.dtype).permute(1, 0, 2)
                tokens = transformer_block(
                    block, torch.cat([prompt, tokens], dim=0)
                )[prompt.shape[0]:]
                for key, value in zip(routing, [weights, indices, scores, mass]):
                    routing[key].append(value)
            else:
                tokens = transformer_block(block, tokens)
        features = encoder.ln_post(tokens[0])
        if encoder.proj is not None:
            features = features @ encoder.proj
        return features, routing

    def forward(self, image, label=None, return_features=False):
        image_features, routing = self.encode_image(image)
        text_features = self.text_encoder(*self.prompt_learner.text_inputs())
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        logits = self.logit_scale.exp() * image_features @ text_features.t()
        if return_features:
            return logits, image_features, text_features, routing
        if self.training:
            if label is None:
                raise ValueError("Training requires labels")
            ce_loss = F.cross_entropy(logits, label)
            balance_loss = routing_balance_loss(routing["masses"], self.prompt_learner.num_experts)
            return ce_loss + self.lambda_balance * balance_loss, ce_loss, balance_loss
        return logits


@TRAINER_REGISTRY.register()
class SaMoE_CLIP(TrainerX):
    """Dassl trainer for the fixed SaMoE architecture."""

    model_class = CustomCLIP

    def check_cfg(self, cfg):
        samoe_cfg = cfg.TRAINER.SAMOE
        assert samoe_cfg.PREC in ["fp16", "fp32", "amp"]
        assert samoe_cfg.ROUTING == "image-conditioned", (
            "SaMoE only supports image-conditioned routing"
        )
        assert samoe_cfg.ROUTER_SHARING == "shared", (
            "SaMoE uses one router shared by all prompted layers"
        )
        assert samoe_cfg.EXPERT_BANK_SHARING == "independent"
        assert samoe_cfg.TOP_K == TOP_K, "SaMoE fixes TOP_K=2"
        assert samoe_cfg.NUM_EXPERTS >= TOP_K

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.SAMOE.PREC in ["fp32", "amp"]:
            clip_model.float()

        self.model = self.model_class(cfg, classnames, clip_model)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_("prompt_learner" in name)

        trainable = [
            name for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        print(f"SaMoE trainable parameter tensors: {len(trainable)}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        optim_target = (
            self.model.prompt_learner
            if cfg.TRAINER.SAMOE.SAVE_PROMPT_ONLY
            else self.model
        )
        self.optim = build_optimizer(optim_target, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        if cfg.TRAINER.SAMOE.SAVE_PROMPT_ONLY:
            self.register_model(
                "SaMoEPromptLearner",
                self.model.prompt_learner,
                self.optim,
                self.sched,
            )
            print("Checkpoint mode: trainable SaMoE parameters only")
        else:
            self.register_model("SaMoE", self.model, self.optim, self.sched)
        self.scaler = (
            GradScaler() if cfg.TRAINER.SAMOE.PREC == "amp" else None
        )

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(
                f"Multiple GPUs detected (n_gpus={device_count}), "
                "using DataParallel"
            )
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        if self.cfg.TRAINER.SAMOE.PREC == "amp":
            with autocast():
                loss, ce_loss, balance_loss = self.model(image, label)
            loss = loss.mean()
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss, ce_loss, balance_loss = self.model(image, label)
            loss = loss.mean()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        if self.batch_idx + 1 == self.num_batches:
            self.update_lr()

        return {
            "loss": loss.item(),
            "ce_loss": ce_loss.mean().item(),
            "balance_loss": balance_loss.mean().item(),
        }

    def parse_batch_train(self, batch):
        return batch["img"].to(self.device), batch["label"].to(self.device)

    def set_model_mode(self, mode="train", names=None):
        # Prompt-only checkpoint registration still needs whole-model eval mode.
        super().set_model_mode(mode, names)
        self.model.train(mode == "train")

    def model_inference(self, input):
        with autocast(enabled=self.cfg.TRAINER.SAMOE.PREC == "amp"):
            return self.model(input)

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped: no directory given")
            return

        model_file = (
            "model-best.pth.tar"
            if epoch is None
            else f"model.pth.tar-{epoch}"
        )
        for name in self.get_model_names():
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            for key in [
                "prompt_learner.token_prefix",
                "prompt_learner.token_suffix",
                "prompt_learner.tokenized_prompts",
                "token_prefix",
                "token_suffix",
                "tokenized_prompts",
                "token_embeddings",
                "prompt_learner.token_embeddings",
                "input_ids",
                "attention_mask",
            ]:
                state_dict.pop(key, None)
            print(
                f'Loading {name} from "{model_path}" '
                f'(epoch={checkpoint["epoch"]})'
            )
            incompatible = self._models[name].load_state_dict(state_dict, strict=False)
            class_buffers = {"tokenized_prompts", "token_embeddings",
                             "prompt_learner.tokenized_prompts",
                             "prompt_learner.token_embeddings"}
            missing = set(incompatible.missing_keys) - class_buffers
            if missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "Checkpoint architecture does not match the current model: "
                    f"missing={sorted(missing)}, unexpected={incompatible.unexpected_keys}"
                )
