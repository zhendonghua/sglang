# Adapted from qwen3.py and llama_embedding.py
import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.qwen3 import Qwen3Model as Qwen3TransformerModel
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class Qwen3Model(nn.Module):
    """Native embedding model for checkpoints that declare
    ``architectures=["Qwen3Model"]`` (a bare Qwen3 backbone with no LM head),
    e.g. ``microsoft/harrier-oss-v1-0.6b``.

    This wraps the native SGLang Qwen3 backbone with a LAST-token / L2-normalized
    pooler so such checkpoints run on the fused SGLang kernel path (fused QK-norm,
    fused RoPE, fused activations) instead of the generic Transformers fallback.

    It mirrors ``LlamaEmbeddingModel`` (bare ``MistralModel`` arch): the arch is
    served as an embedding model regardless of ``--is-embedding`` because
    ``is_generation_model`` classifies ``"Qwen3Model"`` as non-generative.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        # TP is supported: the native backbone's parallel layers (fused qkv /
        # gate_up projections, row-parallel o / down projections, vocab-parallel
        # embedding) shard themselves through their own weight_loader in
        # load_weights below. PP layer-filtering is not implemented (irrelevant
        # at the ~0.6B embedding scale this bare-backbone arch targets).
        self.model = Qwen3TransformerModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        # Pooling assumes LAST-token + L2-normalize, matching the harrier /
        # Qwen3-Embedding convention. A mean-pooled bare-Qwen3 checkpoint would
        # need a different PoolingType here -- the same latent limitation as the
        # existing Qwen3ForCausalLM embedding path, which also hardcodes LAST.
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = True,
    ) -> EmbeddingPoolerOutput:
        assert (
            get_embedding
        ), "Qwen3Model (bare backbone) is only supported for embedding"
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.pooler(hidden_states, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            # Bare-backbone checkpoints name tensors "layers.*", "embed_tokens.*"
            # and "norm.*"; the native backbone lives under the "model." prefix here.
            if not name.startswith("model.") and (
                name.startswith("layers.")
                or name.startswith("embed_tokens.")
                or name.startswith("norm.")
            ):
                name = add_prefix(name, "model")

            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            # A bare Qwen3Model has no LM head; skip any lm_head weights that a
            # non-tied checkpoint might carry (they are unused for embedding).
            if name.startswith("lm_head"):
                continue
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")


EntryClass = Qwen3Model
