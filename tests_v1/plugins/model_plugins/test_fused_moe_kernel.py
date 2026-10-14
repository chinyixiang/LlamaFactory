# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

import pytest
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM


def _build_model() -> Qwen3MoeForCausalLM:
    config = Qwen3MoeConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=256,
    )
    config.architectures = ["Qwen3MoeForCausalLM"]
    torch.manual_seed(0)
    return Qwen3MoeForCausalLM(config)


def _forward_backward(model, input_ids):
    model.zero_grad(set_to_none=True)
    outputs = model(input_ids=input_ids, labels=input_ids)
    outputs.loss.backward()
    grads = {n: p.grad.detach().float().cpu() for n, p in model.named_parameters() if p.grad is not None}
    return outputs.logits.detach().float().cpu(), grads


def _rel_err(ref: torch.Tensor, out: torch.Tensor) -> float:
    return ((ref - out).norm() / (ref.norm() + 1e-12)).item()


@pytest.mark.slow
@pytest.mark.runs_on(["cuda", "xpu"])
def test_fused_moe_matches_default_experts():
    pytest.importorskip("triton")
    from llamafactory.v1.accelerator.helper import get_current_accelerator
    from llamafactory.v1.plugins.model_plugins.kernels.interface import apply_kernels

    device = get_current_accelerator()
    model = _build_model().to(torch.bfloat16).to(device)
    ref32 = copy.deepcopy(model).float()
    input_ids = torch.randint(0, 1000, (2, 64), device=device)

    ref_logits, ref_grads = _forward_backward(ref32, input_ids)
    del ref32
    base_logits, base_grads = _forward_backward(model, input_ids)

    model = apply_kernels(model, {"name": "cuda_fused_moe"})
    fused_logits, fused_grads = _forward_backward(model, input_ids)

    # The fused path must be as close to the fp32 reference as the default bf16 path.
    assert _rel_err(ref_logits, fused_logits) <= 1.5 * _rel_err(ref_logits, base_logits) + 1e-3

    expert_names = [n for n in ref_grads if "experts" in n and n in fused_grads and n in base_grads]
    assert expert_names, "no expert weight gradients were produced"
    for name in expert_names:
        assert torch.isfinite(fused_grads[name]).all(), name
        assert (
            _rel_err(ref_grads[name], fused_grads[name]) <= 1.5 * _rel_err(ref_grads[name], base_grads[name]) + 1e-3
        ), name
