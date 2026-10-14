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

import pytest
import torch

from llamafactory.extras.packages import is_transformers_version_greater_than


@pytest.mark.runs_on(["cuda", "xpu"])
@pytest.mark.skipif(not is_transformers_version_greater_than("5.2.0"), reason="Qwen3.5 requires transformers>=5.2.0")
def test_qwen35_gdn_patch_supports_packing():
    pytest.importorskip("fla")
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig

    from llamafactory.hparams import ModelArguments
    from llamafactory.model.patcher import patch_model
    from llamafactory.v1.accelerator.helper import get_current_accelerator

    device = get_current_accelerator()
    config = Qwen3_5TextConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=["linear_attention"] * 2,  # no full-attention layer, so packing isolation comes from the patch
        max_position_embeddings=256,
    )
    config.architectures = ["Qwen3_5ForCausalLM"]
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(config).to(torch.bfloat16).to(device).eval()

    len1, len2 = 40, 24
    ids1 = torch.randint(0, 1000, (1, len1), device=device)
    ids2 = torch.randint(0, 1000, (1, len2), device=device)
    packed_ids = torch.cat([ids1, ids2], dim=1)
    packed_position_ids = torch.cat([torch.arange(len1), torch.arange(len2)]).unsqueeze(0).to(device)

    def rel_err(ref: torch.Tensor, out: torch.Tensor) -> float:
        return ((ref - out).norm() / ref.norm()).item()

    with torch.no_grad():  # unpatched reference: transformers' own implementation, one sequence at a time
        ref1 = model(input_ids=ids1, use_cache=False).logits.float().cpu()
        ref2 = model(input_ids=ids2, use_cache=False).logits.float().cpu()

    # patch_model decides, per device and attention setting, whether the FLA-based GDN forward is installed.
    tokenizer = AutoTokenizer.from_pretrained("llamafactory/tiny-random-qwen3")
    model_args = ModelArguments(model_name_or_path="llamafactory/tiny-random-qwen3", flash_attn="fa2")
    patch_model(model, tokenizer, model_args, is_trainable=True, add_valuehead=False)

    with torch.no_grad():
        packed = model(input_ids=packed_ids, position_ids=packed_position_ids, use_cache=False).logits.float().cpu()
        unseparated = model(input_ids=packed_ids, use_cache=False).logits.float().cpu()

    err1 = rel_err(ref1, packed[:, :len1])
    err2 = rel_err(ref2, packed[:, len1:])
    assert err1 < 3e-2 and err2 < 3e-2, f"packed sequences differ from separate runs: {err1:.3e}, {err2:.3e}"

    # Without position_ids the second sequence sees the first one, so it must be clearly worse.
    assert rel_err(ref2, unseparated[:, len1:]) > 2 * err2

    model.train()
    loss = model(input_ids=packed_ids, position_ids=packed_position_ids, labels=packed_ids, use_cache=False).loss
    loss.backward()
    assert torch.isfinite(loss)
