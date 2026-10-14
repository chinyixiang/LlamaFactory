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
from transformers import AutoModelForCausalLM


@pytest.mark.runs_on(["cuda", "xpu"])
def test_liger_kernel_patches_model():
    pytest.importorskip("liger_kernel")
    from llamafactory.v1.accelerator.helper import get_current_accelerator
    from llamafactory.v1.plugins.model_plugins.kernels.interface import apply_kernels

    device = get_current_accelerator()
    model = AutoModelForCausalLM.from_pretrained("llamafactory/tiny-random-qwen3", dtype=torch.bfloat16).to(device)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 32), device=device)
    original_rmsnorm_forward = model.model.layers[0].input_layernorm.forward

    with torch.no_grad():
        expected_logits = model(input_ids=input_ids).logits.float().cpu()

    # require_logits keeps the logits-producing cross entropy instead of the fused linear one.
    model = apply_kernels(model, {"name": "liger_kernel", "require_logits": True})
    assert model.model.layers[0].input_layernorm.forward.__func__ is not original_rmsnorm_forward.__func__

    outputs = model(input_ids=input_ids, labels=input_ids)
    outputs.loss.backward()
    assert torch.isfinite(outputs.loss)

    logits = outputs.logits.detach().float().cpu()
    rel_err = ((logits - expected_logits).norm() / expected_logits.norm()).item()
    assert rel_err < 0.05, f"liger-patched logits differ from the default path: {rel_err:.3e}"
