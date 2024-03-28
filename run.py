# Copyright 2024 X.AI Corp.
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

import logging
import jax
from flax.training.common_utils import shard

from model import LanguageModelConfig, TransformerConfig, QuantizedWeight
from runners import sample_from_model


def main():
    grok_1_model = LanguageModelConfig(
        vocab_size=128 * 1024,
        pad_token=0,
        eos_token=2,
        sequence_len=8192,
        embedding_init_scale=1.0,
        output_multiplier_scale=0.5773502691896257,
        embedding_multiplier_scale=78.38367176906169,
        model=TransformerConfig(
            emb_size=48 * 128,
            widening_factor=8,
            key_size=128,
            num_q_heads=48,
            num_kv_heads=8,
            num_layers=64,
            attn_output_multiplier=0.08838834764831845,
            shard_activations=True,
            # MoE.
            num_experts=8,
            num_selected_experts=2,
            # Activation sharding.
            data_axis="data",
            model_axis="model",
        ),
    )

    # Create a new model instance
    rng = jax.random.PRNGKey(0)
    model = grok_1_model.make()
    params = model.init(rng, jax.numpy.ones((1, 1), dtype=jax.numpy.int32))

    # Shard the model parameters
    params = jax.tree_map(lambda x: shard(x), params)

    # Define the input prompt
    inp = "The answer to life the universe and everything is of course"

    # Generate output from the model
    output = sample_from_model(
        model.apply, params, inp, max_len=100, temperature=0.01)
    print(f"Output for prompt: {inp}\n{output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
