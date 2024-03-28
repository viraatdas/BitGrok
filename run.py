import jax.numpy as jnp
import logging

from model import LanguageModelConfig, TransformerConfig
from runners import InferenceRunner, ModelRunner, sample_from_model

CKPT_PATH = "./checkpoints/"

# Original Grok model saves the checkpoints in the 8 bit
# weight:
# https://github.com/xai-org/grok-1/tree/main?tab=readme-ov-file#downloading-the-weights

# This method converts the 8 bit weight to 1 bit weight


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
    inference_runner = InferenceRunner(
        pad_sizes=(1024,),
        runner=ModelRunner(
            model=grok_1_model,
            bs_per_device=0.125,
            checkpoint_path=CKPT_PATH,
        ),
        name="local",
        load=CKPT_PATH,
        tokenizer_path="./tokenizer.model",
        local_mesh_config=(1, 8),
        between_hosts_config=(1, 1),
    )
    inference_runner.initialize()
    gen = inference_runner.run()

    inp = "The answer to life the universe and everything is of course"
    print(f"Output for prompt: {inp}", sample_from_model(
        gen, inp, max_len=100, temperature=0.01))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
