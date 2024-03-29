# BitGrok

Grok-1 inspired model using ternary {-1, 0, 1} weights.

A lot of the implementation currently has been taken from the Grok implementation. The main change is the exchange of the quantized 8 bit weight to a quantized 1 bit weight.

## What?

1. Grok inspired model using 1 bit weights

## Inspiration

1. [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)
1. [Grok's architecture](https://github.com/xai-org/grok-1/blob/main/model.py)

## Usage

```
pip install -r requirements.txt
python run.py
```

## Model Specifications (to be changed to be optimized with the 1 bit weight version)

BitGrok (like Grok-1) is currently designed with the following specifications:

    - Parameters: 314B
    - Architecture: Mixture of 8 Experts (MoE)
    - Experts Utilization: 2 experts used per token
    - Layers: 64
    - Attention Heads: 48 for queries, 8 for keys/values
    - Embedding Size: 6,144
    - Tokenization: SentencePiece tokenizer with 131,072 tokens
    - Additional Features:
        - Rotary embeddings (RoPE)
        - Supports activation sharding and 8-bit quantization
    - Maximum Sequence Length (context): 8,192 tokens

## Downloading the weights

_original: https://github.com/xai-org/grok-1?tab=readme-ov-file#downloading-the-weights_

You can download the weights using a torrent client and this magnet link:

```
magnet:?xt=urn:btih:5f96d43576e3d386c9ba65b883210a393b68210e&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce
```

or directly using [HuggingFace 🤗 Hub](https://huggingface.co/xai-org/grok-1):

```
git clone https://github.com/xai-org/grok-1.git && cd grok-1
pip install huggingface_hub[hf_transfer]
huggingface-cli download xai-org/grok-1 --repo-type model --include ckpt-0/* --local-dir checkpoints --local-dir-use-symlinks False
```

These weights are from what xAI open sources for Grok-1. The checkpoints are stored as 8 bit weights. Whenever the weights are loaded into BitGrok, it converts it into 1 bit weight.

## Todo

- [ ] Run the code (need GPU with enough memory)
- [ ] Checkpoint and save the 1 bit weight version of the model
- [ ] Optimize the model (lol obviously)
