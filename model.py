from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union, NamedTuple

from flax import linen as nn
from jax import numpy as jnp
import jax


@dataclass
class QuantizedWeight:
    weight: jnp.array  # Ternary weight values (-1, 0, 1)
    scale: jnp.array   # Scaling factors for the ternary weights

    @property
    def shape(self):
        return self.weight.shape


class Linear(nn.Module):
    output_size: int
    with_bias: bool = True

    @nn.compact
    def __call__(self, inputs: jnp.array) -> jnp.array:
        input_size = inputs.shape[-1]

        # Initialize and quantize the weights
        weight_scale = self.param(
            "w_scale", nn.initializers.ones, (input_size, self.output_size)
        )
        weight = self.param(
            "w", nn.initializers.zeros, (input_size,
                                         self.output_size), jnp.int8
        )
        quantized_weight = QuantizedWeight(weight=weight, scale=weight_scale)

        # Compute the output using the ternary weights
        out = jnp.dot(inputs, quantized_weight.weight * quantized_weight.scale)

        if self.with_bias:
            bias = self.param("b", nn.initializers.zeros, (self.output_size,))
            out += jnp.broadcast_to(bias, out.shape)
        return out


def ffn_size(emb_size, widening_factor):
    _ffn_size = int(widening_factor * emb_size) * 2 // 3
    _ffn_size = _ffn_size + (8 - _ffn_size) % 8  # ensure it's a multiple of 8
    return _ffn_size


@dataclass
class DenseBlock(nn.Module):
    num_q_heads: int
    num_kv_heads: int
    key_size: int
    widening_factor: float = 4.0
    sharding_constraint: bool = False
    mesh: Any = None

    @nn.compact
    def __call__(self, inputs: jnp.array) -> jnp.array:
        _, _, model_size = inputs.shape
        h_v = Linear(
            ffn_size(model_size, self.widening_factor),
            with_bias=False,
            name="linear_v",
        )(inputs)
        h_w1 = nn.gelu(
            Linear(
                ffn_size(model_size, self.widening_factor),
                with_bias=False,
            )(inputs)
        )
        h_dense = Linear(model_size, with_bias=False)(h_w1 * h_v)
        return h_dense


class KVMemory(NamedTuple):
    k: Optional[jnp.ndarray]
    v: Optional[jnp.ndarray]
    step: Optional[jnp.ndarray]


class MHAOutput(NamedTuple):
    embeddings: jnp.ndarray
    memory: Any


class MultiHeadAttention(nn.Module):
    num_q_heads: int
    num_kv_heads: int
    key_size: int
    value_size: Optional[int] = None
    model_size: Optional[int] = None
    attn_output_multiplier: float = 1.0
    data_axis: Union[str, Tuple[str, ...]] = "data"
    model_axis: Union[str, Tuple[str, ...]] = "model"

    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,
        key: Optional[jnp.ndarray],
        value: Optional[jnp.ndarray],
        mask: Optional[jnp.ndarray] = None,
        kv_memory: Optional[KVMemory] = None,
    ) -> MHAOutput:
        # In shape hints below, we suppress the leading dims [...] for brevity.
        # Hence e.g. [A, B] should be read in every case as [..., A, B].
        sequence_length = query.shape[1]
        projection = self._linear_projection
        use_memory = False
        if kv_memory is not None:
            if kv_memory.k is None:
                assert kv_memory.v is None
                assert key is not None
                assert value is not None
            else:
                assert kv_memory.v is not None
                use_memory = True
        else:
            assert key is not None
            assert value is not None

        # Check that the keys and values have consistent batch size and sequence length.
        if not use_memory:
            assert key.shape[:2] == value.shape[:
                                                2], f"key/value shape: {key.shape}/{value.shape}"

        if mask is not None:
            assert mask.ndim == 4
            assert mask.shape[0] in {
                1,
                query.shape[0],
            }, f"mask/query shape: {mask.shape}/{query.shape}"
            if not use_memory:
                assert key.shape[0] in {
                    1,
                    query.shape[0],
                }, f"key/query shape: {key.shape}/{query.shape}"
            assert mask.shape[1] == 1
            assert mask.shape[2] in {
                1,
                query.shape[1],
            }, f"mask/query shape: {mask.shape}/{query.shape}"
            if not use_memory:
                assert mask.shape[3] in {
                    1,
                    key.shape[1],
                }, f"mask/query shape: {mask.shape}/{key.shape}"

        # Compute key/query/values (overload K/Q/V to denote the respective sizes).
        assert self.num_q_heads % self.num_kv_heads == 0

        # Use the linear projection with ternary weights for query
        query_heads = projection(
            query, self.key_size, self.num_q_heads, name="query"
        )  # [B, T', H, Q=K]

        new_memory = None

        # Use the linear projection with ternary weights for key
        key_heads = projection(
            key, self.key_size, self.num_kv_heads, name="key"
        )  # [B, T, H, K]

        # Use the linear projection with ternary weights for value
        value_heads = projection(
            value, self.value_size or self.key_size, self.num_kv_heads, name="value"
        )  # [B, T, H, V]

        rotate = RotaryEmbedding(dim=self.key_size, base_exponent=int(1e4))
        key_heads = rotate(key_heads, seq_dim=1,
                           offset=(kv_memory.step if kv_memory else 0))
        query_heads = rotate(query_heads, seq_dim=1,
                             offset=(kv_memory.step if kv_memory else 0))

        @functools.partial(jax.vmap)
        def update_into(mem, start, update):
            return jax.lax.dynamic_update_slice_in_dim(mem, update, start, axis=0)

        if kv_memory:
            key_heads = update_into(kv_memory.k, kv_memory.step, key_heads)
            value_heads = update_into(kv_memory.v, kv_memory.step, value_heads)

            new_step = kv_memory.step + sequence_length
            memory_mask = jnp.arange(kv_memory.k.shape[1]) < new_step[:, None]
            memory_mask = memory_mask[:, None, None, :]  # [B, H, T, T]
            if mask is not None:
                mask = memory_mask * mask
            else:
                mask = memory_mask

            new_memory = KVMemory(
                k=key_heads,
                v=value_heads,
                step=new_step,
            )

        # Add separate dimension for grouped query heads.
        b, t, h, d = query_heads.shape
        _, _, kv_h, _ = key_heads.shape
        assert h % kv_h == 0, f"query_heads {h} must be a multiple of kv_heads {kv_h}"

        query_heads = query_heads.reshape(b, t, kv_h, h // kv_h, d)

        # Compute attention weights.
        # Attention softmax is always carried out in fp32.
        attn_logits = jnp.einsum("...thHd,...Thd->...hHtT", query_heads, key_heads).astype(
            jnp.float32
        )
        attn_logits *= self.attn_output_multiplier
        max_attn_val = jnp.array(30.0, dtype=attn_logits.dtype)
        attn_logits = max_attn_val * jnp.tanh(attn_logits / max_attn_val)

        mask = mask[:, :, None, :, :]

        if mask is not None:
            if mask.ndim != attn_logits.ndim:
                raise ValueError(
                    f"Mask dimensionality {mask.ndim} must match logits dimensionality "
                    f"{attn_logits.ndim} for {mask.shape}/{attn_logits.shape}."
                )
            attn_logits = jnp.where(mask, attn_logits, -1e30)
        attn_weights = jax.nn.softmax(
            attn_logits).astype(query.dtype)  # [H, T', T]

        # Weight the values by the attention and flatten the head vectors.
        attn = jnp.einsum("...hHtT,...Thd->...thHd", attn_weights, value_heads)
        leading_dims = attn.shape[:2]
        attn = attn.reshape((*leading_dims, -1))  # [T', H*V]

        # Apply another projection to get the final embeddings.
        final_projection = Linear(
            self.model_size or self.key_size * self.num_q_heads,
            with_bias=False,
        )
        return MHAOutput(final_projection(attn), new_memory)

    def _linear_projection(
        self,
        x: jnp.ndarray,
        head_size: int,
        num_heads: int,
        name: str,
    ) -> jnp.ndarray:
        # Use the linear layer with ternary weights for projection
        y = Linear(num_heads * head_size, with_bias=False, name=name)(x)
        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, num_heads, head_size))


def rotate_half(
    x: jax.Array,
) -> jax.Array:
    """Obtain the rotated counterpart of each feature"""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


class RotaryEmbedding(nn.Module):
    """Applies rotary embeddings (RoPE) to the input sequence tensor,
    as described in https://arxiv.org/abs/2104.09864.

    Attributes:
        dim (int): Dimensionality of the feature vectors
        base_exponent (int): Base exponent to compute embeddings from
    """

    def __init__(
        self,
        dim: int,
        name: Optional[str] = None,
        base_exponent: int = 10000,
    ):
        super().__init__(name)
        self.dim = dim
        self.base_exponent = base_exponent
        assert self.dim % 2 == 0

    def __call__(
        self,
        x: jax.Array,
        seq_dim: int,
        offset: jax.Array,
        const_position: Optional[int] = None,
        t: Optional[jax.Array] = None,
    ) -> jax.Array:
        fprop_dtype = x.dtype
        # Compute the per-dimension frequencies
        exponents = jnp.arange(0, self.dim, 2, dtype=jnp.float32)
        inv_freq = jnp.asarray(
            1.0 / (self.base_exponent ** (exponents / self.dim)), dtype=jnp.float32
        )

        if jnp.shape(offset) == ():
            # Offset can be a scalar or one offset per batch element.
            offset = jnp.expand_dims(offset, 0)

        # Compute the per element phase (to pass into sin and cos)
        if const_position:
            t = const_position * jnp.ones(
                (
                    1,
                    x.shape[seq_dim],
                ),
                dtype=jnp.float32,
            )
        elif t is None:
            t = jnp.arange(
                x.shape[seq_dim], dtype=jnp.float32) + jnp.expand_dims(offset, -1)
        phase = jnp.einsum("bi,j->bij", t, inv_freq)
        phase = jnp.tile(phase, reps=(1, 2))[:, :, None, :]

        x = x * jnp.cos(phase) + rotate_half(x) * jnp.sin(phase)
        x = x.astype(fprop_dtype)

        return x
