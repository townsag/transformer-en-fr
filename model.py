from flax import linen as nn
from typing import Optional

from flax.linen.dtypes import promote_dtype
from flax.typing import Array, Dtype, Initializer
from jax import numpy as jnp
import jax
import math


default_kernel_init = nn.initializers.lecun_normal()

class EmbedLayer(nn.Module):
    vocab_size: int
    features_in_embedding: int
    embedding_initializer: Initializer
    param_dtype: Optional[Dtype] = jnp.float32
    result_dtype: Optional[Dtype] = jnp.float32

    def setup(self):
        # create a param, ToDo: find a more concrete explanation about the use of the param function?
        # does this give us a mutable reference to a field inside a frozen dict?
        self.embed_lookup = self.param(
            'embed_lookup', 
            self.embedding_initializer, 
            (self.vocab_size, self.features_in_embedding), 
            self.param_dtype
            )

    # inputs will be an array of tokens like [12,34,1902, 82,...]
    # output is a two dimensional array of embedings of shape (input length, num features in embedding)
    def __call__(self, inputs: Array) -> Array:
        # ToDo: value error wont work on accelerator, use Chex
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError("inputs to embed layer must be integer or integer subtype")
        return jnp.take(
                    jnp.asarray(self.embed_lookup, dtype=self.result_dtype),
                    inputs,
                    axis=0
                    )

class PositionalEncoding(nn.Module):
    max_input_length: int
    features_in_embedding: int
    pe_dtype: Optional[Dtype] = jnp.float32

    def setup(self):
        # the positional encodings only need to be calculated once at the begining of the 
        # training so they can be applied to each sequence in each batch of training
        pe = jnp.zeros((self.max_input_length, self.features_in_embedding), dtype=self.pe_dtype)
        # creates an array of shape (max input length, 1) with each value being the index of that position
        # like [[0],[1],[2], ....,[2000]]
        positions = jnp.arange(0, self.max_input_length, dtype=self.pe_dtype)
        positions = jnp.expand_dims(positions, axis=1)
        # the div term is identical for each position, only depends on the embedding feature 
        # dimension (i think axis=2)
        # this makes an array of shape (featurs_in_embedding/2,)
        # the negative is because its in the denominator
        div_term = jnp.exp(
            jnp.arange(0, self.features_in_embedding, step=2, dtype=self.pe_dtype) * 
            -math.log(10000) / self.features_in_embedding
        )
        even_dims = jnp.arange(0,self.features_in_embedding,2)
        odd_dims = jnp.arange(1,self.features_in_embedding,2)

        # my understanding is that this will be jit compiled into an in place operation, noice
        pe = pe.at[:,even_dims].set(jnp.sin(jnp.outer(positions, div_term)))
        pe = pe.at[:,odd_dims].set(jnp.cos(jnp.outer(positions,div_term)))
        # add a batch dimension to the positional encoding matrix so that addition with the matrix
        # can be broadcast across a batch of sequences of embeddings
        pe = jnp.expand_dims(pe, axis=0)

        # ToDo: check to see if Flax is managing this as a varaible or a trainable parameter
        # ToDo: find out where I need to explicitly put stuff on the accelerator
        self.positional_encoding_matrix = jax.device_put(pe)

    def __call__(self, x):
        # x is a batch of encoded inputs like (batch_size, sequence_length, num_embedding_dims)
        # we only want the encodings up to the sequence length of each "sentance" in the input
        # this opperation performs the element wise addition of the positional encodings at each
        # position to the respective embeddings at each position broadcast across each sequence 
        # the batch
        x = x + self.positional_encoding_matrix[:,:x.shape[1]]
        return x
    
class MultiHeadSelfAttention(nn.Module):
    features_in_embedding: int
    head_dim: int
    num_heads: int
    QKV_initializer: Optional[Initializer] = default_kernel_init
    param_dtype: Optional[Dtype] = jnp.float32
    result_dtype: Optional[Dtype] = jnp.float32

    def setup(self):
        self.W_q = self.param(
            "query_weights", 
            self.QKV_initializer, 
            (self.num_heads, self.features_in_embedding, self.head_dim), 
            self.param_dtype
        )
        self.W_k = self.param(
            "key_weights", 
            self.QKV_initializer, 
            (self.num_heads, self.features_in_embedding, self.head_dim), 
            self.param_dtype
        )
        self.W_v = self.param(
            "value_weights", 
            self.QKV_initializer, 
            (self.num_heads, self.features_in_embedding, self.head_dim),  
            self.param_dtype
        )
        self.W_o = self.param(
            "output_weights",
            self.QKV_initializer,
            (self.features_in_embedding, self.features_in_embedding),
            self.param_dtype
        )

    def __call__(self, x):
        # this is my first time using einsum notation so I am very open to feeback
        # the input tensor is of shape (batch_size, sequence_length, embedding_dim)
        # the weights tensor is of shape (num_heads, embedding_dim, query_embedding_dim)
        # the result of the operation should be of shape (batch_size, num_heads, sequence_length, query_embedding_dim)
        # the query_embedding_dim should = embedding_dim // num_heads
        # I interpret this as a stack of sequences with each sequence having a stack of 8 distinct query vectors at each position
        querys_matrix = jnp.einsum("ble,heq->bhlq",x, self.W_q)
        keys_matrix = jnp.einsum("ble,hek->bhlk",x, self.W_k)
        values_matrix = jnp.einsum("ble,hev->bhlv",x, self.W_v)
        # take the transpose of the key matrix so that each key matrix for each head at each position in the 
        # sequence for each sequence in the batch can be multiplied by the respective query matrix
        keys_matrix_T = jnp.einsum("bhlk->bhkl", keys_matrix)
        # this notation is a bit slopy. In this case bh are the batch and head dimensions for both the query
        # and key matrix. l and s are both the length of the input sequence but they needed different names
        # for the einsum notation to be valid. p (for projection) is the name I gave to the embedding dimension for each query/key
        # vector. The size of this vector is equivalent to (features in input embeddings // num heads)
        # note: p has to have the same name for each matrix in the einsum string notation for the syntax to reflect the intention
        query_key_product = jnp.einsum("bhlp,bhps->bhls", querys_matrix, keys_matrix_T)
        scaled_attention_logits = query_key_product / math.sqrt(self.head_dim)
        # these are the softmax values coresponding to how much each embedding sequence should attend to each
        # other embedding
        attention_weights = jax.nn.softmax(scaled_attention_logits, axis=-1)
        # Scale the values matrix by the attention weights for each input position in the sequence relative to each other
        # input in that sequence. Attention_weights is of shape (batch, num_heads, len_input_sequence, len_input_sequence)
        # I interpret this as each row (l) being a row of attention values where the value in column (s) corresponds to how
        # important the input at that position (s) is to the input at the position (l). This is where the "attention" happens
        # The attention values are like a weighted sum of all the input embeddings where the sum is weighted by the relative
        # importance of the input to that position
        # values matrix is of shape (batch, num_heads, input_seq_length, value_projection_dimension)
        # attention values is of shape (batch, num_heads, sequence_len, value_projection_dimension)
        attention_values = jnp.einsum("bhls,bhsv->bhlv", attention_weights, values_matrix)
        # take the transpose of the attention array such that the head dimension and the seq_len dimension are swapped
        # the result is of shape (batch, seq_length, num_heads, value_projection_dimension)
        attention_values_T = jnp.transpose(attention_values, (0,2,1,3))
        # concatenate the attention values for each head into vector of shape (batch, seq lenghth, featues_in_embedding)
        concat_attention_values = attention_values_T.reshape(attention_values_T.shape[:-2] + (attention_values_T.shape[-2] * attention_values_T.shape[-1],))
        # this function multiplies the concatenated attention values by the output mmatrix
        # the output matrix is of shape (embedding_dim, output_embedding_dim) , they should be the samesize
        # multihead attention values should be a matrix of shape (batch, sequence_length, output_embedding_dim)
        multihead_attention_values = jnp.einsum("ble,eo->blo", concat_attention_values, self.W_o)
        return jnp.asarray(multihead_attention_values, dtype=self.result_dtype)

        # ToDo: visualize softmax values, should they be relatively uniform because of untrained weights?

# this class is equivalent to xW + b
class MyLinear(nn.Module):
    input_dimensionality: int
    output_dimensionality: int
    kernel_init: Optional[Initializer] = default_kernel_init
    bias_init: Optional[Initializer] = nn.initializers.zeros_init()
    param_dtype: Optional[Dtype] = jnp.float32
    output_dtype: Optional[Dtype] = jnp.float32

    def setup(self):
        self.kernel = self.param(
            "weights",
            self.kernel_init,
            (self.input_dimensionality, self.output_dimensionality),
            self.param_dtype
        )
        self.bias = self.param(
            "bias",
            self.bias_init,
            (self.output_dimensionality,),
            self.param_dtype
        )

    def __call__(self, x):
        activation = jnp.einsum("...le,ep->...lp", x, self.kernel)
        activation = activation + self.bias
        return jnp.asarray(activation, dtype=self.output_dtype)

class FeedForward(nn.Module):
    features_in_embedding: int
    feed_forward_dimension: int
    param_dtype: Optional[Dtype] = jnp.float32
    output_dtype: Optional[Dtype] = jnp.float32

    def setup(self):
        self.layers = [
            MyLinear(self.features_in_embedding, self.feed_forward_dimension, param_dtype=self.param_dtype,
                     output_dtype=self.output_dtype),
            MyLinear(self.feed_forward_dimension, self.features_in_embedding, param_dtype=self.param_dtype,
                     output_dtype=self.output_dtype)
        ]
    
    def __call__(self, x):
        activation = self.layers[0](x)
        activation = jnp.maximum(activation, 0)
        activation = self.layers[1](activation)
        return jnp.asarray(activation, dtype=self.output_dtype)

class MyLayerNorm(nn.Module):
    input_dimensionality: int
    epsilon: Optional[int] = 1e-10
    bias_init: Optional[Initializer] = nn.initializers.zeros_init()
    gain_init: Optional[Initializer] = nn.initializers.ones_init()
    param_dtype: Optional[Dtype] = jnp.float32
    output_dtype: Optional[Dtype] = jnp.float32

    def setup(self):
        self.gain = self.param(
            "gain",
            self.gain_init,
            (self.input_dimensionality,),
            self.param_dtype
        )
        self.bias = self.param(
            "bias",
            self.bias_init,
            (self.input_dimensionality,),
            self.param_dtype
        )
    def __call__(self, x):
        sample_wise_mean = jnp.mean(x, axis=-1, keepdims=True)
        sample_wise_std = jnp.sqrt(jnp.mean(jnp.square(x - sample_wise_mean), axis=-1, keepdims=True))
        sample_wise_normalized_x = (x - sample_wise_mean) / (sample_wise_std + self.epsilon)
        activation = jnp.multiply(self.gain, (sample_wise_normalized_x + self.bias))
        return jnp.asarray(activation, dtype=self.output_dtype)

class EncoderBlock(nn.Module):
    features_in_embedding: int
    num_heads: int
    feed_forward_dimension: int

    def setup(self):
        # ToDo: asserts dont work on accelerator, use Chex
        assert self.features_in_embedding % self.num_heads == 0, \
            f"the size of the embedding dimension  ({self.features_in_embedding}) should be evenly divisible by the number of heads {self.num_heads}"
        head_dim = self.features_in_embedding // self.num_heads
        self.attention_layer = MultiHeadSelfAttention(
            features_in_embedding=self.features_in_embedding,
            head_dim=head_dim,
            num_heads=self.num_heads,
        )
        self.feed_forward_layer = FeedForward(
            features_in_embedding=self.features_in_embedding,
            feed_forward_dimension=self.feed_forward_dimension
        )
    
    # def __call__(self, x):
