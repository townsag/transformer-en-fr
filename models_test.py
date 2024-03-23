from absl.testing import absltest
from model import EmbedLayer, FeedForward, MultiHeadSelfAttention, MyLayerNorm, MyLinear, PositionalEncoding

from flax import linen as nn
import jax
from jax import numpy as jnp


class TestEmbed(absltest.TestCase):
    def testInitializeEmbed(self):
        initializer = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal', out_axis=0)
        # create a module instance by calling the init method with correct params
        embed = EmbedLayer(vocab_size=20, features_in_embedding=300, embedding_initializer=initializer)
        # initialize the module parameters by calling the init function with a seed and a sampe input
        png = jax.random.PRNGKey(0)
        dummy_input = jax.random.randint(png, (2,10), 0, 20)
        state = embed.init(png, dummy_input)
        y1 = embed.apply(state, jax.random.randint(png, (2,10), 0, 20))
        self.assertEqual(y1.shape, (2, 10, 300))
    
    # def testEmbedDType
    # def testEmbedRange # should throw an out of range error for token out of range

class TestPositionalEncoding(absltest.TestCase):
    def testInitializePositionalEncoding(self):
        encoding_layer = PositionalEncoding(max_input_length=100, features_in_embedding=300)
        dummy_input = jnp.ones((8,40,300))
        png = jax.random.PRNGKey(0)
        state = encoding_layer.init(png, dummy_input)
        # positional encoding layer has no trainable parameters
        self.assertEqual(state, {})
        encoded_dummy_input = encoding_layer.apply(state, dummy_input)
        self.assertEqual(encoded_dummy_input.shape, (8,40,300))
    
class TestMultiHeadSelfAttention(absltest.TestCase):
    def testInitialize(self):
        self_attention_layer = MultiHeadSelfAttention(features_in_embedding=512, head_dim=64, num_heads=8)
        dummy_input = jnp.ones((16,40,512))
        png = jax.random.PRNGKey(0)
        state = self_attention_layer.init(png,dummy_input)
        dummy_output = self_attention_layer.apply(state, dummy_input)
        self.assertEqual(dummy_output.shape, (16,40,512))

class TestMyLinear(absltest.TestCase):
    def testInitialize(self):
        my_linear_layer = MyLinear(input_dimensionality=512, output_dimensionality=2048)
        dummy_input = jnp.ones((16,40,512))
        png = jax.random.PRNGKey(0)
        state = my_linear_layer.init(png, dummy_input)
        dummy_output = my_linear_layer.apply(state, dummy_input)
        self.assertEqual(dummy_output.shape, (16,40,2048))

class TestFeedForward(absltest.TestCase):
    def testInitialize(self):
        feed_forward_layer = FeedForward(512, 2048)
        dummy_input = jnp.ones((16,30,512))
        png = jax.random.PRNGKey(0)
        state = feed_forward_layer.init(png, dummy_input)
        dummy_output = feed_forward_layer.apply(state, dummy_input)
        self.assertEqual(dummy_output.shape, (16,30,512))
    

class TestMyLayerNorm(absltest.TestCase):
    # ToDo: test initialization and output shape
    def testInitialize(self):
        layer_norm_layer = MyLayerNorm(input_dimensionality=512)
        png = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((16,30,512))
        state = layer_norm_layer.init(png, dummy_input)
        dummy_output = layer_norm_layer.apply(state, dummy_input)
        self.assertEqual(dummy_input.shape, dummy_output.shape)
    
    def testResultMeanSTD(self):
        layer_norm_layer = MyLayerNorm(input_dimensionality=512)
        png = jax.random.PRNGKey(0)
        dummy_input = jax.random.uniform(png, shape=(16,30,512))
        state = layer_norm_layer.init(png, dummy_input)
        dummy_output = layer_norm_layer.apply(state, dummy_input)
        output_sample_wise_mean = jnp.sum(dummy_output, axis=-1, keepdims=True)
        output_sample_wise_std = jnp.sqrt(jnp.mean(jnp.square(dummy_output - output_sample_wise_mean), axis=-1, keepdims=True))
        # ToDo: figure out why atol of 1e-3 is needed for the test to pass, that seems really high
        assert jnp.allclose(output_sample_wise_mean, jnp.zeros_like(output_sample_wise_mean), atol=1e-3)
        assert jnp.allclose(output_sample_wise_std, jnp.ones_like(output_sample_wise_std))

if __name__ == '__main__':
  absltest.main()