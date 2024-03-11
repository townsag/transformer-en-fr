from absl.testing import absltest
from Trainer import EmbedLayer, PositionalEncoding

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

if __name__ == '__main__':
  absltest.main()