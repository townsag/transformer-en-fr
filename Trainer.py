"""
Trainer is an object oriented wrapper for our functional flax code. It is meant to
store all the constants and strings needed to train the model, hyperparameters and 
other related things
- Storing model and parameters: 
    - In order to train multiple models with different hyperparameters, the 
    trainer module creates an instance of the model class, and keeps the 
    parameters in the same class. This way, we can easily apply a model with its 
    parameters on new inputs.

- Initialization of model and training state: 
    - During initialization of the trainer, we initialize the model parameters and
      a new train state, which includes the optimizer and possible learning rate 
      schedulers.

- Training, validation and test loops: 
    - Similar to PyTorch Lightning, we implement simple training, validation and 
    test loops, where subclasses of this trainer could overwrite the respective 
    training, validation or test steps. Since in this tutorial, all models will 
    have the same objective, i.e. classification on CIFAR10, we will pre-specify 
    them in the trainer module below.

- Logging, saving and loading of models: 
    - To keep track of the training, we implement functionalities to log the 
    training progress and save the best model on the validation set. Afterwards, 
    this model can be loaded from the disk.
"""


from ml_collections import config_dict
from flax import linen as nn
from typing import Optional
from flax.linen.dtypes import promote_dtype
from flax.typing import Array, Dtype, Initializer
from jax import numpy as jnp
import jax
import math

class Trainer:
    def __init__(
            self,
            model_name: str,
            model_class: nn.Module,
            hyper_parameters: config_dict,
            verbose: bool,
            wandb_on: bool,
            checkpoint_on: bool,
            checkpoint_output_dir: str,
            example_input: any,
            seed: int = 0
            ):
        self.model_name = model_name
        self.model_class = model_class
        self.hyper_parameters = hyper_parameters
        self.verbose = verbose
        self.wandb_on = wandb_on
        self.checkpoint_on = checkpoint_on
        self.checkpoint_output_dir = checkpoint_output_dir
        self.seed = seed
        # create an empty instance of the model class
        self.model = self.model_class(**self.hyper_parameters)

    # this function is responsible for making jited training and evaluating functions
    # def create_functions(self):
    #     def calculate_loss

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

def main():
    print("hello world")

    

if __name__ == "__main__":
    main()


# class Transformer
    


        