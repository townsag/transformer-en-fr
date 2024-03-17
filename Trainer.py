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



def main():
    print("hello world")

    

if __name__ == "__main__":
    main()


# class Transformer
    


        