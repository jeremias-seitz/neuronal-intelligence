<!-- Thank you @: https://github.com/othneildrew/Best-README-Template -->
<a id="readme-top"></a>

<!-- PROJECT LOGO -->
<h3 align="center">Neuronal Intelligence</h3>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Continual learning for artificial neural networks has been a hot topic for a while.
- Point of attack here is to reduce the amount of memory
- Possibility to implement everything as an optimizer
- Significant reduction in parameters but performance does not surpass previous methods
- Purpose is to provide a framework for easy integration of new methods
- Uses PyTorch

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

A CUDA compatible GPU is required.

### Installation

1. Clone the repo
    ```sh
    git clone https://github.com/jeseitz/neuronal_intelligence.git
    ```
2. Install packages
    ```sh
    pip install -r requirements.txt
    ```
    Depending on the local Python version and the Cuda version on the GPU, only specific combinations of torch, torchvision and torchaudio work. See [here](todo) for some combinations that can be tried when the vanilla install does not work.
3. Install the hydra package manually (does not work within requirements.txt)
    ```sh
    pip install hydra-core --upgrade
    ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Run in default settings

This project uses the [hydra](https://hydra.cc/docs/intro/) configuration package. By default, this [configuration file](config/main.yaml) is used. You can run the project in the default configuration with
```sh
python main.py
```

#### Specify path to datasets
Before running the project, it is advisable to adjust the path setting in the [dataset configuration](config/dataset/) for the appropriate dataset. For small datasets from torchvision (e.g. CIFAR10) it will be downloaded into the specified directory, for larger datasets (e.g. ImageNet) the user first has to download the dataset. Please check the [dataset implementation](datasets/) for implementation details.

### Change run settings

To change the configuration, there are four points to consider:
1. Customize the configuration
2. Configure a new object
3. Specify a different configuration file
4. Override arguments at runtime

#### Customize the hydra configuration
Hydra uses the YAML format for its configurations. The main configuration file used is [main.yaml](config/main.yaml). To change a setting, specify the desired value after the key. For example, to change the number of epochs during training, simply adjust the value after the 'epochs' key.
```yaml
epochs: 50
```
More complicated objects such as e.g. an optimizer can have their own parameter groups. Hydra allows the use of nested configurations, where the object parameters are specified in a separate configuration that is then referenced to in the main file. Nested configurations have to be listed under the 'defaults' key, e.g.:
```yaml
defaults:
  - model: resnet18
  - dataset: cifar100
  - optimizer: sgd
```
In the above example, 'resnet18' refers to the separate configuration file [resnet18.yaml](config/model/resnet18.yaml) in the `config/model/` subdirectory. Every module will have a corresponding subdirectory in the config folder.

#### Configure a new object
If an object should be configured, say a new optimizer, create a new YAML file in the `config/optimizer` directory e.g. 'adam.yaml' with the example content:
```yaml
_target_: torch.optim.Adam
lr: 1e-4
betas: [0.9, 0.999]
```
The `_target_` key specifies the object (can be a class or a function) where the other keys must match the constructor call signature. Then in the main configuration file change the value to:
```yaml
  - optimizer: adam
```
This system allows arbitrary referencing and enables modular configurations. The actual instantiation of objects is finally done using instantiate calls, see:
```py
hydra.utils.instantiate()
```
e.g. in [main.py](main.py).

#### Specify a different main configuration file
The configuration file to be used during the run is specified in the decorator of the main function in [main.py](main.py). See the line
```py
@hydra.main(version_base=None, config_path="config/", config_name="main.yaml")
```
To use a different configuration file, simply adjust `config_path` and `config_name` appropriately.  

#### Override arguments at runtime
To change the configuration at runtime, override the arguments after the python call. For example, to make sure the optimizer is set to SGD, use the command
```sh
python main.py optimizer=sgd
```
Use spaces when multiple arguments should be overwritten.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Module summaries
This section aims to provide a quick overview over the modules in this project.

### callbacks
All callback functions defined here will be called by the trainer at specific times during the training and testing process. See the [BaseTrainer](trainer/base_trainer.py) class for the exact times the callback functions are called. Examples for such functions are logging or saving the model dictionary.

### config
Hydra configuration tree. Contains a YAML file for every configurable object in the project. See <a href="#usage">Usage</a> for more details.

### datasets
Contains classes to handle datasets. The actual data will have to be downloaded into a specific directory. For more information, see [how to specify the path for datasets](#specify-path-to-datasets).
The following datasets are currently implemented:
<ol>
    <li>CIFAR10</li>
    <li>CIFAR100</li>
    <li>MNIST and variants (permuted, rotated, split)</li>
    <li>TinyImagenet</li>
</ol>

### learning scenarios
A learning scenario defines how tasks are learned which varies the difficulty of continual learning. The three scenarios are adapted from [Three scenarios for continual learning](http://arxiv.org/abs/1904.07734)(van de Ven & Tolias, 2019).
<details>
  <summary>Learning scenarios:</summary>
  <ol>
    <li>
        Class incremental learning expands the single-headed output layer for each new task. This is the hardest learning scenario since output neurons corresponding to previous task are not masked out. As a result, their readout weights can still experience gradient flow and can accordingly be subject to change.
    </li>
    <li>
        Domain incremental learning uses the same single output head for all tasks during training and testing. It can be used when the input distribution is changing but the number of classes remains constant.
    </li>
    <li>
        In the task incremental learning scenario the context is always known. The output is multi-headed with the 
        context defining the head to be used during training as well as testing. While every task is assigned a different output head the rest of the model is shared between the tasks.
    </li>
  </ol>
</details>
The scenarios are implemented using two components. First, the scenario class that takes care of defining the mask for the relevant output nodes and correcting for label offsets. And second, the network propagator class that propagates the input through the network.
 
### loss entities (continual learning algorithms)
All continual learning algorithms are defined here. The main training and testing functionality is shared between these algorithm classes and the [trainer class](trainer/base_trainer.py), see also [trainer](#trainer). Implemented here are always these four functionalities:

1. Preparation for a new task
2. Loss computation
3. Train a single batch
4. Test a single batch

The following continual learning algorithms are currently implemented:
<ol>
    <li>synaptic intelligence and variants</li>
    <li>neuronal intelligence and variants (OURS)</li>
    <li>learning rate scaling (OURS)</li>
    <li>elastic weight consolidation</li>
    <li>memory aware synapses</li>
    <li>riemannian walk</li>
    <li>vanilla backpropagation</li>
</ol>

### lr scheduler
Contains a wrapper and interface for different learning rate schedulers.

### models
Defines a number of backbones to the training process. Currently defined are:
<ol>
    <li>MLP</li>
    <li>ResNets</li>
    <li>Vision transformer<li>
</ol>


### optimizer
Implements the learning rate scaling algorithm implemented as custom continual learning optimizers based on SGD and Adam respectively. The optimizers are self-contained and can be used with any arbitrary loss function.

Note that when using either of the custom optimizers, the loss entity has to be set to the normal loss in the configuration.

### trainer
Implements training and testing routines for different tasks and calls the hooks for callback functions. This class calls the specific training and testing functions defined in the [continual learning algorithms](loss_entities/), see also [loss entities](#loss-entities-continual-learning-algorithms).

### utils
Utility functions shared between different classes.


<p align="right">(<a href="#readme-top">back to top</a>)</p>


## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Contact

Jeremias Seitz - jeremias.seitz@gmail.com

Project Link: [https://github.com/jeseitz/neuronal_intelligence](https://github.com/jeseitz/neuronal_intelligence)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->