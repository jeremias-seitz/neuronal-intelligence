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
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#run-in-default-settings">Run in detault settings</a></li>
        <li><a href="#change-run-settings">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#module-summaries">Module summaries</a>
      <ul>
        <li><a href="#callbacks">Callbacks</a></li>
        <li><a href="#config">Configuration</a></li>
        <li><a href="#datasets">Datsets</a></li>
        <li><a href="#learning-scenarios">Learning scenarios</a></li>
        <li><a href="#algorithms">Continual learning algorithms</a></li>
        <li><a href="#lr-scheduler">Learning rate scheduler</a></li>
        <li><a href="#models">Models</a></li>
        <li><a href="#optimizer">Optimizer</a></li>
        <li><a href="#trainer">Trainer</a></li>
        <li><a href="#utils">Utility functions</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Intelligent agents have to continuously learn when faced with non-stationary environments. 
The challenge is to efficiently adapt to new tasks while retaining good performance on tasks learned previously. Traditional deep learning algorithms - primarily designed for stationary data - struggle in this continual learning setting where they are prone to catastrophic forgetting of already acquired information ([Parisi et al., 2019](http://arxiv.org/abs/1802.07569), [Wang et al., 2023](http://arxiv.org/abs/2302.00487)).

Among the proposed continual learning strategies that mitigate catastrophic forgetting, the most dominant ones are replay-based methods, which employ a replay buffer or a generative model to jointly mix data from old tasks and new tasks, and regularization-based methods such as Elastic Weight Consolidation (EWC) ([Kirkpatrick et al;, 2017](http://arxiv.org/abs/1612.00796)) or Synaptic Intelligence (SI) ([Zenke et al., 2017](http://arxiv.org/abs/1703.04200)), that employ structural regularization to protect important parameters. 
However, both approaches incur a cost. While the former requires extra circuitry to enable replay, the latter requires storing one or more variables per synapse in the network to implement the regularization term. This added cost limits their applicability whenever memory is a constraining resource, such as large models or edge applications. To overcome this [Jung et al. (2021)](http://arxiv.org/abs/2003.13726) introduced a node-based group sparse regularization technique that employs neuronal instead of synaptic regularization parameters. In a similar vein [Barry et al. (2024)](http://arxiv.org/abs/2306.01690) also use importance values based on neuronal activation levels. However, both approaches cannot completely dispense with synaptic parameters in their computations.
To close the gap, we propose Neuronal Intelligence (NI), a purely neuron-based importance measure with minimal memory footprint that can be used as a drop-in replacement for structural regularization or continual learning algorithm. We further simplify the algorithm to implement it as a self-contained optimizer that requires only minimal change compared to a non-CL approach.
The algorithms are inspired by neurobiology where neurons possess the bio-chemical machinery to modulate their propensity of being recruited into a new memory trace ([Yiu et al., 2014](https://www.sciencedirect.com/science/article/pii/S089662731400628X), [Josselyn & Tonegawa, 2020](https://www.science.org/doi/full/10.1126/science.aaw4325)). In our case, we employ an importance ranking such that the subset of least important neurons always remains plastic while high importance guarantees stability.

Our main contributions are the following:
<ul>
    <li>We propose a simplification of the importance computation in SI with negligible performance loss.</li>
    <li>We introduce Regularization-based Neuronal Intelligence (NI-light) for resource-efficient continual learning, a regularization-based strategy that stores the aggregate synaptic importance at the neuronal level.</li>
    <li>We replace the regularization in Optimizer-based Neuronal Intelligence (NI-opt) with a learning rate multiplier to rely solely on neuronal parameters and to provide minimal implementation effort.</li>
    <li>We show that NI-light and NI-opt yield the same or comparable performance to SI on Permuted MNIST and split CIFAR-100, with a two- respectively fourfold reduction in required memory.</li>
</ul>

This repository implements the continual learning algorithms SI, SI-light, NI-light, NI-opt, Online EWC, MAS and RWalk for supervised classification tasks, see [algorithms](#algorithms). Functionalities such as joint training or task shuffling are implemented and can be changed via flags in the configuration file.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

A CUDA compatible GPU is required.

### Installation

1. Clone the repo
    ```sh
    git clone https://github.com/jeremias-seitz/neuronal-intelligence.git
    ```

2. Install packages
    ```sh
    pip install -r requirements.txt
    ```

3. Install the hydra package manually (does not work within requirements.txt)
    ```sh
    pip install hydra-core --upgrade
    ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Run in default settings

This project uses the [hydra](https://hydra.cc/docs/intro/) configuration package. By default, [this configuration file](config/main.yaml) is used. You can run the project in the default configuration with
```sh
python main.py
```

#### Specify path to datasets
Before running the project, it is advisable to adjust the path setting in the [dataset configuration](config/dataset/) for the appropriate dataset. For small datasets from torchvision (e.g. CIFAR10) it will be downloaded into the specified directory, for larger datasets (e.g. ImageNet) the user first has to download the dataset. Please check the [dataset implementation](datasets/) for implementation details.

### Change run settings

To change the configuration, there are four points to consider:
[1. Customize the configuration](#customize-the-hydra-configuration)

[2. Configure a new object](#configure-a-new-object)

[3. Specify a different configuration file](#specify-a-different-main-configuration-file)

[4. Override arguments at runtime](#override-arguments-at-runtime)

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
<ul>
    <li>CIFAR10</li>
    <li>CIFAR100</li>
    <li>MNIST and variants (permuted, rotated, split)</li>
    <li>TinyImagenet</li>
</ul>

### learning scenarios
A learning scenario defines how tasks are learned which varies the difficulty of continual learning. The three scenarios are adapted from [Three scenarios for continual learning](http://arxiv.org/abs/1904.07734)(van de Ven & Tolias, 2019).
<details>
  <summary>Learning scenarios:</summary>
  <ul>
    <li>
        **Class incremental learning** expands the single-headed output layer for each new task. This is the hardest learning scenario since output neurons corresponding to previous task are not masked out. As a result, their readout weights can still experience gradient flow and can accordingly be subject to change.
    </li>
    <li>
        Domain **incremental learning** uses the same single output head for all tasks during training and testing. It can be used when the input distribution is changing but the number of classes remains constant.
    </li>
    <li>
        In the task **incremental learning** scenario the context is always known. The output is multi-headed with the 
        context defining the head to be used during training as well as testing. While every task is assigned a different output head the rest of the model is shared between the tasks.
    </li>
  </ul>
</details>
The scenarios are implemented using two components. First, the scenario class that takes care of defining the mask for the relevant output nodes and correcting for label offsets. And second, the network propagator class that propagates the input through the network.
 
### algorithms
All continual learning algorithms are defined here. The main training and testing functionality is shared between these algorithm classes and the [trainer class](trainer/base_trainer.py), see also [trainer](#trainer). Implemented here are always these four functionalities:

<ul>
  <li>Preparation for a new task</li>
  <li>Loss computation</li>
  <li>Train a single batch</li>
  <li>Test a single batch</li>
</ul>

The following continual learning algorithms are currently implemented:
<ul>
    <li>synaptic intelligence and variants</li>
    <li>neuronal intelligence and variants (OURS)</li>
    <li>learning rate scaling (OURS)</li>
    <li>elastic weight consolidation</li>
    <li>memory aware synapses</li>
    <li>riemannian walk</li>
</ul>

### lr scheduler
Contains a wrapper and interface for different learning rate schedulers.

### models
Defines a number of backbones to the training process. Currently defined are:
<ul>
    <li>MLP</li>
    <li>ResNets</li>
    <li>Vision transformer</li>
</ul>


### optimizer
Implements the learning rate scaling algorithm implemented as custom continual learning optimizers based on SGD and Adam respectively. The optimizers are self-contained and can be used with any arbitrary loss function.

Note that when using either of the custom optimizers, the algorithm has to be set to the `vanilla_backprop` in the configuration.

### trainer
Implements training and testing routines for different tasks and calls the hooks for callback functions. This class calls the specific training and testing functions defined in the [continual learning algorithms implementations](algorithms/), see also [the algorithm section](#algorithms).

### utils
Utility functions shared between different classes.


<p align="right">(<a href="#readme-top">back to top</a>)</p>


## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



## Contact

Jeremias Seitz - jeremias.seitz@gmail.com

Project Link: [https://github.com/jeremias-seitz/neuronal-intelligence](https://github.com/jeremias-seitz/neuronal-intelligence)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
