
# CIFAR-10-Mini Project

This is a project that aim to train a neural network on a reduced subset of the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The original dataset contains 50,000 training examples and 10,000 test cases. Here I try to select no more than 1000 training examples and try to reach the best result on the test cases.

## Choosing the Subset

A valid assumption is to select 100 images from each of the 10 classes. I want a smaller set of images that are quite different from each other - actually, they differ as much as possible. This is called as the **Subset Diversity Maximisation** problem. For that, I need to define a distance metric, which measures the distance between two images.

For that, I train a [Convolutional_Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network) to solve the classification problem, using the entire database. It is safe to assume that a trained network produces rich feature vectors on its last layer which encodes the key features of an image. These feature vectors can be used to define our distance metric. Literature suggests to use [cosine distance](https://en.wikipedia.org/wiki/Cosine_similarity) if the feature vectors originate from a convolutional network.

Since I don't want to optimize the classification here (just to produce feature vectors that good enough), a simple and fast network with 70-80% accuracy will be sufficient.

Now that I have the feature vectors and the distance, the distance matrices for each class can be set up. Here, I run 10,000 attempts: with each attempt the script selects 100 images randomly, and measure their overall feature vector distances. The script keeps the best attempt with the highest distance sum. This is a fairly simple solution, and more adequate optimizations can be used here, for example [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing). Also, note that according to [this article](https://arxiv.org/abs/1606.04232)<sup>1</sup>, one can define a more sofisticated method for chosing a subset, considering Subset Representativeness as well, which minimizes the average distance between selected and non-selected training samples.

## Choosing the Base Neural Network

Since our training sample is much smaller than the original, I need to find a suitable network that can learn even from such a small dataset. Because the problem is image classification, I consider two networks commonly used in the literature: [VGG](https://neurohive.io/en/popular-networks/vgg16/) and [ResNet](https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035). Furthermore, I use two versions of VGG: an 8-layer and a 16-layer version. For ResNet, I choose ResNet-18. All networks are convolutional neural networks, with [softmax layer](https://deepai.org/machine-learning-glossary-and-terms/softmax-layer) on top, and [cross-entropy](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy) loss. For the results and training details, see **Results in Jupyter Notebook**.

## Improving the Chosen Network

Based on the results, I choose VGG-8 as the basic network. I try to improve the initial accuracy of 38.5% by experimenting with data augmentation, dropout, weight decay, early stop, and different learning rates and batch sizes. The final network reached a **48.1% accuracy** - which is much worse than a [network can achieve](https://dawn.cs.stanford.edu/benchmark/CIFAR10/train.html) using the entire dataset, but better than a random guess of 10%. Still, better results are surely can be achieved by using more computational intensive optimizations, such as [hyperparameter optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization).

## Experimental: a Network With Two Output Layers

Irrelevant to the initial goal, an experimental version of VGG-8 is also implemented. This network has two output layers; one produces the original softmax output, the other one produces a result equal in size, but places constant 1s in all place (regardless of training or inference). This network is also trainable on the CIFAR-10 subset, however, the second layer is not used in the loss function. Still, in the future it can be easily modified to solve [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification) problems.

# Usage

## Run the Script Locally

Requirements:

- Python 3.8.3
- TensorFlow 2.2.0
- NumPy 1.19.0
- scikit-learn 0.23.1
- Matplotlib 3.2.2

If you want to use your GPU to boost the training:

- Hardware requirements:
  - NVIDIA GPU card with CUDA Compute Capability 3.5 or higher
- Software requirements:
  - CUDA 10.1
  - cuDNN SDK (>= 7.6)

See [https://www.tensorflow.org/install/gpu](https://www.tensorflow.org/install/gpu) for details on GPU usage.

### Parameters

The basic scenarios can be set by command line arguments. However, the more in-depth parameters (such as the learning rate or batch size) are collected into a single file named `saves/configs.json`. Some parameters (like the detailed parameters of data augmentation) are hard-coded.

#### Command Line Parameters

```
usage: main.py [-h] [-g GPU_MEMORY_LIMIT] config_file_name {train_fv_extractor,load_fv_extractor,load_indices} {simple,advanced,experimental} ...

positional arguments:
  config_file_name      Config file name
  {train_fv_extractor,load_fv_extractor,load_indices}
                        Selection method of a subset of the training data. Options: train feature vector extractor network and search for indices / load feature vector extractor network
                        weights and search for indices / load previously searched indices
  {simple,advanced,experimental}
                        Simple, advanced, or experimental training scenario
    simple              Chose a simple network with basic training
    advanced            Chose between advanced training options for a fixed network
    experimental        Train a network with two output layers

optional arguments:
  -h, --help            show this help message and exit
  -g GPU_MEMORY_LIMIT, --gpu_memory_limit GPU_MEMORY_LIMIT
                        GPU memory limit in bytes (default: 2048)

```

The `config_file_name` is usually `saves/configs.json`, but alternatively, you can use your own version of the config file.

Next, you must choose one of the three options how to select the training subset. Either you can use the pre-calculated values stored in `saves/selected_training_indices.json`, or you can load the pre-trained feature vector extractor network and re-calculate the indices with `load_fv_extractor`. The third option is to train the network itself from scratch, and then search for indices by choosing `train_fv_extractor`.

As for the scenarios, there are three options. `simple` will train one of the basic models (VGG-8, VGG-16 or ResNet-18) with or without data augmentation:

```
usage: main.py config_file_name {train_fv_extractor,load_fv_extractor,load_indices} simple [-h] [-a] {VGG_8_simple,VGG_16_simple,ResNet_18_simple}

positional arguments:
  {VGG_8_simple,VGG_16_simple,ResNet_18_simple}
                        Select a simple model

optional arguments:
  -h, --help            show this help message and exit
  -a, --use_augmentation
                        Use data augmentation on the training images
```

The scenario `advanced` will train VGG-8 with selectable optimizer (SGD/Adam), with or without dropout, and with or without weight decay:

```
usage: main.py config_file_name {train_fv_extractor,load_fv_extractor,load_indices} advanced [-h] [-d] [-w] {SGD,Adam}

positional arguments:
  {SGD,Adam}            Training method

optional arguments:
  -h, --help            show this help message and exit
  -d, --use_dropout     Use dropout
  -w, --use_weight_decay
                        Use weight decay (l2 kernel regularizer)
```

Finally, the scenario `experimental` will lauch the the VGG-8 that has two output layers. Unlike the previous scenarios, here all the parameters are hard coded.

Some example usages:

```
python main.py saves/configs.json load_indices simple VGG_8_simple
```
```
python main.py saves/configs.json train_fv_extractor --gpu_memory_limit 4096 advanced Adam --use_dropout
```
```
python main.py saves/configs.json load_fv_extractor experimental
```

#### Content of `saves/configs.json`

- `MODEL_WEIGHTS_FOLDER_NAME`: The script saves the model weights here (except for `experimental`). Jupyter Notebook uses them for presenting the results.
- `TRAINING_HISTORIES_FOLDER_NAME`: The script saves the training histories here. Jupyter Notebook uses them for presenting the results.
- `TRAINING_INDICES_FILE_NAME`: Load/save the subset indices from/to this file.
- `SUBSET_SELECTION`: Parameters of the subset selection.
  - `TRAINING_SUBSET_SIZE`: How many training images will be selected. Equally distributed among the classes (i.e., a value of 1000 will result 100 images per class).
  - `TRAINING_SUBSET_SHUFFLE_ATTEMPT`: How many times the algorithm selects random indices in hope for the highest distance among the selected images' feature vectors.
 
 The rest of the parameters (`LEARNING_RATE`, `MOMENTUM`, `EPOCHS`, `BATCH_SIZE` are self-evident.

## Run the Script From Docker

Alternatively, one can run the script from a docker container. See [https://www.tensorflow.org/install/docker](https://www.tensorflow.org/install/docker) for installing docker with GPU support.

To pull the docker image from the public docker repository:
```
docker pull hantosnorbert/cifar-10-mini:v1
```
The docker image does not contain the `saves` folder and its content. You can mount them into the image's working directory  `/app`:
```
docker run --gpus all -it --rm --name cifar10 -v /[PATH_TO_SAVES]/saves/:/app/saves/ hantosnorbert/cifar-10-mini:v1 [PARAMETERS]
```
where `[PARAMETERS]` are the parameters of the python script as described above. (The docker image uses `main.py` as the entry point.)

If you want to build your own docker image, a docker image file is provided:
```
docker build -f cifar10mini.dockerfile --tag [TAG] .
```
Note that with the `--rm` option you download the CIFAR-10 dataset with each run (since the container will be deleted afterwards). Alternatively, you can reuse the same docker container to avoid that.

## Results in Jupyter Notebook

The results are presented in Jupyter Notebook. If you want to see them, [install Jupyter Notebook](https://jupyter.org/install), and open the `results.ipynb` file.

Note that the Jupyter Notebook is not prepared to use the dockerized script (i.e., you might have to install the necessary python packages on your local machine).

# Future Plans

- Set up a script that runs a broad hyperparameter-search for broader and more automated optimization.
- Instead of train-test dataset split, use train-test-validation datasets. Alternatively: use cross-validation.
- Experiment with BatchNormalization layers.
- Try different networks with different widths and depths.
- Instead of using the built-in funcion `datasets.cifar10.load_data()`, make possible to load the dataset from pre-downloaded files. Furthermore, save the selected subset to make the loading procedure faster.
- Make Jupyter Notebook be runnable from the docker.
- Scalable deployment (see **Deployment** in detail).

## Deployment

In case of someone wants to deploy a trained model, they want to consider it to do in a scalable way - possible in a multi GPU system.

Tensorflow provides tools for [multi GPU inference](https://medium.com/@sbp3624/tensorflow-multi-gpu-for-inferencing-test-time-58e952a2ed95). With [`tf.device`](https://www.tensorflow.org/api_docs/python/tf/device) one can assign on which device the network should perform computation. From the input we can create batches such that each batch is a tuple of inputs to all the gpus, and we use the batches to feedforward the network. With [`tf.distribute.Strategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy) the program can run a copy of the network on each GPU, splitting the input data between them.

Depending on the data, we can consider [pinning the data to GPU](http://deeplearnphysics.org/Blog/2018-10-02-Pinning-data-to-GPU.html).

Also, [docker compose](https://docs.docker.com/compose/) is recommended, so we can setup a proper logging method, as well as automatic health check.

# Known Issues

- During starting the script, sometimes I ran into CUDA related issues when Tensorflow loads the dll-s. The issue (which I suspect to be driver-related) disappears after the 4th or 5th try. Such an example error message:
```
Attempting to fetch value instead of handling error Internal: failed to get device attribute 13 for device 0:
CUDA_ERROR_UNKNOWN: unknown error
```

- Since the development was on Windows Home, I had to use VirtualBox for docker. However, Nvidia docker does not support Windows, so I didn't manage to test the GPU usage in a docker container.

# Development Details

Developed and tested in:
- Windows 10 Home version 1903 (18362.900) <sup>2</sup>
- Intel Core i7, 8GB RAM
- Nvidia GeForce GTX 950M
- PyCharm 2020.1.2 Community Edition
- Oracle VM VirtualBox 6.1.10 (for Docker)

---

*Norbert Hantos
2020. 06. 23.*

---

<sup>1</sup>*Maya Kabkab, Azadeh Alavi, Rama Chellappa: DCNNs on a Diet: Sampling Strategies for Reducing the Training Set Size*

<sup>2</sup>*I prefer Linux for developing projects, however, during the time of this project it was not an option.*
