# Why Spectral Normalization Stabilizes GANs: Analysis and Improvements

**[[paper (NeurIPS 2021)](https://openreview.net/forum?id=MLT9wFYMlJ9)]**
**[[paper (arXiv)](https://arxiv.org/abs/2009.02773)]**
**[[code](https://github.com/fjxmlzn/BSN)]**


**Authors:** [Zinan Lin](http://www.andrew.cmu.edu/user/zinanl/), [Vyas Sekar](https://users.ece.cmu.edu/~vsekar/), [Giulia Fanti](https://www.andrew.cmu.edu/user/gfanti/)

**Abstract:** Spectral normalization (SN) is a widely-used technique for improving the stability and sample quality of Generative Adversarial Networks (GANs). However, there is currently limited understanding of why SN is effective. In this work, we show that SN controls two important failure modes of GAN training: exploding and vanishing gradients. Our proofs illustrate a (perhaps unintentional) connection with the successful LeCun initialization. This connection helps to explain why the most popular implementation of SN for GANs requires no hyper-parameter tuning, whereas stricter implementations of SN have poor empirical performance out-of-the-box. Unlike LeCun initialization which only controls gradient vanishing at the beginning of training, SN preserves this property throughout training. Building on this theoretical understanding, we propose a new spectral normalization technique: Bidirectional Scaled Spectral Normalization (BSSN), which incorporates insights from later improvements to LeCun initialization: Xavier initialization and Kaiming initialization. Theoretically, we show that BSSN gives better gradient control than SN. Empirically, we demonstrate that it outperforms SN in sample quality and training stability on several benchmark datasets.

---
This repo contains the codes for reproducing the experiments of our BSN and different SN variants in the paper. The codes were tested under Python 2.7.5, TensorFlow 1.14.0.

## Preparing datasets

### CIFAR10
Download `cifar-10-python.tar.gz` from [https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) (or from other sources).

### STL10
Download `stl10_binary.tar.gz` from [http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz](http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz) (or from other sources), and put it in `dataset_preprocess/STL10` folder. Then run `python preprocess.py`. This code will resize the images into 48x48x3 format, and save the images in `stl10.npy`.


### CelebA
Download `img_align_celeba.zip` from [https://www.kaggle.com/jessicali9530/celeba-dataset](https://www.kaggle.com/jessicali9530/celeba-dataset) (or from other sources), and put it in `dataset_preprocess/CelebA` folder. Then run `python preprocess.py`. This code will crop and resize the images into 64x64x3 format, and save the images in `celeba.npy`.


### ImageNet
Download `ILSVRC2012_img_train.tar` from [http://www.image-net.org/](http://www.image-net.org/) (or from other sources), and put it in `dataset_preprocess/ImageNet` folder. Then run `python preprocess.py`. This code will crop and resize the images into 128x128x3 format, and save the images in `ILSVRC2012`folder. Each subfolder in `ILSVRC2012` folder corresponds to one class. Each npy file in the subfolders corresponds to an image.


## Training BSN and SN variants
### Prerequisites

The codes are based on [GPUTaskScheduler](https://github.com/fjxmlzn/GPUTaskScheduler) library, which helps you automatically schedule the jobs among GPU nodes. Please install it first. You may need to change GPU configurations according to the devices you have. The configurations are set in `config.py` in each directory. Please refer to [GPUTaskScheduler's GitHub page](https://github.com/fjxmlzn/GPUTaskScheduler) for the details of how to make proper configurations.

> You can also run these codes without GPUTaskScheduler. Just run `python gan.py` in `gan` subfolders.

### CIFAR10, STL10, CelebA

#### Preparation

Copy the preprocessed datasets from the previous steps into the following paths:

* CIFAR10: `<code folder>/data/CIFAR10/cifar-10-python.tar.gz`.
* STL10: `<code folder>/data/STL10/cifar-10-stl10.npy`.
* CelebA: `<code folder>/data/CelebA/celeba.npy`.

Here `<code folder>` means

* Vanilla SN and our proposed BSSN/SSN/BSN without gammas: `no_gamma-CNN`.
* SN with the same gammas: `same_gamma-CNN`.
* SN with different gammas: `diff_gamma-CNN`.

Alternatively, you can directly modify the dataset paths in `<code folder>/gan_task.py` to the path of the preprocessed dataset folders.

#### Running codes

Now you can directly run `python main.py` in each `<code folder>` to train the models.

All the configurable hyper-parameters can be set in `config.py`. The hyper-parameters in the file are already set for reproducing the results in the paper. Please refer to [GPUTaskScheduler's GitHub page](https://github.com/fjxmlzn/GPUTaskScheduler) for the details of the grammar of this file.

### ImageNet

#### Preparation

Copy the preprocessed folder `ILSVRC2012` from the previous steps to `<code folder>/data/imagenet/ILSVRC2012`, where `<code folder>` means

* Vanilla SN and our proposed BSSN/SSN/BSN without gammas: `no_gamma-ResNet`.

Alternatively, you can directly modify the dataset path in `<code folder>/gan_task.py` to the path of the preprocessed folder `ILSVRC2012`.

#### Running codes

Now you can directly run `python main.py` in each `<code folder>` to train the models.

All the configurable hyper-parameters can be set in `config.py`. The hyper-parameters in the file are already set for reproducing the results in the paper. Please refer to [GPUTaskScheduler's GitHub page](https://github.com/fjxmlzn/GPUTaskScheduler) for the details of the grammar of this file.

The code supports multi-GPU training for speed-up, by separating each data batch equally among multiple GPUs. To do that, you only need to make minor modifications in `config.py`. For example, if you have two GPUs with IDs 0 and 1, then all you need to do is to (1) change `"gpu": ["0"]` to `"gpu": [["0", "1"]]`, and (2) change `"num_gpus": [1]` to `"num_gpus": [2]`. Note that the number of GPUs might influence the results because in this implementation the batch normalization layers on different GPUs are independent. In our experiments, we were using only one GPU.

### Results

The code generates the following result files/folders:

* `<code folder>/results/<hyper-parameters>/worker.log`: Standard output and error from the code.
* `<code folder>/results/<hyper-parameters>/metrics.csv`: Inception Score and FID during training.
* `<code folder>/results/<hyper-parameters>/sample/*.png`: Generated images during training.
* `<code folder>/results/<hyper-parameters>/checkpoint/*`: TensorFlow checkpoints.
* `<code folder>/results/<hyper-parameters>/time.txt`: Training iteration timestamps.
