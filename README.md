# Sharpened Cosine Similarity
An alternative to convolution for neural networks

## Implementations

[PyTorch](https://github.com/brohrer/sharpened-cosine-similarity/tree/main/pytorch)

[Keras](https://github.com/brohrer/sharpened-cosine-similarity/tree/main/keras)

[Jax](https://github.com/brohrer/sharpened-cosine-similarity/tree/main/jax)


## Background

One day there will be a comprehensive post and paper describing sharpened cosine similarity (SCS), but today is not that day. 

The idea behind sharpened cosine similarity first surfaced as
[a Twitter thread](https://twitter.com/_brohrer_/status/1232063619657093120)
in 2020. There's some more development in this [blog post](https://www.rpisoni.dev/posts/cossim-convolution-part2/). 

## Tips and Tricks

These are some things that have been reported to work so far.

* The big benefit of SCS appears to be parameter efficiency and architecture simplicity. 
It doesn't look like it's going to beat any accuracy records, and it doesn't always run very fast, but it's killing
in this [parameter efficiency leaderboard](https://github.com/brohrer/parameter_efficiency_leaderboard).
* Skip the nonlinear activation layers, like ReLU and sigmoid, after SCS layers.
* Skip the dropout layers after SCS layers.
* Skip the normalization layers, like batch normalization or layer normalization, after SCS layers.
* Use MaxAbsPool instead of MaxPool. It selects the element with the highest magnitude of activity, even if it's negative.
* Raising activities to the power p generally doesn't parallelize well on GPUs and TPUs. It will slow your code down a LOT compared to straight convolutions. Disabling the p parameters results in a huge speedup on GPUs, but this takes the "sharpened" out of SCS. Regular old cosine similarity is cool, but it is its own thing with its own limitations.

## Examples
In the age of gargantuan language models, it's uncommon to talk about how *few* parameters a model uses,
but it matters when you hope to deploy on compute- or power-limited devices. Sharpened cosine similarity
is exceptionally parameter efficient.

The repository <a href="https://github.com/brohrer/scs_torch_gallery">scs_torch_gallery</a>
has a handful of working examples.
[`cifar10_80_25214.py`](https://github.com/brohrer/scs-gallery/blob/main/cifar10_80_25214.py) is an image
classification model that gets 80% accuracy on CIFAR 10, using only 25.2k parameters.
According to the [CIFAR-10 Papers With Code](https://paperswithcode.com/sota/image-classification-on-cifar-10?dimension=PARAMS)
this is somewhere around one-tenth of the parameters in previous models in this accuracy range.

## Reverse Chronology

| Date | Milestone |
| ------------- | ------------- |
| 2022-03-28 | [Code]( https://colab.research.google.com/drive/1KUKFEMneQMS3OzPYnWZGkEnry3PdzCfn?usp=sharing) by [Raphael Pisoni]( https://twitter.com/ml_4rtemi5/status/1508341568188649474?s=20&t=YdSrNvUI-zqmaB83nwk42Q). Jax implementation. |
| 2022-03-11 | [Code](https://github.com/brohrer/sharpened-cosine-similarity/blob/main/pytorch/demo_fashion_mnist_lightning.py) by [Phil Sodmann](https://twitter.com/PSodmann). PyTorch Lightning demo on the Fashion MNIST data. |
| 2022-02-25 | [Experiments and analysis](https://twitter.com/_clashluke/status/1497092150906941442) by [Lucas Nestler](https://github.com/ClashLuke/) . TPU implementation of SCS. Runtime performance comparison with and without the p parameter |
| 2022-02-24 | [Code](https://github.com/DrJohnWagner/Kaggle-Notebooks) by [Dr. John Wagner](https://twitter.com/DrJohnWagner). Head to head comparison with convnet on American Sign Language alphabet dataset. |
| 2022-02-22 | [Code](https://github.com/hukkelas/sharpened_cosine_similarity_torch/blob/main/sharpened_cosine_similarity.py) by [Håkon Hukkelås](https://github.com/hukkelas). Reimplementation of SCS in PyTorch with a performance boost from using Conv2D. Achieved 91.3% CIFAR-10 accuracy with a model of 1.2M parameters. |
| 2022-02-21 | [Code](https://github.com/zimonitrome/scs_gan) by [Zimonitrome](https://twitter.com/zimonitrome/status/1495906518876794881?s=20&t=f8MNbUaIMWB4qhWChDZoEw). An SCS-based GAN, the first of its kind. |
| 2022-02-20 | [Code](https://github.com/brohrer/sharpened_cosine_similarity_torch/pull/6) by [Michał Tyszkiewicz](https://twitter.com/jatentaki/status/1495520542295789570?s=20&t=f8MNbUaIMWB4qhWChDZoEw). Reimplementation of SCS in PyTorch with a performance boost from using Conv2D. |
| 2022-02-20 | [Code](https://gist.github.com/ClashLuke/8f6521deef64789e76334f1b72a70d80) by [Lucas Nestler](https://twitter.com/_clashluke/status/1495534576399175680?s=20&t=f8MNbUaIMWB4qhWChDZoEw). Reimplementation of SCS in PyTorch with a performance boost and CUDA optimizations. |
| 2022-02-18 | [Blog post](https://www.rpisoni.dev/posts/cossim-convolution-part2/) by [Raphael Pisoni](https://twitter.com/ml_4rtemi5/status/1494651965036548099?s=20&t=pOd3c_k9VWHlUtMTh-9WtA). SOTA parameter efficiency on MNIST. Intuitive feature interpretation. |
| 2022-02-01 | [PyTorch code](https://github.com/StephenHogg/SCS) by [Stephen Hogg](https://twitter.com/whistle_posse/status/1488656595114663939?s=20&t=lB_T74PcwZmlJ1rrdu8tfQ). PyTorch implementation of SCS. MaxAbsPool implementation. |
| 2022-02-01 | [PyTorch code](https://github.com/oliver-batchelor/scs_cifar) by [Oliver Batchelor](https://twitter.com/oliver_batch/status/1488695910875820037?s=20&t=QOnrCRpXpOuC0XHApi6Z7A). PyTorch implementation of SCS. |
| 2022-01-31 | [PyTorch code](https://github.com/ZeWang95/scs_pytorch) by [Ze Wang](https://twitter.com/ZeWang46564905/status/1488371679936057348?s=20&t=lB_T74PcwZmlJ1rrdu8tfQ). PyTorch implementation of SCS. |
| 2022-01-30 | [Keras](https://colab.research.google.com/drive/1zeh6_Opjehy_EUwnBDHtyWIC74dxfBu1) code by [Brandon Rohrer](https://twitter.com/_brohrer_/status/1487928078437396484?s=20&t=pOd3c_k9VWHlUtMTh-9WtA). Keras implementation of SCS running on Fashion MNIST. |
| 2022-01-17 | [Code](https://colab.research.google.com/drive/1Lo-P_lMbw3t2RTwpzy1p8h0uKjkCx-RB) by [Raphael Pisoni](https://twitter.com/ml_4rtemi5). Implementation of SCS in paired depthwise/pointwise configuration, the key element of the [ConvMixer](https://arxiv.org/pdf/2201.09792v1.pdf) architecture. |
| 2022-01-06 | [Keras code](https://gist.github.com/4rtemi5/607909e6ac1ef3cfb54d5b85111f92b9) by [Raphael Pisoni](https://gist.github.com/4rtemi5). Keras implementation of SCS. |
| 2020-02-24 | [Twitter thread](https://twitter.com/_brohrer_/status/1232063619657093120) by [Brandon Rohrer](https://twitter.com/_brohrer_). Justification and introduction of SCS.  |
