# Sharpened Cosine Similarity
A layer implementation for PyTorch

## Install
At your command line:

```bash
git clone https://github.com/brohrer/sharpened_cosine_similarity_torch.git
```

## Demo
Run the Fashion MNIST demo to see sharpened cosine similarity in action.

If `python3` is the command you use to invoke Python at your command line:
```bash
cd sharpened_cosine_similarity_torch
python3 demo_fashion_mnist.py
```

## Monitor
```bash
python3 show_results.py
```

This will give a little console summary like this

```
testing errors for version test
mean  : 14.08%
stddev: 0.1099%
stderr: 0.03887%
n runs: 8
```

and drop a couple of plots in the `plots` directory showing how the
classification error on the test data set decreases with each pass through
the training data set.


## Credit where it's due
Based on  and copy/pasted heavily from [code](https://github.com/ZeWang95/scs_pytorch/blob/main/scs.py)
from [Ze Wang](https://twitter.com/ZeWang46564905/status/1488371679936057348?s=20&t=lB_T74PcwZmlJ1rrdu8tfQ)
and [code](https://github.com/oliver-batchelor/scs_cifar/blob/main/src/scs.py)
from [Oliver Batchelor](https://twitter.com/oliver_batch/status/1488695910875820037?s=20&t=QOnrCRpXpOuC0XHApi6Z7A)
and the TensorFlow [implementation](https://colab.research.google.com/drive/1Lo-P_lMbw3t2RTwpzy1p8h0uKjkCx-RB)
and [blog post](https://www.rpisoni.dev/posts/cossim-convolution/)
from [Raphael Pisoni](https://twitter.com/ml_4rtemi5).
