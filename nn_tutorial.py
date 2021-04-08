# -*- coding: utf-8 -*-
"""
What is `torch.nn` *really*?
============================
by Jeremy Howard, `fast.ai <https://www.fast.ai>`_. Thanks to Rachel Thomas and Francisco Ingham.
Adapted as a non-web version by Keigh Rim.
Original version is available at https://github.com/pytorch/tutorials/blob/cecea517b5a0d5d2ddbcb5de0ec2733f33722aae/beginner_source/nn_tutorial.py
"""
# PyTorch provides the elegantly designed modules and classes
# to help you create and train neural networks.
# In order to fully utilize their power and customize
# them for your problem, you need to really understand exactly what they're
# doing. To develop this understanding, we will first train basic neural net
# on the MNIST data set without using any features from these models; we will
# initially only use the most basic PyTorch tensor functionality. Then, we will
# incrementally add one feature from ``torch.nn``, ``torch.optim``, ``Dataset``, or
# ``DataLoader`` at a time, showing exactly what each piece does, and how it
# works to make the code either more concise, or more flexible.
#
# **This tutorial assumes you already have PyTorch installed, and are familiar
# with the basics of tensor operations.** (If you're familiar with Numpy array
# operations, you'll find the PyTorch tensor operations used here nearly identical).
#

import gzip
import pickle
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from matplotlib import pyplot
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

# MNIST data setup
# ----------------
#
# We will use the classic `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset,
# which consists of black-and-white images of hand-drawn digits (between 0 and 9).

DATA_PATH = Path("data") / "mnist"

DATA_PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (DATA_PATH / FILENAME).exists():
    urllib.request.urlretrieve(URL+FILENAME, (DATA_PATH / FILENAME))

# This dataset is in numpy array format, and has been stored using pickle,
# a python-specific format for serializing data.

with gzip.open((DATA_PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

# Each image is 28 x 28, and is being stored as a flattened row of length
# 784 (=28x28). Let's take a look at one; we need to reshape it to 2d
# first.

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
pyplot.show()
print(f'Training data shape: {x_train.shape}')

# PyTorch uses ``torch.tensor``, rather than numpy arrays, so we need to
# convert our data.

x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
num_instances, num_features = x_train.shape

# Neural net from scratch (no torch.nn)
# ---------------------------------------------
#
# Let's first create a model using nothing but PyTorch tensor operations.
#
# PyTorch provides methods to create random or zero-filled tensors, which we will
# use to create our weights and bias for a simple logistic regression model. These are just regular
# tensors, with one very special addition: we tell PyTorch that they require a
# gradient. This causes PyTorch to record all of the operations done on the tensor,
# so that it can calculate the gradient during back-propagation *automatically*!
#
# For the weights, we set ``requires_grad`` **after** the initialization, since we
# don't want that step included in the gradient. (Note that a trailing ``_`` in
# PyTorch signifies that the operation is performed in-place.)
#
# .. note:: We are initializing the weights here with
#    `Xavier initialization <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_
#    (by multiplying with 1/sqrt(n)).


# Thanks to PyTorch's ability to calculate gradients automatically, we can
# use any standard Python function (or callable object) as a model! So
# let's just write a plain matrix multiplication and broadcasted addition
# to create a simple linear model. We also need an activation function, so
# we'll write `log_softmax` and use it. Remember: although PyTorch
# provides lots of pre-written loss functions, activation functions, and
# so forth, you can easily write your own using plain python. PyTorch will
# even create fast GPU or vectorized CPU code for your function
# automatically.

# Using torch.nn.functional
# ------------------------------
#
# We will now refactor our code, so that it does the same thing as before, only
# we'll start taking advantage of PyTorch's ``nn`` classes to make it more concise
# and flexible. At each step from here, we should be making our code one or more
# of: shorter, more understandable, and/or more flexible.
#
# The first and easiest step is to make our code shorter by replacing our
# hand-written activation and loss functions with those from ``torch.nn.functional``
# (which is generally imported into the namespace ``F`` by convention). This module
# contains all the functions in the ``torch.nn`` library (whereas other parts of the
# library contain classes). As well as a wide range of loss and activation
# functions, you'll also find here some convenient functions for creating neural
# nets, such as pooling functions. (There are also functions for doing convolutions,
# linear layers, etc, but as we'll see, these are usually better handled using
# other parts of the library.)
#
# If you're using negative log likelihood loss and log softmax activation,
# then Pytorch provides a single function ``F.cross_entropy`` that combines
# the two. So we can even remove the activation function from our model.

loss_func = F.cross_entropy

# Refactor using nn.Module
# -----------------------------
# Next up, we'll use ``nn.Module`` and ``nn.Parameter``, for a clearer and more
# concise training loop. We subclass ``nn.Module`` (which itself is a class and
# able to keep track of state).  In this case, we want to create a class that
# holds our weights, bias, and method for the forward step.  ``nn.Module`` has a
# number of attributes and methods (such as ``.parameters()`` and ``.zero_grad()``)
# which we will be using.
#
# .. note:: ``nn.Module`` (uppercase M) is a PyTorch specific concept, and is a
#    class we'll be using a lot. ``nn.Module`` is not to be confused with the Python
#    concept of a (lowercase ``m``) `module <https://docs.python.org/3/tutorial/modules.html>`_,
#    which is a file of Python code that can be imported.

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        # Refactor using nn.Linear
        # -------------------------
        #
        # We continue to refactor our code.  Instead of manually defining and
        # initializing ``self.weights`` and ``self.bias``, and calculating ``xb  @
        # self.weights + self.bias``, we will instead use the Pytorch class
        # `nn.Linear <https://pytorch.org/docs/stable/nn.html#linear-layers>`_ for a
        # linear layer, which does all that for us. Pytorch has many types of
        # predefined layers that can greatly simplify our code, and often makes it
        # faster too.
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)

# Since we're now using a class instead of just using a function, we
# first have to instantiate our model:

model = Mnist_Logistic()

# In the above, the ``@`` stands for the dot product operation. We will call
# our function on one batch of data (in this case, 64 images).  This is
# one *forward pass*.  Note that our predictions won't be any better than
# random at this stage, since we start with random weights.

batch_size = 64  # batch size

xb = x_train[0:batch_size]  # a mini-batch from x
predictions = model(xb)  # predictions

yb = y_train[0:batch_size]
print(f'Loss at the beginning: {loss_func(predictions, yb)}')


# Let's also implement a function to calculate the accuracy of our model.
# For each prediction, if the index with the largest value matches the
# target value, then the prediction was correct.

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

print(f'Accuracy at the beginning: {accuracy(predictions, yb):.2f} %')

# We can now run a training loop.  For each iteration, we will:
#
# - select a mini-batch of data (of size ``batch_size``)
# - use the model to make predictions
# - calculate the loss
# - ``loss.backward()`` updates the gradients of the model, in this case, ``weights``
#   and ``bias``.
#
# We now use these gradients to update the weights and bias.  We do this
# within the ``torch.no_grad()`` context manager, because we do not want these
# actions to be recorded for our next calculation of the gradient.  You can read
# more about how PyTorch's Autograd records operations
# `here <https://pytorch.org/docs/stable/notes/autograd.html>`_.
#
# We then set the gradients to zero, so that we are ready for the next loop.
# Otherwise, our gradients would record a running tally of all the operations
# that had happened (i.e. ``loss.backward()`` *adds* the gradients to whatever is
# already stored, rather than replacing them).
#

lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

# Refactor using optim
# ------------------------------
#
# Pytorch also has a package with various optimization algorithms, ``torch.optim``.
# We can use the ``step`` method from our optimizer to take a forward step, instead
# of manually updating each parameter.
#


opt = optim.SGD(model.parameters(), lr=lr)

# Refactor using Dataset
# ------------------------------
#
# PyTorch has an abstract Dataset class.  A Dataset can be anything that has
# a ``__len__`` function (called by Python's standard ``len`` function) and
# a ``__getitem__`` function as a way of indexing into it.
# `This tutorial <https://pytorch.org/tutorials/beginner/data_loading_tutorial.html>`_
# walks through a nice example of creating a custom ``FacialLandmarkDataset`` class
# as a subclass of ``Dataset``.
#
# PyTorch's `TensorDataset <https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#TensorDataset>`_
# is a Dataset wrapping tensors. By defining a length and way of indexing,
# this also gives us a way to iterate, index, and slice along the first
# dimension of a tensor. This will make it easier to access both the
# independent and dependent variables in the same line as we train.

# Both ``x_train`` and ``y_train`` can be combined in a single ``TensorDataset``,
# which will be easier to iterate over and slice.

# Refactor using DataLoader
# ------------------------------
#
# Pytorch's ``DataLoader`` is responsible for managing batches. You can
# create a ``DataLoader`` from any ``Dataset``. ``DataLoader`` makes it easier
# to iterate over batches. Rather than having to use ``train_ds[i*bs : i*bs+bs]``,
# the DataLoader gives us each minibatch automatically.

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=batch_size)

def fit():
    for epoch in range(epochs):
        # Now, our loop is much cleaner, as (xb, yb) are loaded automatically from the data loader:
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            opt.step()
            opt.zero_grad()

# (``optim.zero_grad()`` resets the gradient to 0 and we need to call it before
# computing the gradient for the next minibatch.)

# That's it: we've created and trained a minimal neural network (in this case, a
# logistic regression, since we have no hidden layers) entirely from scratch!
#
# Let's check the loss and accuracy and compare those to what we got
# earlier. We expect that the loss will have decreased and accuracy to
# have increased, and they have.
fit()

print(f'Loss at the end: {loss_func(model(xb), yb)}')
print(f'Accuracy on the validation set: {accuracy(model(x_valid), y_valid):.2f} %')

