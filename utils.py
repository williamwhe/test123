import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data


def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.
    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]
    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.
    :param num_classes:
        Number of classes. If None then use max(class_numbers)+1.
    :return:
        2-dim array of shape: [len(class_numbers), num_classes]
    """

    # Find the number of classes if None is provided.
    # Assumes the lowest class-number is zero.
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1

    return np.eye(num_classes, dtype=float)[class_numbers]


def process_mnist(fname = "MNIST_data/mnist.mat"):

	d = sio.loadmat(fname)
	train_data = d['train_data']
	test_data = d['test_data']
	train_label = d['train_label']
	test_label = d['test_label']
	train_data = train_data * 2.0- 1.0
	test_data = test_data * 2.0 - 1.0
	sio.savemat( fname, {\
		"train_data" : train_data, \
		"test_data" : test_data, \
		"train_label" : train_label,\
		"test_label" : test_label})

def plot(samples, img_dim = 32):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(img_dim, img_dim), cmap='Greys_r')

    return fig

def download_preprocess(fname = "MNIST_data/mnist.mat"):
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
	train_data = mnist.train.images *2.0 -1.0
	test_data = mnist.test.images *2.0 -1.0
	train_label = mnist.train.labels
	test_label = mnist.test.labels
	sio.savemat( fname, {\
		"train_data" : train_data, \
		"test_data" : test_data, \
		"train_label" : train_label,\
		"test_label" : test_label})
