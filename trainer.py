
import math
import numpy as np 
import tensorflow as tf
from ops import *
import os
import time
import scipy.io  as sio
import random
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# import os
# from discriminator import RandomForrest
from model import MLP
import pdb
import opts 
from Dataset import Dataset
from utils import plot
from evagan import EvaGAN

def train():
    opt = opts.parse_opt()

    # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mnist = sio.loadmat('MNIST_data/mnist.mat')
    train_data =  mnist['train_data']
    train_label = mnist['train_label']
    loader = Dataset(train_data, train_label)
    test_data = mnist['test_data']
    test_label = mnist['test_label']
    test_loader = Dataset(test_data, test_label)

    x_dim = train_data.shape[1]
    y_dim = train_label.shape[1]
    # pdb.set_trace()
    batch_size = opt.batch_size
    h_dim = opt.h_dim

    opt.input_c_dim = 1
    opt.output_c_dim = 1
    opt.input_dim = x_dim
    opt.label_dim = y_dim
    loader = Dataset(train_data, train_label)
    fine_tune = opt.fine_tune
    D = MLP(layers = (h_dim))
    if fine_tune:
        D.load( opt.checkpoint_path + "/mlp_model")
    else:
        D.train(train_data, train_label)
        D.save( opt.checkpoint_path + "/mlp_model")
    NUM_THREADS = 2
    tf_config = tf.ConfigProto()
    tf_config.intra_op_parallelism_threads=NUM_THREADS
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        # Initialize the variables, and restore the variables form checkpoint if there is.
        # and initialize the writer

        model = EvaGAN(D, opt, sess)
        iteration = 0
        if fine_tune:

            iteration = opt.iteration
            checkpoint_path = os.path.join(opt.load_checkpoint_path, 'model.ckpt')
            checkpoint_path += "-" + str(iteration)
            model.load(checkpoint_path)
            epoch = opt.pretrain_epoch + 1
        else:
            #start from begin
            epoch = 0
        
        # Assign the learning rate
        if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
            frac = (epoch - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
            decay_factor = 0.5  ** frac
            sess.run(tf.assign(model.lr, opt.learning_rate * decay_factor)) # set the decayed rate
            
        else:
            sess.run(tf.assign(model.lr, opt.learning_rate))
        # Assure in training mode
        while True:
            # pretrain at first
            start = time.time()
            # Load data from train split (0)
            data = loader.next_batch(batch_size)
            # print('Read data:', time.time() - start)
            if epoch < opt.pretrain_epoch:
            #stageI CGAN + look like loss
                start = time.time()

                feed = {model.source : data[0], model.target: data[0], model.label : data[1]}
                for _ in range(5):
                    G_loss, _ = sess.run([model.G_loss, model.G_pre_train_op], feed)
                D_loss, _ = sess.run([model.D_loss, model.D_pre_train_op], feed)

                end = time.time()

                if iteration != 0 and iteration % opt.losses_log_every == 0:
                    print "time: ", end - start 
                    print "loss", D_loss, G_loss
                    print "epoch", epoch, iteration
                    print "pretrain lr", sess.run(model.lr)
            else:
            #stage II add adversarial loss
                start = time.time()
                feed = {model.source : data[0], model.label : data[1]}
                sample = sess.run(model.fake_images_sample_flatten, feed)
                predict_labels = D.predict(sample, data[1]).reshape(batch_size, 1)
                
                feed = {model.source :  data[0], \
                    model.predict_labels: predict_labels,\
                    model.negative_sample : data[2], \
                    model.target : data[0]}
                
                G_loss, _ = sess.run([model.G_loss2, model.G_train_op], feed)
                D_loss, _ = sess.run([model.D_loss2, model.D_train_op], feed)
    
                end = time.time()

                if iteration != 0 and iteration % opt.losses_log_every == 0:
                    print "time: ", end - start 
                    print "loss", D_loss, G_loss
                    print "epoch", epoch, iteration
                # Update the iteration and epoch
            iteration += 1
            epoch = loader._epochs_completed
                
            if (iteration != 0 and iteration % opt.save_checkpoint_every == 0):

                num = test_loader._num_examples
                iter_num = (num - batch_size )  / batch_size
                acc = 0.0
                if iter_num > 500:
                    iter_num = 500
                for i in range( iter_num):

                    s_imgs, s_label, n_imgs, n_label  = test_loader.next_batch(batch_size)
            
                    feed = {model.source : s_imgs, model.label : s_label}
                    samples = sess.run(model.fake_images_sample_flatten, feed)
                    predict_label = D.predict( samples, s_label)
                    acc += float( np.sum(predict_label) ) 
                print "total accuracy: ", acc / float(i * batch_size)

                # eval model
                # Write validation result into summary
                s_imgs, s_label  = test_loader.get_all_images(batch_size)
                feed = {model.source : s_imgs, model.label : s_label}
                samples = sess.run(model.fake_images_sample, feed)
                fig = plot(samples)
                plt.savefig( opt.image_path + '/{}.png'.format(str(iteration)), bbox_inches='tight')
                ### save log into a file 
                with open(opt.checkpoint_path + "result.txt", 'a') as fid :
                    log = "iteration:{}, attack rate: {} \n".format( str(iteration), str(acc / float(i * batch_size) ))
                    fid.write(log)
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.ckpt')
                model.saver.save(sess, checkpoint_path, global_step = iteration)            


def test():
    opt = opts.parse_opt()
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mnist = sio.loadmat('MNIST_data/mnist.mat')
    train_data =  mnist['train_data']
    train_label = mnist['train_label']
    loader = Dataset(train_data, train_label)
    test_data = mnist['test_data']
    test_label = mnist['test_label']
    test_loader = Dataset(test_data, test_label)

    x_dim = train_data.shape[1]
    y_dim = train_label.shape[1]
    batch_size = opt.batch_size

    opt.input_c_dim = 1
    opt.output_c_dim = 1
    opt.input_dim = x_dim
    opt.label_dim = y_dim
    opt.img_dim = 28

    opts.str_iter = 7500
    # opts.str_iter = 2000
   
    D = MLP(layers = (h_dim))
    # D.train(train_data, train_label)
    # D.save("save/mlp_model")
    D.load("save/mlp_model")
    #pdb.set_trace()
    # D.predict( train_data[0:10], train_label[0:10])
    #pdb.set_trace()
    NUM_THREADS = 2
    tf_config = tf.ConfigProto()
    tf_config.intra_op_parallelism_threads=NUM_THREADS
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        # Initialize the variables, and restore the variables form checkpoint if there is.
        # and initialize the writer

        model = EavGAN(D, opts, sess)
#        pdb.set_trace()
        iteration = opts.str_iter
        checkpoint_path = os.path.join(opt.checkpoint_path, 'model.ckpt')
        checkpoint_path += "-" + str(iteration)
        model.load(checkpoint_path)
        num = test_loader._num_examples
        iter_num = (num - batch_size )  / batch_size
        acc = 0.0
        for i in range( iter_num):

            s_imgs, s_label, n_imgs, n_label  = test_loader.next_batch(batch_size)
    
            feed = {model.source : s_imgs, model.label : s_label}
            samples = sess.run(model.fake_images_sample_flatten, feed)
            predict_label = D.predict( samples, s_label)
            acc += float( np.sum(predict_label) ) 
            print predict_label 
            # s_label
            # pdb.set_trace()
            if i % 100 == 0:

#                fig = plot(samples, 28)
#                plt.savefig(opt.image_path + '/{}.png'.format(str(iteration)), bbox_inches='tight')

                print "accuracy: ", acc / float( batch_size *(i +1))
        print "total accuracy: ", acc / float(iter_num * batch_size)

              
train()
# test()
