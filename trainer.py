
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
    if not os.path.exists(opt.image_path):
        os.mkdir(opt.image_path)
    if not os.path.exists(opt.checkpoint_path):
        os.mkdir(opt.checkpoint_path)
    if opt.input_data == "MNIST" :
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        train_data =  mnist.train.images * 2.0 - 1.0
        train_label = data.train.labels
        loader = Dataset(train_data, train_label)
        test_data = mnist.test.images * 2.0 - 1.0
        test_label = mnist.test.labels
    elif opt.input_data == "CIFAR":
        data = ....
    # mnist = sio.loadmat('MNIST_data/mnist.mat')

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
        print fine_tune
        if fine_tune:
            iteration = opt.iteration
            checkpoint_path = os.path.join(opt.load_checkpoint_path, 'model.ckpt')
            checkpoint_path += "-" + str(iteration)
            model.load(checkpoint_path)
            opt.train_adv = True
        else:
            #start from begin
            epoch = 0

        # Assign the learning rate
            
        # Assure in training mode
        while True:
            # pretrain at first
            start = time.time()
            # Load data from train split (0)
            data = loader.next_batch(batch_size)
            # print('Read data:', time.time() - start)
            if iteration != 0 \
                and ( iteration %  opt.learning_rate_decay_every )== 0:
            
                frac = (iteration - opt.learning_rate_decay_start) / opt.learning_rate_decay_every

                decay_factor = 0.5  ** frac
                sess.run(tf.assign( model.lr, opt.learning_rate * decay_factor))
            else:
                sess.run(tf.assign(model.lr, opt.learning_rate))
        

            if iteration < opt.pretrain_iteration or opt.train_adv == False:
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
                    print "iteration: ", iteration
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
                    print "loss: ", D_loss, G_loss
                    print "iteration: ", iteration
                    print "adv lr: ",  sess.run(model.lr)
                # Update the iteration and epoch
            iteration += 1
            # epoch = loader._epochs_completed
                
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
                with open(opt.image_path + "/" + "ld_" + str(opt.ld) +  "_result.txt", 'a') as fid :
                    log = "iteration:{}, attack rate: {} \n".format( str(iteration), str(acc / float(i * batch_size) ))
                    fid.write(log)
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.ckpt')
                model.saver.save(sess, checkpoint_path, global_step = iteration) 

if __name__ == "__main__":
    train()
# test()
