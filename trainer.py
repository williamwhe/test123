
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
from model import CNN
import model as Blackbox
import pdb
import opts 
from Dataset import Dataset
from utils import plot
from evagan import EvaGAN
import cifar10
from utils import save_images
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"

def train():
    flatten_flag = True ### the output of G need to flatten or not?
    opt = opts.parse_opt()
    if opt.input_data.upper() == "MNIST" :
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        train_data =  mnist.train.images * 2.0 - 1.0
        train_label = mnist.train.labels

        loader = Dataset(train_data, train_label)
        test_data = mnist.test.images * 2.0 - 1.0
        test_label = mnist.test.labels
        test_loader = Dataset(test_data, test_label)

        x_dim = train_data.shape[1]
        y_dim = train_label.shape[1]
        
        opt.input_c_dim = 1
        opt.output_c_dim = 1
        opt.input_dim = x_dim
        opt.label_dim = y_dim

    elif opt.input_data.upper() == "CIFAR10":
        flatten_flag = False
        cifar10.maybe_download_and_extract()
        images_train, cls_train, labels_train = cifar10.load_training_data()
        images_test, cls_test, labels_test = cifar10.load_test_data()
        assert( images_train.shape[1] == images_train.shape[2])
        assert( len( images_train.shape) == 4)
        #maybe need mapping to -1-1
        images_train = images_train * 2.0 - 1.0
        images_test = images_test * 2.0 - 1.0
        loader = Dataset( images_train, labels_train)
        test_loader = Dataset( images_test, labels_test)
      
        shapes = images_train.shape
        opt.img_dim = shapes[1]
        opt.input_c_dim = shapes[3]
        opt.output_c_dim = shapes[3]
        opt.label_dim = labels_train.shape[1]
        opt.input_dim = shapes[1]
        # opt.batch_size = 128

    #add prefix(dataset name) to the path 
    opt.image_path = opt.input_data + '/' + opt.image_path
    opt.checkpoint_path = opt.input_data + '/' +  opt.checkpoint_path
    opt.load_checkpoint_path = opt.input_data + '/' + opt.load_checkpoint_path


    if not os.path.exists(opt.image_path):
        os.mkdir(opt.image_path)
    if not os.path.exists(opt.checkpoint_path):
        os.mkdir(opt.checkpoint_path)
    batch_size = opt.batch_size
    fine_tune = opt.fine_tune

    NUM_THREADS = 2
    tf_config = tf.ConfigProto()
    tf_config.intra_op_parallelism_threads=NUM_THREADS
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config) as sess:
        # Initialize the variables, and restore the variables form checkpoint if there is.
        # and initialize the writer
        model = EvaGAN(opt, sess)
        D = Blackbox.setup(opt, sess)

        iteration = 0
        if fine_tune:
            print "FINE-TUNE"
            iteration = opt.iteration
            checkpoint_path = os.path.join(opt.load_checkpoint_path, 'model.ckpt')
            checkpoint_path += "-" + str(iteration)
            model.load(checkpoint_path)
            ###load negative samples
            checkpoint_path = opt.load_checkpoint_path + "/negative_"  + str(iteration) +".mat"
            loader.load_negative_sample(checkpoint_path)

        # Assign the learning rate
            
        # Assure in training mode
        end2 = time.time()
        sess.run(tf.assign( model.lr, opt.learning_rate ))
        while True:
            # pretrain at first
            # Load data from train split (0)
            # print('Read data:', time.time() - start)
            if iteration != 0 \
                and ( iteration %  opt.learning_rate_decay_every )== 0:
            
                frac = iteration / opt.learning_rate_decay_every

                decay_factor = 0.5  ** frac
                sess.run(tf.assign( model.lr, opt.learning_rate * decay_factor))


            if iteration < opt.pretrain_iteration and opt.train_adv == False:

                #stageI CGAN + look like loss
                # one to one mapping 
 
                data = loader.next_batch(batch_size, negative = False ) 
                #training without negative samples 
                feed = {model.source : data[0], model.target: data[0]}


                for _ in range(3):
                    G_loss, _ = sess.run([model.G_loss, model.G_pre_train_op], feed)
                D_loss, _ = sess.run([model.D_loss, model.D_pre_train_op], feed)

                 ### genrate negative samples;
                if iteration != 0 and iteration % opt.losses_log_every == 0:
                    start2 = time.time()
                    print "total time: ", start2 - end2 
                    print "loss", D_loss, G_loss
                    print "iteration: ", iteration
                    end2 = time.time()
            else:#stage II add adversarial loss
                data = loader.next_batch(batch_size, negative = True, priority = True) 
                feed = { model.source : data[0]}
                if flatten_flag:
                    sample = sess.run(model.fake_images_sample_flatten, feed)
                else:
                    sample = sess.run(model.fake_images_sample, feed)
                predict_labels = D.predict(sample, data[1]).reshape(batch_size, 1)
                ###select success 
                feed = {model.source :  data[0], \
                    model.predict_labels: predict_labels,\
                    model.negative_sample : data[2], \
                    model.target : data[0]}
                for _ in range(2):
                    adv_G_loss, G_loss, _ = sess.run([model.adv_G_loss, model.G_loss2, model.G_train_op], feed)
                adv_D_loss, D_loss,_ = sess.run([model.adv_D_loss, model.D_loss2, model.D_train_op], feed)
    
                if iteration != 0 and iteration % opt.losses_log_every == 0:    
                    start2 = time.time()
                    print "total time:", start2 - end2
                    print "loss(D, G, adv_D,adv_G): ", D_loss, G_loss, adv_D_loss, adv_G_loss
                    print "iteration: ", iteration
                    end2 = time.time()
            
            # when it can genearator "look like " sample, 
            # select negative samples from the generator 

            if iteration > opt.add_neg_iteration and iteration % opt.add_neg_every == 0:
                    
                feed = {model.source : data[0]}
                if flatten_flag:
                    sample = sess.run(model.fake_images_sample_flatten, feed)
                else:
                    sample = sess.run(model.fake_images_sample, feed)
                predict_labels = D.predict(sample, data[1]).reshape(batch_size, 1)
                atr_idx = np.where(predict_labels == 1)[0]
                # insert some negative samples
                
                magnitude = np.sum( abs( sample - data[0] ) ,axis = 1)
                loader.insert_negative_sample(sample[atr_idx, :],  data[1][atr_idx, :] , magnitude[atr_idx])
            # Update the iteration and epoch
            iteration += 1
                
            if (iteration != 0 and iteration % opt.save_checkpoint_every == 0):

                num = test_loader._num_examples
                iter_num = (num - batch_size )  / batch_size
                acc = 0.0
                if iter_num > 500:
                    iter_num = 500
                for i in range( iter_num):

                    s_imgs, s_label  = test_loader.next_batch(batch_size, negative = False)
            
                    feed = {model.source : s_imgs}
                    if flatten_flag:
                        sample = sess.run(model.fake_images_sample_flatten, feed)
                    else:
                        sample = sess.run(model.fake_images_sample, feed)
                
                    predict_label = D.predict( sample, s_label)
                    acc += float( np.sum(predict_label) ) 
                print "total accuracy: ", acc / float(i * batch_size)

                # eval model
                # Write validation result into summary
                s_imgs, s_label  = test_loader.get_all_images(batch_size)
                feed = {model.source : s_imgs}
                samples = sess.run(model.fake_images_sample, feed)

                save_images(samples[0:16], [4,  4], opt.image_path + '/{}.png'.format(str(iteration)) )
                # fig = plot(samples)
                # plt.savefig( opt.image_path + '/{}.png'.format(str(iteration)), bbox_inches='tight')
                ### save log into a file 
                with open(opt.image_path + "/" + "ld_" + str(opt.ld) +  "_result.txt", 'a') as fid :
                    log = "iteration:{}, attack rate: {} \n".format( str(iteration), str(acc / float(i * batch_size) ))
                    fid.write(log)
                # save model 
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model.ckpt')
                model.saver.save(sess, checkpoint_path, global_step = iteration) 
                # save negative samples
                loader.save_negative_sample(opt.checkpoint_path + "/negative_" + str(iteration) + ".mat" )

if __name__ == "__main__":
    train()
