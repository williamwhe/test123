import math
import numpy as np 
import tensorflow as tf
from ops import *
import os
import time
import scipy.io  as sio
import random

class EvaGAN():
    """
    GAN for generating malware.
    """
    def __init__(self,  D, opts, sess):
        """
        :param D: the discriminator object
        :param params: the dict used to train the generative neural networks
        """
        self.D = D
        self.opts = opts

        self._build_model()
        self.sess = sess
        self.saver = tf.train.Saver(max_to_keep =100000)
        self.sess.run(tf.global_variables_initializer())
        # self.sess.run(tf.local_variables_initializer())
        # self.sess.run(tf.initialize_all_variables())
    def load(self, checkpoint_path):
        self.saver.restore(self.sess, checkpoint_path)

    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out],
                                               stddev=stddev/tf.sqrt(float(dim_in) / 2.)), name=name)

    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)


    def _build_model(self):
        """ build a tensorflow model
        :return
        """
        ld = self.opts.ld
        L1_lambda = self.opts.L1_lambda
        self.gf_dim = self.opts.gf_dim
        self.df_dim = self.opts.df_dim

        ###batch normalization
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.d2_bn1 = batch_norm(name='adv_d_bn1')
        self.d2_bn2 = batch_norm(name='adv_d_bn2')
        self.d2_bn3 = batch_norm(name='adv_d_bn3')


        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
       
        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        
        self.lr = tf.Variable(0.001, trainable = False,  name = "learning_rate")

        input_c_dim = self.opts.input_c_dim
        self.output_c_dim = self.opts.output_c_dim
        input_dim = self.opts.input_dim
        label_dim = self.opts.label_dim
        img_dim = self.opts.img_dim
        self.batch_size = self.opts.batch_size
        # the source image which we want to attack.
        self.source = tf.placeholder(tf.float32, [None,  input_dim ],\
                                name='source_image')
        # resize
        self.image = tf.image.resize_bilinear(\
            tf.reshape( self.source, [-1, img_dim, img_dim, input_c_dim]), \
            [32, 32])
        # the target images, which has the different label. We can also conduct target attack later
        self.target = tf.placeholder(tf.float32, [None, input_dim],\
                                name='source_image')
        self.real_image = tf.image.resize_bilinear(\
            tf.reshape( self.target, [-1, img_dim, img_dim, input_c_dim]), \
            [32, 32])   
        ####for adversarial loss
        self.negative_sample = tf.placeholder(tf.float32, [None, input_dim ], \
                                name = "negative_samples")
        self.adv_image = tf.image.resize_bilinear(\
            tf.reshape( self.negative_sample, [-1, img_dim, img_dim, input_c_dim]), \
            [32, 32])   
        
        self.predict_labels = tf.placeholder(tf.float32, [None, 1], \
                                name = "predict_label")
        self.fake_images = self.generator(self.image)


        D_fake_loss, D_fake_logit =  self.discriminator(tf.concat( [self.fake_images, self.image], axis = 3) )
        D_real_loss, D_real_logit = self.discriminator(tf.concat( [self.real_image, self.image], axis = 3) , reuse = True)

        D_fake_loss2, D_fake_logit2 = self.discriminator2(tf.concat( [self.fake_images, self.image], axis = 3))
        D_real_loss2, D_real_logit2 = self.discriminator2(tf.concat( [self.adv_image, self.image], axis = 3), reuse = True) 


        self.fake_images_sample = self.sampler(self.image)

        self.fake_images_sample_flatten = tf.reshape( tf.image.resize_bilinear(\
            self.fake_images_sample, [28, 28]), [-1, 28 * 28 ]  )

        #go through a blackbox algorithm
        # self.predict_labels = self.D.predict(self.adv_A, self.real_A_label)
        # give the others 
        ####
        D_fake_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake_logit, labels = tf.zeros_like(D_fake_logit)))    
        D_real_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real_logit, labels = tf.ones_like(D_real_logit) ))

        # D_fake_loss2 = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake_logit2, labels = tf.zeros_like(D_fake_logit2)))    
        D_fake_loss2 = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits = D_fake_logit2, labels = self.predict_labels))    

        D_real_loss2 = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits = D_real_logit2, labels = tf.ones_like(D_real_logit2)))



        G_loss = tf.reduce_mean( \
            tf.nn.sigmoid_cross_entropy_with_logits(\
                logits=D_fake_logit, \
                labels=tf.ones_like(D_fake_logit)\
                )\
            )

        adv_G_loss= tf.reduce_mean( \
            tf.nn.sigmoid_cross_entropy_with_logits(\
                logits=D_fake_logit2, \
                labels=tf.ones_like(D_fake_logit2)\
                )\
            )


        ####
      
        self.G_loss = G_loss + L1_lambda * tf.reduce_mean( tf.abs(self.fake_images - self.image) )
        D_loss = D_real_loss + D_fake_loss
        self.D_loss = D_loss

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
     
        self.adv_d_vars = [var for var in t_vars if 'adv_d_' in var.name]
        
        D_pre_opt = tf.train.AdamOptimizer(self.lr)
        D_grads_and_vars_pre = D_pre_opt.compute_gradients(self.D_loss, self.d_vars)
        D_grads_and_vars_pre = [(tf.clip_by_value(gv[0], -1.0, 1.0), gv[1]) for gv in D_grads_and_vars_pre]
        #grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 1)
        self.D_pre_train_op = D_pre_opt.apply_gradients(D_grads_and_vars_pre)


        G_pre_opt = tf.train.AdamOptimizer(self.lr)
        G_grads_and_vars_pre = G_pre_opt.compute_gradients(self.G_loss, self.g_vars)
        #end 
        G_grads_and_vars_pre = [(tf.clip_by_value(gv[0], -1.0, 1.0), gv[1]) for gv in G_grads_and_vars_pre]
        self.G_pre_train_op = G_pre_opt.apply_gradients(G_grads_and_vars_pre)
    

        #adversarial perturbation 
        adv_D_loss = D_real_loss2 + D_fake_loss2
        self.adv_G_loss = adv_G_loss
        self.adv_D_loss = adv_D_loss

   
        self.G_loss2 = ld * G_loss + (1 - ld) * adv_G_loss + L1_lambda * tf.reduce_mean( tf.abs(self.fake_images - self.image) )
        self.D_loss2 = ld * D_loss  + (1 - ld) * adv_D_loss


        # self.G_loss2 = ld * self.G_loss + (1 - ld) * self.G_loss2
        # self.D_loss2 = ld * ( self.D_loss ) + (1 - ld)*(self.D_loss2)
        
        # gradient clipping
     
        D_opt = tf.train.AdamOptimizer(self.lr)
        D_grads_and_vars = D_opt.compute_gradients(self.D_loss2, self.adv_d_vars + self.d_vars)
        D_grads_and_vars = [(tf.clip_by_value(gv[0], -1.0, 1.0), gv[1]) for gv in D_grads_and_vars]
        #grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 1)
        self.D_train_op = D_opt.apply_gradients(D_grads_and_vars)


        G_opt = tf.train.AdamOptimizer(self.lr)

        G_grads_and_vars = G_opt.compute_gradients(self.G_loss2, self.g_vars)
        # G_grads_and_vars = G_opt.compute_gradients(self.G_loss, G_Ws + G_bs)
        # #end 
        G_grads_and_vars = [(tf.clip_by_value(gv[0], -1.0, 1.0), gv[1]) for gv in G_grads_and_vars]
        self.G_train_op = G_opt.apply_gradients(G_grads_and_vars)
        
        # #add new one 



    def discriminator(self, image, y = None, reuse = False):

        with tf.variable_scope("discriminator") as scope:
            s = 32
            s2, s4, s8, s16  = int(s/2), int(s/4), int(s/8), int(s/16)            
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim) 16 x 16 x df_dim
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            # h1 is (64 x 64 x self.df_dim*2) 8 x 8 x df_dim*2
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            # h2 is (32x 32 x self.df_dim*4)  4 x 4 x df_dim*4
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
            # pdb.set_trace()
            # h3 is (16 x 16 x self.df_dim*8) 4 x 4 x df_dim*8
            # h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
            h4 = linear(tf.reshape(h3, [-1, s8 * s8 * self.df_dim * 8]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    ####
    #adversarial discriminator
    ####
    def adv_discriminator(self, image, y = None, reuse = False):

        with tf.variable_scope("discriminator") as scope:
            s = 32
            # s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
            s2, s4, s8, s16  = int(s/2), int(s/4), int(s/8), int(s/16)
            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='adv_d_h0_conv'))
            # h0 is (128 x 128 x self.df_dim) 32 x 32
            h1 = lrelu(self.d2_bn1(conv2d(h0, self.df_dim*2, name='adv_d_h1_conv')))
            # h1 is (64 x 64 x self.df_dim*2) 16 x 16
            h2 = lrelu(self.d2_bn2(conv2d(h1, self.df_dim*4, name='adv_d_h2_conv')))
            # h2 is (32x 32 x self.df_dim*4)  8 x 8
            h3 = lrelu(self.d2_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='adv_d_h3_conv')))
            # h3 is (16 x 16 x self.df_dim*8) 4 x 4 
            h4 = linear(tf.reshape(h3, [-1, s8 * s8 * self.df_dim * 8]), 1, 'adv_d_h3_lin')

            return tf.nn.sigmoid(h4), h4


    def generator(self, image, y=None):
        with tf.variable_scope("generator") as scope:
            ###assume image input size is 32 
            s = 32
            # s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
            s2, s4, s8, s16  = int(s/2), int(s/4), int(s/8), int(s/16)
            #16, 8 ,4, 2,1 s:32
            # image is (32 x 32 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (16 x 16 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (8 x 8 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # # e3 is (4 x 4 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # # e4 is (2 x 2 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # # e5 is (1 x 1 x self.gf_dim*8)
            ##
            # MNIST version 
            ##            
            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e5),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e4], 3)
            # d1 is ( 2 x 2 x self.gf_dim*8*2)            

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e3], 3)
            # d2 is (4 x 4 x self.gf_dim*4*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e2], 3)
            # d3 is (8 x 8 x self.gf_dim*2*2)
            
            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e1], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s, s, self.output_c_dim], name='g_d5', with_w=True)
            return tf.nn.tanh(self.d5)
    
    def sampler(self, image, y=None):
        with tf.variable_scope("generator") as scope:
            
            tf.get_variable_scope().reuse_variables()
            #images = tf.image.resize_bilinear(image, [32, 32, 1])

            ###assume image input size is 32 
            s = 32
            # s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
            s2, s4, s8, s16  = int(s/2), int(s/4), int(s/8), int(s/16)
            #16, 8 ,4, 2,1 s:32
            # image is (32 x 32 x input_c_dim)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            # e1 is (16 x 16 x self.gf_dim)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            # e2 is (8 x 8 x self.gf_dim*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            # # e3 is (4 x 4 x self.gf_dim*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            # # e4 is (2 x 2 x self.gf_dim*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            # # e5 is (1 x 1 x self.gf_dim*8)
            ##
            # MNIST version 
            ##            
            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e5),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e4], 3)
            # d1 is ( 2 x 2 x self.gf_dim*8*2)            

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e3], 3)
            # d2 is (4 x 4 x self.gf_dim*4*2)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e2], 3)
            # d3 is (8 x 8 x self.gf_dim*2*2)
            
            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e1], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s, s, self.output_c_dim], name='g_d5', with_w=True)
            return tf.nn.tanh(self.d5)