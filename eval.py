
def test():
    opt = opts.parse_opt()

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # mnist = sio.loadmat('MNIST_data/mnist.mat')
    train_data =  mnist.train.images * 2.0 - 1.0
    train_label = mnist.train.labels
    loader = Dataset(train_data, train_label)
    test_data = mnist.test.images * 2.0 - 1.0
    test_label = mnist.test.labels
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
    D.load( opt.checkpoint_path + "/mlp_model")
    NUM_THREADS = 2
    tf_config = tf.ConfigProto()
    tf_config.intra_op_parallelism_threads=NUM_THREADS
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        
        model = EavGAN(D, opt, sess)
#        pdb.set_trace()
        iteration = opt.iteration
        checkpoint_path = os.path.join(opt.load_checkpoint_path, 'model.ckpt')
        checkpoint_path += "-" + str(iteration)
        model.load(checkpoint_path)
        epoch = opt.pretrain_epoch + 1
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