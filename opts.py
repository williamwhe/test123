import argparse

def parse_opt():

    parser = argparse.ArgumentParser()
    # Data input settings

    parser.add_argument('--ld', type=int, default = 0.5,
                    help='lambda_ratio')      
    parser.add_argument('--h_dim', type=int, default = 128,
                    help='hidden_dim')       
    parser.add_argument('--batch_size', type=int, default = 16,
                    help='batch_size')       
    parser.add_argument('--input_c_dim', type=int, default = 1,
                    help='input_channel_dim')       
    parser.add_argument('--output_c_dim', type=int, default = 1,
                    help='output_channel_dim')       
    parser.add_argument('--gf_dim', type=int, default = 8,
                    help='generator_filter_dim')       
    parser.add_argument('--df_dim', type=int, default = 8,
                    help='discriminator_filter_dim')       
    parser.add_argument('--iteration', type=int, default = 35000,
                    help='load_iteration_number')       
    parser.add_argument('--fine_tune', type=bool, default=True,
                    help='fine_tune')
    parser.add_argument('--img_dim', type=int, default = 28,
                    help='image_w_h_dim')      


    parser.add_argument('--image_path', type=str, default = 'out5',
                    help='image_path')       
    parser.add_argument('--load_checkpoint_path', type=str, default='save5',
                    help='directory to store checkpointed models')
    parser.add_argument('--checkpoint_path', type=str, default='save5',
                    help='directory to store checkpointed models')
    parser.add_argument('--L1_lambda', type=float, default = 1,
                    help='L1_lambda')               
    parser.add_argument('--str_iter', type=int, default=7500,
                    help='start_iteration')    
    parser.add_argument('--pretrain_epoch', type=float, default=100,
                    help='pretrain epoch')
    parser.add_argument('--learning_rate', type=float, default=4e-4,
                    help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1, 
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--decay_iteration_max', type=int, default= 10000, help='decay iteration max')
    parser.add_argument('--learning_rate_decay_every', type=int, default=10, 
                    help='every how many iterations thereafter to drop LR by half?(in epoch)')

    args = parser.parse_args()

    return args
