import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run Ensemble.")
    parser.add_argument('--resume', type=bool, default=False,
                        help='Wether reload trained parameters')
    parser.add_argument('--resume_batch', type=int, default=134,
                        help='Which batch of saved model to load')
    parser.add_argument('--data_path', type=str, default='./data/', 
                            help='Data path')
    parser.add_argument('--data_name', nargs='?', default='glass',
                        help='Input data name.')
    parser.add_argument('--max_epochs', type=int, default=10,
                        help='Stop training generator after stop_epochs.')
    parser.add_argument('--print_epochs', type=int, default=1,
                        help='print the loss per print_epochs.')
    parser.add_argument('--lr_g', type=float, default=0.001,
                        help='Learning rate of generator.')  
    parser.add_argument('--lr_d', type=float, default=0.001,
                        help='Learning rate of discriminator.')
    parser.add_argument('--active_rate', type=float, default=1,
                        help='the proportion of instances that need to be labeled.')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='batch size.')
    parser.add_argument('--kFold', type=int, default=5,
                        help='k for k-fold cross validation.')
    parser.add_argument('--dim_z', type=int, default=128,
                        help='dim for latent noise.')
    parser.add_argument('--dis_layer', type=int, default=1, 
                        help='hidden_layer number in dis.')
    parser.add_argument('--dis_activation_func', type=str, default="relu",
                        help='activation function on discriminator, include relu,sigmoid,tanh')
    parser.add_argument('--gen_layer', type=int, default=2,
                        help='hidden_layer number in gen.')
    parser.add_argument('--ensemble_num', type=int, default=10,
                        help='the number of dis in ensemble.')
    parser.add_argument('--device', nargs='?', default="CPU", 
                        help='used device,including CPU,GPU,Ascend')
    parser.add_argument('--device_id', type=int, default=0, 
                        help='device id of used device')
    parser.add_argument('--input_path', type=str, default='data_csv')
    parser.add_argument('--SN_used', type=bool, default=False,
                        help='if spectral Normalization used')
    parser.add_argument('--init_type', nargs='?', default="N02",
                        help='init method for both gen and dis, including ortho,N02,xavier')
    parser.add_argument('--print', type=bool, default=True,
                        help='Print the learning procedure')
    return parser.parse_args()