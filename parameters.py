import argparse


parser = argparse.ArgumentParser(description='parameters')

# -- Model parameters --
parser.add_argument('--clients_number', type=int, default=10,
                    help='default: 10 clients')
parser.add_argument('--clients_sample_ratio', type=float, default=0.1,
                    help='default: 0.1')
parser.add_argument('--epoch', type=int, default=50000,
                    help='default: 50000')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='default: 0.001')
parser.add_argument('--decay_rate', type=float, default=1,
                    help='default: 1')
parser.add_argument('--batch_size', type=int, default=50,
                    help='default: 50')
parser.add_argument('--clients_training_epoch', type=int, default=3,
                    help='default: 3')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='default: 0.5')

# dataset parameters
parser.add_argument('--dataset', default="cifar10",
                    help='default: cifar10')
parser.add_argument('--num_classes', type=int, default=10,
                    help='default: 10')

# If you set --distorted_data in your command, it means that you distort data.
parser.add_argument("--distorted_data", action='store_true', help='distort data or not')
parser.add_argument("--homo_flag", action='store_true', help='all the client models are homogeneous or not')
parser.add_argument("--GPU_num", type=int, default=1, help='default: 1')



args = parser.parse_args()