
import os

from collections import namedtuple
import tensorflow as tf

from parameters import args
from modules.Server import Server as server


def main():
    parameters = namedtuple('parameters', ['clients_number',
                                           'clients_sample_ratio',
                                           'epoch',
                                           'learning_rate',
                                           'decay_rate',
                                           'num_input',
                                           'num_input_channel',
                                           'num_classes',
                                           'batch_size',
                                           'clients_training_epoch',
                                           'distorted_data',
                                           'dataset' ])
    
    clients_number = args.clients_number
    clients_sample_ratio = args.clients_sample_ratio
    epoch = args.epoch
    learning_rate = args.learning_rate
    decay_rate = args.decay_rate
    num_input_channel = 3
    num_classes = args.num_classes
    batch_size = args.batch_size
    clients_training_epoch = args.clients_training_epoch
    distorted_data = args.distorted_data
    dataset = args.dataset
    if distorted_data:
        num_input = 24
    else:
        num_input = 32
    if dataset == "mnist":
        num_input = 28
        distorted_data = False
        num_input_channel = 1
    parameters = parameters(clients_number,
                            clients_sample_ratio,
                            epoch,
                            learning_rate,
                            decay_rate,
                            num_input,
                            num_input_channel,
                            num_classes,
                            batch_size,
                            clients_training_epoch,
                            distorted_data,
                            dataset)

    Server = server(parameters)
    Server.run()




if __name__ == '__main__':

    # prohibit GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU_num)
    print(tf.test.is_built_with_cuda())
    print(tf.sysconfig.get_build_info())
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)]
                )
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    main()
