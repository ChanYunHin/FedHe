import math
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.python.ops.gen_math_ops import mean

from tqdm import tqdm
import tensorflow as tf
import pandas as pd


from modules.Dataset import Dataset
from parameters import args
from modules.utils import get_model_mask

from modules.Client import Client

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Server:
    def __init__(self, parameters):

        #### SOME TRAINING PARAMS ####
        self.clients_number = parameters.clients_number
        self.clients_sample_ratio = parameters.clients_sample_ratio
        self.epoch = parameters.epoch
        self.learning_rate = parameters.learning_rate
        self.decay_rate = parameters.decay_rate
        self.num_input = parameters.num_input  # image shape: 32*32
        self.num_input_channel = parameters.num_input_channel  # image channel: 3
        self.num_classes = parameters.num_classes  # Cifar-10 total classes (0-9 digits)
        self.batch_size = parameters.batch_size
        self.clients_training_epoch = parameters.clients_training_epoch
        self.dataset = parameters.dataset
        self.distorted_data = parameters.distorted_data

        if self.dataset == "mnist":
            tf_dataset = tf.keras.datasets.mnist
        else:
            tf_dataset = tf.keras.datasets.cifar10

        self.dataset_server = Dataset(tf_dataset.load_data,
                                      split=self.clients_number,
                                      distorted_data=self.distorted_data,
                                      dataset_name = self.dataset)

        self.clients_dict, self.server_model = self.build_clients_and_server(self.clients_number,
                                                                             self.distorted_data,
                                                                             self.dataset_server)

    def build_clients_and_server(self, num, distorted_data, dataset_server):
        learning_rate = self.learning_rate
        num_input = self.num_input
        num_input_channel = self.num_input_channel
        num_classes = self.num_classes
        batch_size = self.batch_size
        clients_training_epoch = self.clients_training_epoch
        dataset = self.dataset

        clients_dict = {}
        input_data_shape = [batch_size, num_input, num_input, num_input_channel]

        client_models = {}
        if args.homo_flag:
            client_models[0] = {"model_type": "3_layer_CNN", "params": {"n1": 128, "n2": 128, "n3": 192, "dropout_rate": 0.3}}
            client_models[1] = {"model_type": "3_layer_CNN", "params": {"n1": 128, "n2": 128, "n3": 192, "dropout_rate": 0.3}}
            client_models[2] = {"model_type": "3_layer_CNN", "params": {"n1": 128, "n2": 128, "n3": 192, "dropout_rate": 0.3}}
            client_models[3] = {"model_type": "3_layer_CNN", "params": {"n1": 128, "n2": 128, "n3": 192, "dropout_rate": 0.3}}
            client_models[4] = {"model_type": "3_layer_CNN", "params": {"n1": 128, "n2": 128, "n3": 192, "dropout_rate": 0.3}}
            client_models[5] = {"model_type": "3_layer_CNN", "params": {"n1": 128, "n2": 128, "n3": 192, "dropout_rate": 0.3}}
            client_models[6] = {"model_type": "3_layer_CNN", "params": {"n1": 128, "n2": 128, "n3": 192, "dropout_rate": 0.3}}
            client_models[7] = {"model_type": "3_layer_CNN", "params": {"n1": 128, "n2": 128, "n3": 192, "dropout_rate": 0.3}}
            client_models[8] = {"model_type": "3_layer_CNN", "params": {"n1": 128, "n2": 128, "n3": 192, "dropout_rate": 0.3}}
            client_models[9] = {"model_type": "3_layer_CNN", "params": {"n1": 128, "n2": 128, "n3": 192, "dropout_rate": 0.3}}
        else:
            client_models[0] = {"model_type": "2_layer_CNN", "params": {"n1": 128, "n2": 256, "dropout_rate": 0.2}}
            client_models[1] = {"model_type": "2_layer_CNN", "params": {"n1": 128, "n2": 384, "dropout_rate": 0.2}}
            client_models[2] = {"model_type": "2_layer_CNN", "params": {"n1": 128, 'n2': 512, "dropout_rate": 0.2}}
            client_models[3] = {"model_type": "2_layer_CNN", "params": {"n1": 256, "n2": 256, "dropout_rate": 0.3}}
            client_models[4] = {"model_type": "2_layer_CNN", "params": {"n1": 256, "n2": 512, "dropout_rate": 0.4}}
            client_models[5] = {"model_type": "3_layer_CNN", "params": {"n1": 64, "n2": 128, "n3": 256, "dropout_rate": 0.2}}
            client_models[6] = {"model_type": "3_layer_CNN", "params": {"n1": 64, "n2": 128, "n3": 192, "dropout_rate": 0.2}}
            client_models[7] = {"model_type": "3_layer_CNN", "params": {"n1": 128, "n2": 192, "n3": 256, "dropout_rate": 0.2}}
            client_models[8] = {"model_type": "3_layer_CNN", "params": {"n1": 128, "n2": 128, "n3": 128, "dropout_rate": 0.3}}
            client_models[9] = {"model_type": "3_layer_CNN", "params": {"n1": 128, "n2": 128, "n3": 192, "dropout_rate": 0.3}}
        

        # create Clients and models
        for cid in range(num):
            clients_dict[cid] = Client(input_shape=input_data_shape,
                                    num_classes=num_classes,
                                    learning_rate=learning_rate,
                                    client_id=cid,
                                    distorted_data=distorted_data,
                                    batch_size=batch_size,
                                    clients_training_epoch=clients_training_epoch,
                                    dataset=dataset_server.train[cid],
                                    dataset_name=dataset,
                                    is_worker=True,
                                    neural_network_shape=client_models[cid])

        server_model = Client(input_shape=input_data_shape,
                                num_classes=num_classes,
                                learning_rate=learning_rate,
                                client_id=-1,
                                distorted_data=distorted_data,
                                batch_size=batch_size,
                                clients_training_epoch=clients_training_epoch,
                                is_worker=False)


        return clients_dict, server_model

    def choose_clients(self):
        """ randomly choose some clients """
        client_num = self.clients_number
        ratio = self.clients_sample_ratio

        self.choose_num = math.floor(client_num * ratio)
        return np.random.permutation(client_num)[:self.choose_num]


    def list_divided_int(self, a, b):
        assert len(a) > 1
        for k in range(len(a)):
            a[k] /= b
        return a

    def evaluate(self, dataset, batch_size):
        epoch = math.ceil(dataset.size // batch_size)
        accuracy = tf.keras.metrics.CategoricalAccuracy()
        mean_loss = tf.keras.metrics.CategoricalCrossentropy(from_logits=True)

        tmp_loss_list = []
        tmp_acc_list = []
        for i in range(epoch):
            x, y = dataset.next_test_batch(batch_size)
            if self.distorted_data:
                x = tf.image.per_image_standardization(x)

            predictions = self.server_model.model(x)
            accuracy.update_state(y, predictions)
            mean_loss.update_state(y, predictions)

            tmp_loss_list.append(mean_loss.result())
            tmp_acc_list.append(accuracy.result())

        mean_loss = np.mean(np.array(tmp_loss_list))
        acc = np.mean(np.array(tmp_acc_list))

        return mean_loss, acc

    def completed_client_this_iter(self, outdated_flag, completed_number):
        nonzero_index = np.nonzero(outdated_flag)
        if len(nonzero_index[0]) > 0:
            completed_index = np.random.choice(nonzero_index[0], completed_number, replace=False)
            return completed_index
        else:
            return []

    def collect_logits(self, res,
                       client_y_logits):
        
        if "server_y_logits" not in res:
            res["server_y_logits"] = {}
        server_y_logits = res["server_y_logits"]
        if "server_labels_counts_y_logits" not in res:
            res["server_labels_counts_y_logits"] = {}
        server_labels_counts_y_logits = res["server_labels_counts_y_logits"]

        for key, val in client_y_logits.items():
            if key in server_y_logits:
                server_labels_counts_y_logits[key] += 1
                server_y_logits[key] += val
            else:
                server_labels_counts_y_logits[key] = 1
                server_y_logits[key] = val

        res["server_y_logits"] = server_y_logits
        res["server_labels_counts_y_logits"] = server_labels_counts_y_logits

        return res

    def average_logits(self, res, clients_y_logits=None):     
        server_y_logits = res["server_y_logits"]
        server_labels_counts = res["server_labels_counts_y_logits"]
        tmp_y_logits = server_y_logits.copy()
        for key, val in server_y_logits.items():
            tmp_y_logits[key] = val / server_labels_counts[key]
        return tmp_y_logits

    def initialize_counting(self, res):
        server_labels_counts = res["server_labels_counts_y_logits"]
        tmp_labels_counts = server_labels_counts.copy()
        for key, val in server_labels_counts.items():
            tmp_labels_counts[key] = 1
        res["server_labels_counts_y_logits"] = tmp_labels_counts
        return res

    def asynchronous_training(self):
        epoch = self.epoch
        decay_rate = self.decay_rate
        ratio = self.clients_sample_ratio
        batch_size = self.batch_size
        clients_number = self.clients_number

        # record the delayed time steps
        outdated_flag = np.zeros(self.clients_number)

        evaluate_client_acc = {}
        mean_loss_list = []
        acc_list = []
        train_loss_list = []
        train_acc_list = []
        train_distill_loss_list = []
        train_loss = 0
        train_acc = 0
        train_distill_loss = 0
        cnt = 0
        cnt_distill = 0
        tmp_distill_loss = 0
        avg_server_y_logits = 0
        clients_y_logits = {}
        res = {}
        distillation_flag = False
        max_training_epoches = int(epoch * clients_number * ratio)

        for ep in tqdm(range(max_training_epoches)):
            # randomly choose some clients each epoch
            selected_clients = self.choose_clients()
            if ep >= clients_number:
                distillation_flag = True

            if ep % (clients_number * 10) == 0 and ep > 10:
                server_y_logits = self.average_logits(res)
                res["server_y_logits"] = server_y_logits
                res = self.initialize_counting(res)

            # update clients' states
            for idx, cid in enumerate(selected_clients):
                train_model = self.clients_dict[cid]

                if outdated_flag[cid] > 0:
                    outdated_flag[cid] += 1
                    continue
                else:
                    if cid in selected_clients:
                        outdated_flag[cid] = 1

            # Clients which finish their updating processes.
            completed_clients = self.completed_client_this_iter(outdated_flag, 1)

            # apply gradients to a server model
            for idx, cid in enumerate(completed_clients):
                assert outdated_flag[cid] > 0
                train_model = self.clients_dict[cid]
                if distillation_flag:
                    if cid not in clients_y_logits:
                        avg_server_y_logits = self.average_logits(res)
                    else:
                        avg_server_y_logits = self.average_logits(res, clients_y_logits[cid])
                training_result = train_model.training(max_training_epoches, 
                                                       ep, 
                                                       avg_server_y_logits,
                                                       distillation_flag)

                # unpack the training result
                if distillation_flag:
                    tmp_label_loss = training_result[0]
                    tmp_distill_loss = training_result[1]
                    tmp_train_acc = training_result[2]
                    tmp_y_logits = training_result[3]
                else:
                    tmp_label_loss = training_result[0]
                    tmp_train_acc = training_result[1]
                    tmp_y_logits = training_result[2]
                clients_y_logits[cid] = tmp_y_logits
                
                outdated_flag[cid] = 0

                res = self.collect_logits(res, tmp_y_logits)

                train_loss += tmp_label_loss
                train_acc += tmp_train_acc
                cnt += 1
                if distillation_flag:
                    cnt_distill += 1
                    train_distill_loss += tmp_distill_loss

            if ep % 100 == 0:
                train_acc /= cnt
                train_loss /= cnt

                evaluate_mean_loss = []
                evaluate_mean_acc = []
                evaluate_num = 0
                
                if self.clients_number <= 20:
                    evaluate_num = self.clients_number
                else:
                    evaluate_num = int(self.clients_number * 0.1)
                for idx in range(evaluate_num):
                    if self.clients_number >= 20:
                        selected_idx = np.random.choice(self.clients_number)
                    else:
                        selected_idx = idx
                    self.server_model = self.clients_dict[selected_idx]

                    mean_loss, acc = self.evaluate(self.dataset_server.test,
                                                batch_size * batch_size)
                    if selected_idx not in evaluate_client_acc:
                        evaluate_client_acc[selected_idx] = []

                    evaluate_client_acc[selected_idx].append(acc)
                    evaluate_mean_loss.append(mean_loss)
                    evaluate_mean_acc.append(acc)

                mean_loss = np.mean(np.array(evaluate_mean_loss))
                acc = np.mean(np.array(evaluate_mean_acc))

                mean_loss_list.append(mean_loss)
                acc_list.append(acc)

                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)

                if distillation_flag:
                    train_distill_loss /= cnt_distill
                    train_distill_loss_list.append(train_distill_loss)
                    print("\nDistillation: loss={}".format(train_distill_loss))
                    train_distill_loss = 0
                    cnt_distill = 0

                print("training: loss={} | accuracy={}".format(train_loss,
                                                               train_acc))
                print("evaluating: loss={} | accuracy={}".format(mean_loss, acc))
                train_loss = 0
                train_acc = 0
                cnt = 0


        return train_loss_list, train_acc_list, train_distill_loss_list, mean_loss_list, acc_list, evaluate_client_acc

    def run(self):

        train_loss_list, train_acc_list, train_distill_loss_list, eval_loss_list, eval_acc_list, evaluate_client_acc = self.asynchronous_training()

        df_client_acc = pd.DataFrame.from_dict(evaluate_client_acc)
        

        model_mask = get_model_mask(args)

        txt_save_path = "txt_result/{}/{}/alpha{}_lr{}/".format(self.dataset, model_mask, args.alpha, self.learning_rate)
        makedirs(txt_save_path)
        np.savetxt('{}train_loss.txt'.format(txt_save_path), np.array(train_loss_list))
        np.savetxt('{}train_acc.txt'.format(txt_save_path), np.array(train_acc_list))
        np.savetxt('{}distill_loss.txt'.format(txt_save_path), np.array(train_distill_loss_list))
        np.savetxt('{}evaluate_loss.txt'.format(txt_save_path), np.array(eval_loss_list))
        np.savetxt('{}evaluate_acc.txt'.format(txt_save_path), np.array(eval_acc_list))
        df_client_acc.to_pickle('{}evaluate_client_acc.pkl'.format(txt_save_path))

        plt.plot(train_loss_list, label="train_loss")
        plt.plot(eval_loss_list, label="evaluate_loss")
        plt.title("Loss")
        plt.ylabel("loss")
        plt.xlabel("Number of iterations")
        plt.legend()
        pic_save_path = "pic/{}/{}/".format(self.dataset, model_mask)
        makedirs(pic_save_path)
        plt.savefig("{}asydis_loss_alpha{}_lr{}.png".format(pic_save_path, args.alpha, self.learning_rate))
        plt.close()

        plt.plot(train_acc_list, label="train_acc")
        plt.plot(eval_acc_list, label="evaluate_acc")
        plt.title("Accuracy")
        plt.ylabel("Acc")
        plt.xlabel("Number of iterations")
        plt.legend()
        plt.savefig("{}asydis_Acc_alpha{}_lr{}.png".format(pic_save_path, args.alpha, self.learning_rate))
        plt.close()

        plt.plot(train_distill_loss_list, label="distill_loss")
        plt.title("Distillation_loss")
        plt.ylabel("loss")
        plt.xlabel("Number of iterations")
        plt.legend()
        plt.savefig("{}asydis_distill_loss_alpha{}_lr{}.png".format(pic_save_path, args.alpha, self.learning_rate))
        plt.close()
