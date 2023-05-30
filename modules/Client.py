import os
import math
import numpy as np
from collections import namedtuple

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib

from modules.Model import ClientModel
from modules.Model import ClientModel_2CNN
from parameters import args 


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

class Client:
    def __init__(self,
                 input_shape,
                 num_classes,
                 learning_rate,
                 client_id,
                 distorted_data,
                 batch_size,
                 clients_training_epoch,
                 dataset=None,
                 dataset_name=None,
                 is_worker=True,
                 neural_network_shape=None):

        self.input_shape = input_shape

        if neural_network_shape is None:
            self.model = ClientModel(input_shape, num_classes)
        else:
            if neural_network_shape["model_type"] == "2_layer_CNN":
                self.model = ClientModel_2CNN(input_shape, 
                                              num_classes, 
                                              layer1 = neural_network_shape["params"]["n1"],
                                              layer2 = neural_network_shape["params"]["n2"],
                                              dropout_rate = neural_network_shape["params"]["dropout_rate"])
            else:
                self.model = ClientModel(input_shape, 
                                         num_classes, 
                                         layer1 = neural_network_shape["params"]["n1"],
                                         layer2 = neural_network_shape["params"]["n2"],
                                         layer3 = neural_network_shape["params"]["n3"],
                                         dropout_rate = neural_network_shape["params"]["dropout_rate"])
        # If you don't have this function, you would not have true weights in the network,
        # because you only get your weights when data pass through the network.
        self.create_weights_for_model(self.model)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                                name='optimizer_{}'.format(client_id))
        
        self.loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                                name='loss_func_{}'.format(client_id))

        self.distill_loss_func = tf.keras.losses.MeanSquaredError()
        self.distill_loss = tf.keras.metrics.MeanSquaredError()
        self.accuracy = tf.keras.metrics.CategoricalAccuracy(name='accuracy_{}'.format(client_id))
        self.mean_loss = tf.keras.metrics.CategoricalCrossentropy(from_logits=True,
                                                                  name="mean_loss_{}".format(client_id))

        # initialize
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.cid = client_id
        self.dataset = dataset
        self.is_worker = is_worker
        self.distorted_data = distorted_data
        self.batch_size = batch_size
        self.clients_training_epoch = clients_training_epoch

        
    def create_weights_for_model(self, this_model):
        a = [1]
        a.extend(self.input_shape[1:])
        create_weights_data = tf.zeros(a)
        this_model(create_weights_data)

    def create_model_with_weights(self, this_weights):
        this_model = ClientModel(self.input_shape, self.num_classes)
        self.create_weights_for_model(this_model)
        this_model.set_weights(this_weights)
        return this_model
    
    
    
    def train_step(self, x, y, logit_vectors=None, distillation=False):
        assert self.is_worker
        alpha = args.alpha
        predictions = 0
        if distillation:
            assert logit_vectors
        """ one training iteration """
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            pred_loss = self.loss_func(y, predictions)
            if distillation:
                distill_loss = self.distill_loss_func(logit_vectors, predictions)
                pred_loss = pred_loss + alpha * distill_loss

        gradients = tape.gradient(pred_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.accuracy.update_state(y, tf.nn.softmax(predictions))
        self.mean_loss.update_state(y, predictions)
        if distillation:
            self.distill_loss.update_state(logit_vectors, predictions)
        return predictions

    def collect_logits(self, client_logit_vectors, one_hot_y, logit_vectors, count_for_labels):
        # one_hots to class labels
        labels = [np.argmax(yi) for yi in one_hot_y]
        for idx, logit_y in enumerate(client_logit_vectors):
            if labels[idx] not in logit_vectors:
                logit_vectors[labels[idx]] = logit_y
                count_for_labels[labels[idx]] = 1
            else:
                logit_vectors[labels[idx]] += logit_y
                count_for_labels[labels[idx]] += 1
        return logit_vectors, count_for_labels

    def match_labels_to_logits(self, server_y_logits, one_hot_y):
        labels = [np.argmax(yi) for yi in one_hot_y]
        server_logit_vectors = []
        # match logit vectors for each label
        for idx, label in enumerate(labels):
            server_logit_vectors.append(server_y_logits[label])
        return server_logit_vectors

    def data_augmentation(self, x):
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, 63)
        x = tf.image.random_contrast(x, 0.2, 1.8)
        x = tf.image.per_image_standardization(x)
        return x

    def sort_logits(self, 
                    client_y_logits, 
                    count_for_labels):
        for label, logit in client_y_logits.items():
            client_y_logits[label] = logit / count_for_labels[label]
        return client_y_logits


    def tmp_save_pic(self, save_path):
        for i in range(self.distilled_x.shape[0]):
            tf.keras.preprocessing.image.save_img("{}/{}.png".format(save_path, i), self.distilled_x[i])

    def training(self, 
                 max_epoches, 
                 current_epoches, 
                 server_y_logits=None,
                 distillation=False):
        """ Training n times (n=the epoch parameter)"""


        # split x and y at the beginning of each training iteration.
        accumulate_gradients = 0
        server_logit_vectors = None
        tmp_loss_list = []
        tmp_acc_list = []
        tmp_distill_loss_list = []
        client_y_logits = {}
        count_for_labels = {}
        self.mean_loss.reset_states()
        self.accuracy.reset_states()
        self.distill_loss.reset_states()

        # training
        for i in range(self.clients_training_epoch):
            x, y = self.dataset.next_batch(self.batch_size)
            if distillation:
                server_logit_vectors = self.match_labels_to_logits(server_y_logits, y)
            # data augmentation process
            if self.distorted_data:
                x = self.data_augmentation(x)

            client_logit_vectors = self.train_step(x, y, server_logit_vectors, distillation)



            tmp_loss_list.append(self.mean_loss.result())
            tmp_acc_list.append(self.accuracy.result())
            if distillation:
                tmp_distill_loss_list.append(self.distill_loss.result())

            client_y_logits, count_for_labels = self.collect_logits(client_logit_vectors,
                                                                    y,
                                                                    client_y_logits,
                                                                    count_for_labels)


        mean_loss = np.mean(np.array(tmp_loss_list))
        acc = np.mean(np.array(tmp_acc_list))
        
        client_y_logits = self.sort_logits(client_y_logits, count_for_labels)
        if distillation:
            distill_loss = np.mean(np.array(tmp_distill_loss_list))
            return [mean_loss,
                    distill_loss,
                    acc,
                    client_y_logits]
        else:
            return [mean_loss,
                    acc,
                    client_y_logits]

    def server_apply_gradient(self, gradients):
        self.optimizer.apply_gradients(zip(gradients,
                                           self.model.trainable_variables))

    def evaluating_step(self, x, y):
        predictions = self.model(x, training=True)
        self.accuracy.update_state(y, predictions)
        self.mean_loss.update_state(y, predictions)

    def evaluation(self, dataset, batch_size):
        epoch = math.ceil(dataset.size // batch_size)
        self.accuracy.reset_states()
        self.mean_loss.reset_states()

        tmp_loss_list = []
        tmp_acc_list = []
        for i in range(epoch):
            x, y = dataset.next_test_batch(batch_size)
            if self.distorted_data:
                x = tf.image.per_image_standardization(x)
            self.evaluating_step(x, y)
            tmp_loss_list.append(self.mean_loss.result())
            tmp_acc_list.append(self.accuracy.result())

        mean_loss = np.mean(np.array(tmp_loss_list))
        acc = np.mean(np.array(tmp_acc_list))

        return mean_loss, acc

    def get_trainable_weights(self):
        return self.model.trainable_variables

    def get_client_id(self):
        """ Return the client id """
        return self.cid


    def get_weights(self):
        """ Return the model's parameters """
        return self.model.get_weights()

    def set_weights(self, weights):
        """ Assign server model's parameters to this client """
        self.model.set_weights(weights)

