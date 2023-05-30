import tensorflow as tf
from tensorflow.keras import Model, initializers



#### Create tf model for Client ####
class ClientModel(Model):
    def __init__(self, input_shape, num_classes, layer1=128, layer2=192, layer3=256, dropout_rate=0.2):
        super(ClientModel, self).__init__()

        # The output size of convolution layer: [(W-K+2P)/S]+1
        # The output size of maxpooling layer: (W-K)/S + 1
        # W is the input volume: image is 32*32*3, W=32
        # K is the kernel size
        # P is the padding
        # S is the stride

        parameter_initializer = tf.keras.initializers.LecunNormal()

        # 1st convolution layer

        self.conv1 = tf.keras.layers.Conv2D(filters=layer1, kernel_size=(3, 3),
                                            input_shape=input_shape[1:], padding='same',
                                            kernel_initializer=parameter_initializer)
        self.bn1 = tf.keras.Sequential(
            [
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.Dropout(dropout_rate)
            ]
        )
        
        self.pooling1 = tf.keras.layers.AveragePooling2D((2, 2), strides=1, padding="same")
        
        # 2nd convolution layer

        self.conv2 = tf.keras.layers.Conv2D(filters=layer2, kernel_size=(2, 2), strides=2,
                                            padding='valid', kernel_initializer=parameter_initializer)
        self.bn2 = tf.keras.Sequential(
            [
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.Dropout(dropout_rate)
            ]
        )
        self.pooling2 = tf.keras.layers.AveragePooling2D((2, 2), strides=2, padding="valid")

        # 3rd convolution layer

        self.conv3 = tf.keras.layers.Conv2D(filters=layer3, kernel_size=(3, 3), strides=2,
                                            padding='valid', kernel_initializer=parameter_initializer)
        self.bn3 = tf.keras.Sequential(
            [
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.Dropout(dropout_rate)
            ]
        )


        self.sixth_layer = tf.keras.layers.Flatten()

        self.seventh_layer = tf.keras.layers.Dense(num_classes, use_bias=False, 
                                                   kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                                   kernel_initializer=parameter_initializer)



    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.pooling1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pooling2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.sixth_layer(x)
        x = self.seventh_layer(x)
        return x


class ClientModel_2CNN(Model):
    def __init__(self, input_shape, num_classes, layer1=128, layer2=256, dropout_rate=0.2):
        super(ClientModel_2CNN, self).__init__()

        # The output size of convolution layer: [(W-K+2P)/S]+1
        # The output size of maxpooling layer: (W-K)/S + 1
        # W is the input volume: image is 32*32*3, W=32
        # K is the kernel size
        # P is the padding
        # S is the stride

        parameter_initializer = tf.keras.initializers.LecunNormal()

        # 1st convolution layer
        self.conv1 = tf.keras.layers.Conv2D(filters=layer1, kernel_size=(3, 3),
                                            input_shape=input_shape[1:], padding='same',
                                            kernel_initializer=parameter_initializer)
        self.bn1 = tf.keras.Sequential(
            [
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.Dropout(dropout_rate)
            ]
        )
        
        self.pooling1 = tf.keras.layers.AveragePooling2D((2, 2), strides=1, padding="same")
        
        # 2nd convolution layer

        self.conv2 = tf.keras.layers.Conv2D(filters=layer2, kernel_size=(3, 3), strides=2,
                                            padding='valid', kernel_initializer=parameter_initializer)
        self.bn2 = tf.keras.Sequential(
            [
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.Dropout(dropout_rate)
            ]
        )

        self.sixth_layer = tf.keras.layers.Flatten()

        self.seventh_layer = tf.keras.layers.Dense(num_classes, use_bias=False, 
                                                   kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                                   kernel_initializer=parameter_initializer)



    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.pooling1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.sixth_layer(x)
        x = self.seventh_layer(x)
        return x
