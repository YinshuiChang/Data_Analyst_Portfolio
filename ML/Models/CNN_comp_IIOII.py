import tensorflow as tf
import numpy as np

slayers = "IIOII"
version = "01"
elayers = 1
players = 4

class CNN_comp(tf.keras.Model):
    def __init__(self, mpath = "./"):
        super(CNN_comp, self).__init__(mpath)
        self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(182, 362, 1)),
            tf.keras.layers.Conv2D(
                filters=16, kernel_size=3, strides=2, activation='relu', padding='valid'),
            tf.keras.layers.Conv2D(
                filters=16, kernel_size=3, strides=2, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(
                filters=1, kernel_size=3, strides=1, activation='relu', padding='same'),
        ]
        )

        self.pred = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(45,90,4)),
            tf.keras.layers.Conv2DTranspose(
                filters=8, kernel_size=3, strides=1, activation='relu', padding='same'),
            tf.keras.layers.Conv2DTranspose(
                filters=16, kernel_size=3, strides=2, activation='relu', padding='same'),
            tf.keras.layers.Conv2DTranspose(
                filters=16, kernel_size=3, strides=1, activation='relu', padding='same'),
            tf.keras.layers.Conv2DTranspose(
                filters=8, kernel_size=3, strides=2, activation='relu', padding='same'),
            # No activation
            tf.keras.layers.Conv2D(
                filters=1, kernel_size=4, strides=1, padding='same'),
        ]
        )
        
        self.encoder.load_weights(mpath + 'Weights/cnn_ae_vx01_encoder_weights_34')
        self.pred.load_weights(mpath + 'Weights/cnn_pred_IIOII_v01_weights_57')
        
    def call(self, x):
        temp_0 = x[:,0,:,:]
        temp_0 = tf.keras.layers.concatenate([temp_0[180:,:], temp_0[:180,:]], axis = 0)
        temp_0 = tf.keras.layers.Reshape((1,360,4))(temp_0)
        temp_180 = x[:,179,:,:]
        temp_180 = tf.keras.layers.concatenate([temp_180[180:,:], temp_180[:180,:]] , axis=0)
        temp_180 = tf.keras.layers.Reshape((1,360,4))(temp_180)
        x = tf.keras.layers.concatenate([temp_180, x, temp_0], axis=1)
        x_0 = tf.keras.layers.Reshape((182,1,4))(x[:,:,0])
        x_360 = tf.keras.layers.Reshape((182,1,4))(x[:,:,359])
        x = tf.keras.layers.concatenate([x_360, x, x_0], axis=2)
        encoded_0 = self.encoder(x[:,:,:,0])
        encoded_1 = self.encoder(x[:,:,:,1])
        encoded_2 = self.encoder(x[:,:,:,2])
        encoded_3 = self.encoder(x[:,:,:,3])
        encoded = tf.keras.layers.concatenate([encoded_0, encoded_1, encoded_2, encoded_3], axis = 3)
        decoded = self.pred(encoded)
        return decoded
    

