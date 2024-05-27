import tensorflow as tf

slayers = "IIOII"
version = "01"
nlayers = 4

class CNN_pred(tf.keras.Model):
    def __init__(self):
        super(CNN_pred, self).__init__()
        self.model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(45,90,4)),
            tf.keras.layers.Conv2DTranspose(
                filters=8, kernel_size=3, strides=1, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=16, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=16, kernel_size=3, strides=1, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=8, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2D(
                filters=1, kernel_size=4, strides=1, padding='same'),
        ]
        )
    
    def call(self, x):
        return self.model(x)