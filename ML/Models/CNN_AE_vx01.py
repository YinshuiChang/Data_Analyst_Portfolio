import tensorflow as tf

version = "x01"
nlayers = 1

class CNN_AE(tf.keras.Model):
    def __init__(self):
        super(CNN_AE, self).__init__()
        self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(182, 362, 1)),
            tf.keras.layers.Conv2D(
                filters=16, kernel_size=3, strides=(2, 2), activation='relu', padding='valid'),
            tf.keras.layers.Conv2D(
                filters=16, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
            tf.keras.layers.Conv2D(
                filters=1, kernel_size=3, strides=(1, 1), activation='relu', padding='same'),
        ]
        )

        self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(45,90,1)),
            tf.keras.layers.Conv2DTranspose(
                filters=16, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=16, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2D(
                filters=1, kernel_size=3, strides=1, padding='same'),
        ]
        )
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded