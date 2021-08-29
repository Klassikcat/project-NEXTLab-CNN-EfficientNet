import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1

class SEBlock(keras.Model):
    def __init__(self, input_size, r=4):
        super(SEBlock, self).__init__(trainable=True)
        self.excitation = keras.Sequential([
            layers.Dense(input_size * r),
            layers.Activation(activation='swish'),
            layers.Dense(input_size),
            layers.Activation(activation='sigmoid')
        ])
        self.squeeze = layers.AveragePooling2D(pool_size=input_size, strides=input_size, padding="same")

    def call(self, x):
        x = self.squeeze(x)
        x = self.excitation(x)
        return x

class MBConv(keras.Model):
    __expand = 6

    def __init__(self, input, output, strides,
                 kernel_size, se_scale=4, p=0.5):
        super(MBConv, self).__init__(trainable=True)
        self.p = tf.convert_to_tensor(p, dtype=float) if (input == output) else tf.convert_to_tensor(1, dtype=float)

        self.residual = keras.Sequential([
            layers.Conv2D(filters=input * MBConv.__expand, kernel_size=1,
                          strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(momentum=0.99, epsilon=1e-3),
            layers.Activation(activation='swish'),

            layers.Conv2D(filters=input * MBConv.__expand, kernel_size=kernel_size,
                          strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(momentum=0.99, epsilon=1e-3),
            layers.Activation(activation='swish'),
        ])
        self.se = SEBlock(input * self.__expand, se_scale)
        self.project = keras.Sequential([
            layers.Conv2D(output, kernel_size=1, strides=1, padding='valid', use_bias=False),
            layers.BatchNormalization(momentum=0.99, epsilon=1e-3)
        ])
        self.shortcut = (strides == 1 and (input == output))

    def call(self, x):
        if self.fit:
            if not tfp.distributions.Bernoulli(self.p):
                return x
        x_shortcut = x
        x_residual = self.residual(x)
        x_se = self.se.call(x_residual)

        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x = x_shortcut + x
        return x


class SepConv(keras.Model):
    __expand = 1
    def __init__(self, input, output, strides,
                 kernel_size, se_scale=4, p=0.5):
        super(SepConv, self).__init__(trainable=True)
        self.p = tf.convert_to_tensor(p, dtype=float) if (input == output) else tf.convert_to_tensor(1, dtype=float)

        self.residual = keras.Sequential([
            layers.Conv2D(filters=input * MBConv.__expand, kernel_size=kernel_size,
                          strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(momentum=0.99, epsilon=1e-3),
            layers.Activation(activation='swish')
            ])
        elf.se = SEBlock(input * self.__expand, se_scale)
        self.project = keras.Sequential([
            layers.Conv2D(output, kernel_size=1, strides=1, padding='valid', use_bias=False),
            layers.BatchNormalization(momentum=0.99, epsilon=1e-3)
        ])
        self.shortcut = (strides == 1 and (input == output))

    def call(self, x):
        if self.fit:
            if not tfp.distributions.Bernoulli(self.p):
                return x
        x_shortcut = x
        x_residual = self.residual(x)
        x_se = self.se.call(x_residual)

        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x = x_shortcut + x
        return x


if __name__ == '__main__':
    x = tf.random.normal([3, 24, 24, 16])
    print(f"original shape:{x.shape[3]}")
    model = MBConv(16, 16, strides=1, kernel_size=6, p=1)
    output = model(x)
    print(f"shape: {tf.shape(output)}")
    print(output[1, 0, 0, 0])

