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
        self.squeeze = layers.GlobalAveragePooling2D(keepdims=True, data_format='channels_last')

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
        print(f"residual: {tf.shape(x_residual)}\nse: {tf.shape(x_se)}")

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
            layers.Conv2D(filters=input * SepConv.__expand, kernel_size=kernel_size,
                          strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(momentum=0.99, epsilon=1e-3),
            layers.Activation(activation='swish')
            ])
        self.se = SEBlock(input * self.__expand, se_scale)
        self.project = keras.Sequential([
            layers.Conv2D(input * SepConv.__expand, kernel_size=1, strides=1, padding='valid', use_bias=False),
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
        print(f"residual: {tf.shape(x_residual)}\nse: {tf.shape(x_se)}")

        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x = x_shortcut + x
        return x

    class EfficientNet(keras.Model):
        def __init__(self):
            return None

class EfficienNet(keras.Model):
    def __init__(self, fine_tuning=True, weights=str, activation='swish', class_num=int):
        super(EfficienNet, self).__init__(trainable=True)
        if fine_tuning == True:
            include_top = False
        else:
            include_top = True
        self.model = EfficientNetB0(include_top=include_top, weights=weights,
                                    classes=class_num, classifier_activation=activation)

    def call(self, data):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        self.model.fit(data)
        self.model.predict(data)

if __name__ == '__main__':
    net = EfficienNet(fine_tuning=True, weights='imagenet',class_num=10)
    print(net.model.summary())
