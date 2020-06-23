from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Layer
from tensorflow.keras.regularizers import l2
import configs


def create_VGG_8_simple():
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
               input_shape=(configs.IMAGE_HEIGHT, configs.IMAGE_WIDTH, configs.IMAGE_CHANNELS)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', name='last_feature_layer'))
    model.add(Dense(configs.NUM_CLASSES, activation='softmax'))

    return model


def create_VGG_16_simple():
    model = Sequential()
    model.add(Conv2D(input_shape=(configs.IMAGE_HEIGHT, configs.IMAGE_WIDTH, configs.IMAGE_CHANNELS),
                     filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=4096, activation='relu'))
    model.add(Dense(units=4096, activation='relu'))
    model.add(Dense(configs.NUM_CLASSES, activation='softmax'))

    return model


def create_VGG_8_with_dropout():
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
               input_shape=(configs.IMAGE_HEIGHT, configs.IMAGE_WIDTH, configs.IMAGE_CHANNELS)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', name='last_feature_layer'))
    model.add(Dropout(0.2))
    model.add(Dense(configs.NUM_CLASSES, activation='softmax'))

    return model


def create_VGG_8_with_weight_decay():
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
               kernel_regularizer=l2(0.001),
               input_shape=(configs.IMAGE_HEIGHT, configs.IMAGE_WIDTH, configs.IMAGE_CHANNELS)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     kernel_regularizer=l2(0.001)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     kernel_regularizer=l2(0.001)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', name='last_feature_layer',
                    kernel_regularizer=l2(0.001)))
    model.add(Dense(configs.NUM_CLASSES, activation='softmax'))

    return model


def create_VGG_8_with_dropout_and_weight_decay():
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
               kernel_regularizer=l2(0.001),
               input_shape=(configs.IMAGE_HEIGHT, configs.IMAGE_WIDTH, configs.IMAGE_CHANNELS)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     kernel_regularizer=l2(0.001)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     kernel_regularizer=l2(0.001)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', name='last_feature_layer',
                    kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Dense(configs.NUM_CLASSES, activation='softmax'))

    return model


#################################################################
# Experimental model with two output layers
#################################################################

class ConstantLayer(Layer):
    def __init__(self):
        super(ConstantLayer, self).__init__()

    def call(self, inputs, **kwargs):
        return inputs * 0.0 + 1.0
        # return ones_like(input)


class VGG_8_withTwoOutputs(Model):
    def __init__(self, **kwargs):
        super(VGG_8_withTwoOutputs, self).__init__(**kwargs)
        self.conv11 = Conv2D(32,  (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')
        self.conv12 = Conv2D(32,  (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')
        self.conv21 = Conv2D(64,  (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')
        self.conv22 = Conv2D(64,  (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')
        self.conv31 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')
        self.conv32 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')
        self.maxpool1 = MaxPooling2D((2, 2))
        self.maxpool2 = MaxPooling2D((2, 2))
        self.maxpool3 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='relu', kernel_initializer='he_uniform', name='last_feature_layer')
        self.dense2 = Dense(configs.NUM_CLASSES, activation='softmax', name='output_layer')
        self.constant = ConstantLayer()

    def call(self, inputs, training=None, mask=None):
        x = self.conv11(inputs)
        x = self.conv12(x)
        x = self.maxpool1(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxpool2(x)
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x1 = self.dense2(x)
        x2 = self.constant(x1)

        return x1, x2


def create_VGG_8_simple_with_two_outputs():
    model = VGG_8_withTwoOutputs()
    return model
