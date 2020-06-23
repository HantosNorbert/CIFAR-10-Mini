from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import configs


def create_feature_vector_extractor_model():
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
    model.add(Dropout(0.3))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform', name='last_feature_layer'))
    model.add(Dropout(0.5))
    model.add(Dense(configs.NUM_CLASSES, activation='softmax'))

    return model
