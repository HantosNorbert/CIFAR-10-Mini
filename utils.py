import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
import json
from models.VGG import create_VGG_8_simple, create_VGG_16_simple, create_VGG_8_simple_with_two_outputs
from models.VGG import create_VGG_8_with_dropout, create_VGG_8_with_weight_decay
from models.VGG import create_VGG_8_with_dropout_and_weight_decay
from models.ResNet import create_ResNet_18_simple
from configs import PathConfigs, SubsetSelectionConfigs, SimpleTrainingConfigs, AdvancedTrainingConfigs, CLASS_NAMES
from matplotlib import pyplot
import logging


def init_gpu(memory_limit: int):
    logging.info('Configure GPU...')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logging.info(f'{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs')
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            logging.error(e)


def save_history(history, file_name):
    with open(file_name, 'w') as file:
        json.dump(history, file)
        logging.info(f'Training history saved as {file_name}')


def create_model(model_name):
    if model_name == 'VGG_8_simple':
        return create_VGG_8_simple()
    if model_name == 'VGG_16_simple':
        return create_VGG_16_simple()
    if model_name == 'ResNet_18_simple':
        return create_ResNet_18_simple()
    # if model_name == 'ResNet_34_simple':
    #     return create_ResNet_34_simple()
    # if model_name == 'ResNet_50_simple':
    #     return create_ResNet_50_simple()
    # if model_name == 'ResNet_101_simple':
    #     return create_ResNet_101_simple()
    # if model_name == 'ResNet_152_simple':
    #     return create_ResNet_152_simple()

    if model_name == 'VGG_8_with_dropout':
        return create_VGG_8_with_dropout()
    if model_name == 'VGG_8_with_weight_decay':
        return create_VGG_8_with_weight_decay()
    if model_name == 'VGG_8_with_dropout_and_weight_decay':
        return create_VGG_8_with_dropout_and_weight_decay()

    if model_name == 'VGG_8_simple_with_two_outputs':
        return create_VGG_8_simple_with_two_outputs()

    else:
        assert False, f'Unknown model name {model_name}'


def parse_config_file(config_file_name: str):
    logging.info(f'Parsing config file {config_file_name}...')
    with open(config_file_name) as file_name:
        config = json.load(file_name)

        path_config = PathConfigs(config)
        subset_selection_config = SubsetSelectionConfigs(config)
        simple_training_config = SimpleTrainingConfigs(config)
        advanced_training_config = AdvancedTrainingConfigs(config)

        return path_config, subset_selection_config, simple_training_config, advanced_training_config


#################################################################
# Utils for Jupyter Notebook
#################################################################

def load_history(file_name):
    with open(file_name) as file_name:
        history = json.load(file_name)
        logging.info(f'Training history loaded from {file_name}')
        return history


def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history['loss'], color='blue', label='train')
    pyplot.plot(history['val_loss'], color='orange', label='test')
    pyplot.legend(('Train', 'Test'))
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history['accuracy'], color='blue', label='train')
    pyplot.plot(history['val_accuracy'], color='orange', label='test')
    pyplot.legend(('Train', 'Test'))
    pyplot.tight_layout()
    pyplot.show()


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i][0], img[i]
    pyplot.grid(False)
    pyplot.xticks([])
    pyplot.yticks([])

    pyplot.imshow(img, cmap=pyplot.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    pyplot.xlabel("{} {:2.0f}% ({})".format(CLASS_NAMES[predicted_label],
                                            100 * np.max(predictions_array),
                                            CLASS_NAMES[true_label]),
                  color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i][0]
    pyplot.grid(False)
    pyplot.xticks(range(10))
    pyplot.yticks([])
    thisplot = pyplot.bar(range(10), predictions_array, color="#777777")
    pyplot.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def load_and_make_predictions(model_name, weights_name):
    logging.info('Loading the model...')
    model = create_model(model_name)
    model.load_weights(weights_name).expect_partial()

    logging.info('Loading test cases...')
    _, (test_images, test_labels) = datasets.cifar10.load_data()

    logging.info('Make predictions...')
    predictions = model.predict(test_images)

    return test_images, test_labels, predictions


def show_example_result(test_images, test_labels, predictions):
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    pyplot.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        pyplot.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions[i], test_labels, test_images)
        pyplot.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions[i], test_labels)
    pyplot.tight_layout()
    pyplot.show()
