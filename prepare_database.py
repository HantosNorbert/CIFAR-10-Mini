import numpy as np
from tensorflow.keras import datasets, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from sklearn.metrics.pairwise import cosine_similarity
from models.feature_vector_extractor import create_feature_vector_extractor_model
from configs import PathConfigs, SubsetSelectionConfigs, NUM_CLASSES
import utils
import random
import json
import logging


# Class representing an index of a training image and its feature vector
class IndexedFeatureVector:
    def __init__(self, idx, fv):
        self.idx = idx
        self.fv = fv


# Load entire CIFAR-10 database, normalize the images, and transform the labels into one-shot vectors
def load_full_database():
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    train_images, test_images = train_images.astype('float32') / 255.0, test_images.astype('float32') / 255.0
    return (train_images, train_labels), (test_images, test_labels)


# Pick random indices from the distance table multiple times, and return with the bests (i.e., the ones that have
# the greatest overall distance)
def random_index_search(distance_table, subset_selection_config: SubsetSelectionConfigs):
    subset_size_per_class = subset_selection_config.training_subset_size // NUM_CLASSES

    max_distance = -np.inf
    best_indices = []
    index_candidates = range(distance_table.shape[0])

    for i in range(subset_selection_config.training_subset_shuffle_attempt):

        current_indices = random.sample(index_candidates, subset_size_per_class)
        sub_distance_table = distance_table[np.ix_(current_indices, current_indices)]
        sum_distance = np.sum(sub_distance_table)

        if sum_distance > max_distance:
            logging.debug(f'Best at index {i}: {sum_distance}')
            max_distance = sum_distance
            best_indices = current_indices

    logging.debug(f'Best overall distance: {max_distance}')
    return best_indices


# Collect the feature vectors of a given class id, build the distance matrix, and search indices such that their
# overall distance is as big as possible
def select_indices_for_class(train_feature_vectors, train_labels, class_id,
                             subset_selection_config: SubsetSelectionConfigs):
    # A feature vector's global index is its ordeal number among all the training data.
    # A feature vector's local index is its ordeal number among all the training data belonging to the same class.
    # Local indices are strictly consecutive numbers starting from 0.
    local_to_global_idx = {}
    feature_vectors = []

    counter = 0
    for i in range(len(train_labels)):
        label = np.argmax(train_labels[i])
        if label == class_id:
            local_to_global_idx[counter] = i
            feature_vectors.append(train_feature_vectors[i])
            counter += 1

    cosine_distance_table = 1 - cosine_similarity(feature_vectors)
    selected_local_indices = random_index_search(cosine_distance_table, subset_selection_config)
    selected_global_indices = [local_to_global_idx[i] for i in selected_local_indices]

    return selected_global_indices


def load_training_indices_from_file(file_name):
    with open(file_name) as json_file:
        data = json.load(json_file)
        selected_indices = data['training_indices']
        logging.info(f'Selected indices loaded from {file_name}')
        return selected_indices


def save_training_indices_to_file(selected_indices, file_name):
    selected_indices_data = {'training_indices': selected_indices}
    with open(file_name, 'w') as outfile:
        json.dump(selected_indices_data, outfile)
        logging.info(f'Selected indices saved as {file_name}')


# For each class id, select the best training image indices that represent a wide spectrum of the training samples
def select_indices(feature_vectors, train_labels, subset_selection_config: SubsetSelectionConfigs):
    selected_indices = []

    for class_id in range(NUM_CLASSES):
        logging.info(f'Select indices for class {class_id}...')
        selected_indices_for_class = select_indices_for_class(feature_vectors, train_labels, class_id,
                                                              subset_selection_config)

        # Self-checking that we did select correct indices
        for i in selected_indices_for_class:
            train_label = np.argmax(train_labels[i])
            assert train_label == class_id, f'Something went wrong: {train_label} != {class_id}'

        selected_indices += selected_indices_for_class

    return selected_indices


# Return with a subset of the original CIFAR-10 dataset training images and labels; as well as the full testing data
def load_subset_database(args, path_config: PathConfigs, subset_selection_config: SubsetSelectionConfigs):
    logging.info('Loading subset database...')

    logging.info('Loading full CIFAR-10 database...')
    (train_images, train_labels), (test_images, test_labels) = load_full_database()

    if args.subset_selection_method == 'load_indices':
        selected_indices = load_training_indices_from_file(path_config.training_indices_file_name)
        return (train_images[selected_indices, :, :, :], train_labels[selected_indices, :]), (test_images, test_labels)

    # create and compile model we later use for feature vector extracting
    model = create_feature_vector_extractor_model()
    opt = SGD(lr=subset_selection_config.fv_extractor_learning_rate,
              momentum=subset_selection_config.fv_extractor_momentum)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    if args.subset_selection_method == 'load_fv_extractor':
        logging.info('Loading feature vector extractor model...')
        model.load_weights(path_config.model_weights_folder_name + '/fv_extractor_model').expect_partial()
    else:
        logging.info('Training feature vector extractor model...')
        history = model.fit(train_images, train_labels, epochs=subset_selection_config.fv_extractor_epochs,
                            batch_size=subset_selection_config.fv_extractor_batch_size,
                            validation_data=(test_images, test_labels), verbose=2)
        model.save_weights(path_config.model_weights_folder_name + '/fv_extractor_model')
        logging.info(f"Model weights saved as {path_config.model_weights_folder_name + '/fv_extractor_model'}")
        utils.save_history(history.history, path_config.training_histories_folder_name + '/fv_extractor_history.json')

    _, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
    logging.info(f'Accuracy of the feature vector generator model: {test_accuracy}')

    # We use the trained model to generate rich feature vectors
    fv_extractor_model = Model(model.input, model.get_layer(name='last_feature_layer').output)
    feature_vectors = fv_extractor_model.predict(train_images)

    # Based on the feature vectors, select the best indices for each class id
    selected_indices = select_indices(feature_vectors, train_labels, subset_selection_config)
    save_training_indices_to_file(selected_indices, path_config.training_indices_file_name)
    return (train_images[selected_indices, :, :, :], train_labels[selected_indices, :]), (test_images, test_labels)
