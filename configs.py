#################################################################
# CIFAR-10 specific data
#################################################################

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_CHANNELS = 3
NUM_CLASSES = 10
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


#################################################################
# Classes for holding different configurations
#################################################################

class PathConfigs:
    def __init__(self, configs):
        self.model_weights_folder_name = configs['MODEL_WEIGHTS_FOLDER_NAME']
        self.training_histories_folder_name = configs['TRAINING_HISTORIES_FOLDER_NAME']
        self.training_indices_file_name = configs['TRAINING_INDICES_FILE_NAME']


class SubsetSelectionConfigs:
    def __init__(self, configs):
        subset_selection = configs['SUBSET_SELECTION']
        self.training_subset_size = subset_selection['TRAINING_SUBSET_SIZE']
        self.training_subset_shuffle_attempt = subset_selection['TRAINING_SUBSET_SHUFFLE_ATTEMPT']
        self.fv_extractor_learning_rate = subset_selection['FEATURE_VECTOR_EXTRACTOR_LEARNING_RATE']
        self.fv_extractor_momentum = subset_selection['FEATURE_VECTOR_EXTRACTOR_MOMENTUM']
        self.fv_extractor_epochs = subset_selection['FEATURE_VECTOR_EXTRACTOR_EPOCHS']
        self.fv_extractor_batch_size = subset_selection['FEATURE_VECTOR_EXTRACTOR_BATCH_SIZE']


class SimpleTrainingConfigs:
    def __init__(self, configs):
        simple_training = configs['SIMPLE_TRAINING']
        self.learning_rate = simple_training['LEARNING_RATE']
        self.momentum = simple_training['MOMENTUM']
        self.epochs = simple_training['EPOCHS']
        self.batch_size = simple_training['BATCH_SIZE']


class AdvancedTrainingConfigs:
    def __init__(self, configs):
        advanced_training = configs['ADVANCED_TRAINING']
        self.learning_rate = advanced_training['LEARNING_RATE']
        self.momentum = advanced_training['MOMENTUM']
        self.epochs = advanced_training['EPOCHS']
        self.batch_size = advanced_training['BATCH_SIZE']
