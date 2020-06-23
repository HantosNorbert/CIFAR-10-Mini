from configs import PathConfigs, SimpleTrainingConfigs, AdvancedTrainingConfigs
from configs import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS
import utils
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import logging


def train_basic_model(args, path_configs: PathConfigs, train_configs: SimpleTrainingConfigs,
                      train_images, train_labels, test_images, test_labels):
    logging.info('Training basic network...')

    model_name = args.model
    model = utils.create_model(model_name)

    model.build(input_shape=(train_configs.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    model.summary()

    opt = SGD(lr=train_configs.learning_rate, momentum=train_configs.momentum)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    if args.use_augmentation:
        model_name += '_augment'
        data_gen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1,
                                      horizontal_flip=True)
        it_train = data_gen.flow(train_images, train_labels, batch_size=train_configs.batch_size)
        steps = int(train_images.shape[0] / train_configs.batch_size)
        history = model.fit_generator(it_train, steps_per_epoch=steps, epochs=train_configs.epochs,
                                      validation_data=(test_images, test_labels), verbose=2)

    else:
        history = model.fit(train_images, train_labels, epochs=train_configs.epochs,
                            batch_size=train_configs.batch_size,
                            validation_data=(test_images, test_labels), verbose=2)

    model.save_weights(path_configs.model_weights_folder_name + '/' + model_name)
    utils.save_history(history.history,
                       path_configs.training_histories_folder_name + '/' + model_name + '_history.json')


def train_advanced_model(args, path_configs: PathConfigs, train_configs: AdvancedTrainingConfigs,
                         train_images, train_labels, test_images, test_labels):
    logging.info(f'Training advanced network...')

    model_name = 'VGG_8_advanced_lr' + str(train_configs.learning_rate)

    if args.training_method == 'SGD':
        opt = SGD(lr=train_configs.learning_rate, momentum=train_configs.momentum)
        model_name += '_sgd'
    elif args.training_method == 'Adam':
        opt = Adam(lr=train_configs.learning_rate)
        model_name += '_adam'
    else:
        assert False, f'Unknown optimizer method: {args.training_method}'

    model = utils.create_model('VGG_8_simple')
    if args.use_dropout and not args.use_weight_decay:
        model = utils.create_model('VGG_8_with_dropout')
        model_name += '_do'
    elif args.use_weight_decay and not args.use_dropout:
        model = utils.create_model('VGG_8_with_weight_decay')
        model_name += '_wd'
    elif args.use_dropout and args.use_weight_decay:
        model = utils.create_model('VGG_8_with_dropout_and_weight_decay')
        model_name += '_do_wd'

    if train_configs.batch_size != 64:
        model_name += f'_batch{train_configs.batch_size}'

    model.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
    model.summary()

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    data_gen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1,
                                  horizontal_flip=True)

    it_train = data_gen.flow(train_images, train_labels, batch_size=train_configs.batch_size)

    checkpoint = ModelCheckpoint(path_configs.model_weights_folder_name + '/' + model_name + '_checkpoint',
                                 monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False,
                                 mode='auto', period=1)

    early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')

    steps = int(train_images.shape[0] / train_configs.batch_size)
    history = model.fit_generator(it_train, steps_per_epoch=steps, epochs=train_configs.epochs,
                                  validation_data=(test_images, test_labels), verbose=2, callbacks=[checkpoint, early])

    model.save_weights(path_configs.model_weights_folder_name + '/' + model_name)
    utils.save_history(history.history,
                       path_configs.training_histories_folder_name + '/' + model_name + '_history.json')
