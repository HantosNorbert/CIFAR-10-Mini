from prepare_database import load_subset_database
import utils
from train import train_basic_model, train_advanced_model
from experimental_train import train_experimental_model
import argparse
import logging


def main(args):
    path_config, subset_selection_config, simple_training_config, advanced_training_config = \
        utils.parse_config_file(args.config_file_name)

    utils.init_gpu(memory_limit=args.gpu_memory_limit)

    (train_images, train_labels), (test_images, test_labels) = load_subset_database(args, path_config,
                                                                                    subset_selection_config)
    assert len(train_images) == len(train_labels)
    assert len(test_images) == len(test_labels)
    logging.info(f'Number of train samples: {len(train_images)}')
    logging.info(f'Number of test samples: {len(test_images)}')

    if args.scenario == 'experimental':
        train_experimental_model(train_images, train_labels, test_images, test_labels)

    elif args.scenario == 'simple':
        train_basic_model(args, path_config, simple_training_config,
                          train_images, train_labels, test_images, test_labels)

    elif args.scenario == 'advanced':
        train_advanced_model(args, path_config, advanced_training_config,
                             train_images, train_labels, test_images, test_labels)

    else:
        assert False, f'Unhandled scenario configuration: {args.scenario}'


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(filename)s][%(funcName)s()] %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    parser = argparse.ArgumentParser(prog='main.py')

    parser.add_argument('config_file_name', help='Config file name', type=str)
    parser.add_argument('subset_selection_method',
                        help='Selection method of a subset of the training data. Options: '
                             'train feature vector extractor network and search for indices / '
                             'load feature vector extractor network weights and search for indices / '
                             'load previously searched indices',
                        choices=['train_fv_extractor', 'load_fv_extractor', 'load_indices'])
    parser.add_argument('-g', '--gpu_memory_limit', help='GPU memory limit in bytes (default: 2048)', type=int,
                        default=2048)

    subparsers = parser.add_subparsers(help='Simple, advanced, or experimental training scenario', dest='scenario')
    subparsers.required = True

    # create the parser for the 'simple' command
    parser_simple = subparsers.add_parser('simple', help='Chose a simple network with basic training')
    parser_simple.add_argument('model', type=str, help='Select a simple model',
                               choices=['VGG_8_simple', 'VGG_16_simple', 'ResNet_18_simple'])
    parser_simple.add_argument('-a', '--use_augmentation', action='store_true',
                               help='Use data augmentation on the training images')

    # create the parser for the 'advanced' command
    parser_advanced = subparsers.add_parser('advanced',
                                            help='Chose between advanced training options for a fixed network')
    parser_advanced.add_argument('training_method', help='Training method', choices=['SGD', 'Adam'])
    parser_advanced.add_argument('-d', '--use_dropout', action='store_true', help='Use dropout')
    parser_advanced.add_argument('-w', '--use_weight_decay', action='store_true',
                                 help='Use weight decay (l2 kernel regularizer)')

    # create the parser for the 'experimental' scenario
    parser_experimental = subparsers.add_parser('experimental', help='Train a network with two output layers')

    parsed_args = parser.parse_args()

    main(parsed_args)
