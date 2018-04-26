__author__ = 'jjamjung'

import argparse
import operator
from functools import reduce

ENC_A = "encoder_A"
ENC_B = "encoder_B"
DEC_A = "decoder_A"
DEC_B = "decoder_B"
DISC_A = "discriminator_A"
DISC_B = "discriminator_B"

RECON_LOSS_A = "recon_A"
RECON_LOSS_B = "recon_B"
CYCLE_LOSS_A = "cycle_A"
CYCLE_LOSS_B = "cycle_B"
FM_LOSS_A = "feature_A"
FM_LOSS_B = "feature_B"
DISC_LOSS_A = "disc_A"
DISC_LOSS_B = "disc_B"
GAN_LOSS_A = "gan_A"
GAN_LOSS_B = "gan_B"

LJ_STR = 'LJSpeech-1.0'
BLIZZARD_STR = 'Blizzard2012'
TRAINING_STR = 'training'
VALIDATION_STR = 'validation'
TEST_STR = 'test'
META_FILE_NAME = 'metadata.txt'


def calc_model_size(model):
    model_size = 0
    for vv in list(model.parameters()):
        model_size += reduce(operator.mul, vv.size())
    return model_size


def print_modules_size(modules_dict):
    total_model_size = 0
    for module_key, module in modules_dict.items():
        model_size = calc_model_size(module)
        total_model_size += model_size
        print("%s size: %d" % (module_key, model_size))
    print("Total model size: %d\n" % total_model_size)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
