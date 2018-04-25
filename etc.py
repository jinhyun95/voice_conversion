__author__ = 'jjamjung'

import argparse
import operator
from functools import reduce

TEXT_ENC = "text_enc"
WAVE_DEC = "wave_dec"
ASR_MODULE = "asr_module"
Z_DISC = "z_disc"
W_DISC = "w_disc"

STEP_GEN_1ST = "generator_step_1st"
STEP_DISC_1ST = "discriminator_step_1st"
STEP_GEN_2ND = "generator_step_2nd"
STEP_DISC_2ND = "discriminator_step_2nd"

T_RECON_LOSS = "Text_recon"
W_DISC_LOSS = "Wave_disc"
W_GAN_LOSS = "Wave_GAN"
W_RECON_LOSS = "Wave_recon"
MEL_LOSS = "Mel"
SPEC_LOSS = "Spec"
Z_DISC_LOSS = "Text_disc"
Z_GAN_LOSS = "Text_GAN"

TTS_LOSS = "TTS"
ASR_LOSS = "ASR"

# ORDERED_LOSS_KEYS = {STEP_GEN_1ST: [T_RECON_LOSS, W_GAN_LOSS],
#                      STEP_DISC_1ST: [W_DISC_LOSS],
#                      STEP_GEN_2ND: [W_RECON_LOSS, MEL_LOSS, SPEC_LOSS, Z_GAN_LOSS],
#                      STEP_DISC_2ND: [Z_DISC_LOSS]}

ORDERED_LOSS_KEYS = {STEP_GEN_1ST: [T_RECON_LOSS, W_GAN_LOSS],
                     STEP_DISC_1ST: [W_DISC_LOSS],
                     STEP_GEN_2ND: [W_RECON_LOSS, MEL_LOSS, SPEC_LOSS]}

GAN_STR = 'CycleGAN_TTS_ASR'
TACOTRON_STR = 'Tacotron'
DEEPSPEECH_STR = 'deepspeech.pytorch'
AE_STR = 'Autoencoder'

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
