import os

from etc import *

__author__ = 'jjamjung'


PROJECT_PATH = os.path.dirname(os.path.realpath(__file__))

DATA_BASE_PATH = os.path.join(PROJECT_PATH, 'dataset')

DATA_LJ_PATH = os.path.join(DATA_BASE_PATH, LJ_STR)
DATA_LJ_TRAINING_PATH = os.path.join(DATA_LJ_PATH, TRAINING_STR)
DATA_LJ_VALIDATION_PATH = os.path.join(DATA_LJ_PATH, VALIDATION_STR)
DATA_LJ_TEST_PATH = os.path.join(DATA_LJ_PATH, TEST_STR)

DATA_BLIZZARD_PATH = os.path.join(DATA_BASE_PATH, BLIZZARD_STR)
DATA_BLIZZARD_TRAINING_PATH = os.path.join(DATA_BLIZZARD_PATH, TRAINING_STR)
DATA_BLIZZARD_VALIDATION_PATH = os.path.join(DATA_BLIZZARD_PATH, VALIDATION_STR)
DATA_BLIZZARD_TEST_PATH = os.path.join(DATA_BLIZZARD_PATH, TEST_STR)

DICTIONARY_PATH = os.path.join(PROJECT_PATH, 'dictionary.txt')

MODEL_BASE_PATH = os.path.join(PROJECT_PATH, 'model')

RESULT_PATH = os.path.join(PROJECT_PATH, 'result')

if not os.path.exists(MODEL_BASE_PATH):
    os.mkdir(MODEL_BASE_PATH)

if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)
