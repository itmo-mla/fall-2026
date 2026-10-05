from random import randint

RANDOM_STATE = randint(1, 65535)
# RANDOM_STATE = 35005
print(f'{RANDOM_STATE=}', end='\n\n')

DATASET_DOWNLOAD_PATH = 'mssmartypants/water-quality'
DATASET_FILENAME = 'waterQuality1.csv'
DATASET_TARGET = 'is_safe'
DATASET_NAME = 'Water Quality Dataset'
