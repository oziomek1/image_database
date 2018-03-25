import os
import random
import time
import shutil

path = '../kostki/gen_x2/'
train_dir = 'train/'
test_dir = 'test/'

"""
This script is made for moving a fraction of file to 
different directory, preserving same directories structure
"""


def make_test_dirs():
    try:
        os.mkdir(path + test_dir)
    except OSError:
        if not os.path.isdir(path + test_dir):
            raise
    for directory in os.listdir(path + train_dir):
        try:
            os.mkdir(path + test_dir + directory)
        except OSError:
            if not os.path.isdir(path + test_dir + directory):
                raise


def remove_test_dirs():
    shutil.rmtree(path + test_dir)


def separate_test_data(factor):
    make_test_dirs()
    iter = 0
    for directory in os.listdir(path + train_dir):
        print(directory)
        for filename in os.listdir(path + train_dir + directory):
            if iter % int(1/factor) == 0:
                shutil.move(path + train_dir + directory + '/' + filename, path + test_dir + directory + '/' + filename)
            iter += 1


def merge_test_train_data():
    for directory in os.listdir(path + test_dir):
        print(directory)
        for filename in os.listdir(path + test_dir + directory):
            shutil.move(path + test_dir + directory + '/' + filename, path + train_dir + directory + '/' + filename)

    remove_test_dirs()


separate_test_data(0.2)
# merge_test_train_data()

