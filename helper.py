# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:28:54 2017
@author: Haiit

EXTRACT VGG16 DEEP FEATURES &
HELP CREATE TEST SET
"""
import json
import random
import numpy as np
import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from operator import itemgetter
from shutil import copyfile


# Create struct of test set
def create_test_set(file_indices_path, num_of_test):
    # load image paths with labels
    with open(file_indices_path) as f:
        files = json.load(f)

    total_image = len(files)
    num_test_image = total_image * 0.2
    num_train_image = total_image * 0.8

    test_sets = []

    # create test set
    for i in range(0, num_of_test):
        random_list = random.sample(range(0, total_image), total_image)
        test_indices = random_list[:int(num_test_image)]
        train_indices = random_list[-int(num_train_image):]
        # get item
        test = itemgetter(*test_indices)(files)
        train = itemgetter(*train_indices)(files)

        test_sets.append((i, list(train), list(test)))

    return test_sets


# Extract deep features
def extract_deep_features_block5_pool(file_indices, last_extracted=None, is_debug=False):
    # our features will be store in this var
    features = []
    # Load keras vgg16 pretrained model, no including FC
    model = VGG16(weights='imagenet', include_top=False)

    # extracting deep features
    if is_debug:
        print("Extracting deep features...")

    counter = 0
    num_files = len(file_indices)

    for img in file_indices:
        if is_debug:
            if counter == int(num_files / 2):
                print("Reached 50% data ...")
        counter += 1

        path = img['path']
        if last_extracted is not None:
            if img['id'] in last_extracted:
                features.append(last_extracted[img['id']])
            else:
                # read image
                _img = image.load_img(path, target_size=(100, 100))
                x = image.img_to_array(_img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                feature = model.predict(x)
                flat = feature.flatten() # our feature
                # add to list
                features.append(flat)
                last_extracted[img['id']] = flat
        else:
            # read image
            _img = image.load_img(path, target_size=(100, 100))
            x = image.img_to_array(_img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feature = model.predict(x)
            flat = feature.flatten()  # our feature
            # add to list
            features.append(flat)

    if is_debug:
        print("Extracted deep features !")

    return features

def extract_deep_features_fc2(file_indices, last_extracted=None, is_debug=False):
    # our features will be store in this var
    features = []
    # Load keras vgg16 pretrained model, no including FC
    model = VGG16(weights='imagenet', include_top=True)
    model_extractfeatures = Model(input=model.input, output=model.get_layer('fc2').output)

    # extracting deep features
    if is_debug:
        print("Extracting deep features...")

    counter = 0
    num_files = len(file_indices)

    for img in file_indices:
        if is_debug:
            if counter == int(num_files / 2):
                print("Reached 50% data ...")
        counter += 1

        path = img['path']
        if last_extracted is not None:
            if img['id'] in last_extracted:
                features.append(last_extracted[img['id']])
            else:
                # read image
                _img = image.load_img(path, target_size=(100, 100))
                x = image.img_to_array(_img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                feature = model.predict(x)
                flat = feature.flatten() # our feature
                # add to list
                features.append(flat)
                last_extracted[img['id']] = flat
        else:
            # read image
            _img = image.load_img(path, target_size=(100, 100))
            x = image.img_to_array(_img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feature = model.predict(x)
            flat = feature.flatten()  # our feature
            # add to list
            features.append(flat)

    if is_debug:
        print("Extracted deep features !")

    return features


def calculate_delta(acc_list, avg_acc):
    delta = 0
    for acc in acc_list:
        delta += abs(avg_acc - acc)
    delta = float(delta / len(acc_list))
    return delta


def output_after_classification(p_label, test_set, class_indices_path, output_folder, is_debug=False):

    if is_debug:
        print("Making output....")

    with open(class_indices_path) as f:
        indices = json.load(f)

    idx = 0
    for test in test_set:
        path = test['path']
        class_name = indices[str(int(p_label[idx]))]
        idx += 1
        # create folder of that class
        try:
            os.stat(output_folder + "\\" + class_name)
        except:
            os.mkdir(output_folder + "\\" + class_name)
        # copy file
        copyfile(path, output_folder + "\\" + class_name + "\\" + os.path.basename(path))

    if is_debug:
        print("Made output !")
