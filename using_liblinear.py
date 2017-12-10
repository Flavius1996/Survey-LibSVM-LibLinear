# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:28:54 2017

@author: Haiit
"""
import json
import h5py
import numpy as np
import liblinear as lib
import liblinearutil as libu
import helper


def survey_liblinear(train_features,
                     train_labels,
                     test_features,
                     test_labels,
                     is_debug=False,
                     input_as_a_file=False,
                     output_result=False,
                     class_indices_path="",  # set this param if you wanna output images after classification
                     output_folder=""):

    # if input are files we saved

    if input_as_a_file:
        if is_debug:
            print("Load data....")
        # read features
        h5f_data = h5py.File(train_features, 'r')
        if is_debug:
            print("Loaded")
        train_f = np.array(h5f_data['TRAIN'])
        if is_debug:
            print("Train features: " + str(len(train_f)))
        h5f_data.close()
        h5f_data = h5py.File(test_features, 'r')
        test_f = np.array(h5f_data['TEST'])
        if is_debug:
            print("Test features: " + str(len(test_f)))
        h5f_data.close()
        # read labels
        with open(train_labels) as f:
            train_lb = json.load(f)
        with open(test_labels) as f:
            test_lb = json.load(f)

    else:
        train_f = train_features
        test_f = test_features
        if is_debug:
            print("Train features: " + str(len(train_f)))
            print("Test features: " + str(len(test_f)))
        train_lb = train_labels
        test_lb = test_labels

    main_lb_train = [int(x['label_num']) for x in train_lb]
    main_lb_test = [int(x['label_num']) for x in test_lb]

    if is_debug:
        print("Training....")
    # train
    prob = lib.problem(main_lb_train, train_f)
    m = libu.train(prob, '-q')
    if is_debug:
        print("Predicting....")

    p_label, p_acc, p_val = libu.predict(main_lb_test, test_f, m)

    # save images after classification
    if output_result:
        helper.output_after_classification(p_label, test_lb, class_indices_path, output_folder, is_debug)

    return p_acc


#survey_liblinear("ROOT\\TEST_SETS\\0\\features.h5",
#                 "ROOT\\TEST_SETS\\0\\train_files.json",
#                 "ROOT\\TEST_SETS\\0\\features.h5",
#                 "ROOT\\TEST_SETS\\0\\test_files.json",
#                 True, True, True,
#                 "ROOT\\DATASET\\class_indices.json",
#                 "ROOT\\output")

