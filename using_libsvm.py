# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 16:28:54 2017

@author: Haiit
"""
import svm as libsvm
import svmutil as libsvmu
import json
import h5py
import numpy as np
import helper


def survey_svm(train_features,
               train_labels,
               test_features,
               test_labels,
               is_linear=False,
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
        train_f = np.array(h5f_data['TRAIN']).tolist()
        h5f_data.close()
        h5f_data = h5py.File(test_features, 'r')
        test_f = np.array(h5f_data['TEST']).tolist()
        h5f_data.close()
        # read labels
        with open(train_labels) as f:
            train_lb = json.load(f)
        with open(test_labels) as f:
            test_lb = json.load(f)

    else:
        train_f =  np.array(train_features).tolist()
        test_f =  np.array(test_features).tolist()
        if is_debug:
            print("Train features: " + str(len(train_f)))
            print("Test features: " + str(len(test_f)))
        train_lb = train_labels
        test_lb = test_labels

    main_lb_train = [float(x['label_num']) for x in train_lb]
    main_lb_test = [float(x['label_num']) for x in test_lb]

    if is_debug:
        print("Training....")

    # train
    prob = libsvm.svm_problem(main_lb_train, train_f)
    if is_linear:
        m = libsvmu.svm_train(prob, '-t 0 -q')
    else:
        m = libsvmu.svm_train(prob, '-t 2 -q')

    if is_debug:
        print("Predicting....")
    # predict
    p_label, p_acc, p_val = libsvmu.svm_predict(main_lb_test, test_f, m)

    # save images after classification
    if output_result:
        helper.output_after_classification(p_label, test_lb, class_indices_path, output_folder, is_debug)

    return p_acc