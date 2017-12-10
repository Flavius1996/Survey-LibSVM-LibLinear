"""
Created on Mon Dec  4 16:28:54 2017
@author: Haiit

SURVEY LIBLINEAR - LIBSVM_RBF - LIBSVM_LINEAR
"""
import helper
import os
import json
import h5py
import numpy as np
import using_liblinear
import using_libsvm

# Paths
file_indices_path = "ROOT\\DATASET\\file_indices.json"
test_set_path = "ROOT\\TEST_SETS"

# Config
is_debug = True
save_file_for_report = True

# Create 10 test sets
num_test = 5
test_set = helper.create_test_set(file_indices_path, num_test)

# Create folder structure (for the report)
if save_file_for_report:
    if is_debug:
        print("Create test-set folders")
    for i, tr, te in test_set:
        test_set_folder = test_set_path + "\\" + str(i)
        try:
            os.stat(test_set_folder)
        except:
            os.mkdir(test_set_folder)
        # Save indices
        with open(test_set_folder + "\\train_files.json", 'w') as outfile:
            json.dump(tr, outfile)
        with open(test_set_folder + "\\test_files.json", 'w') as outfile:
            json.dump(te, outfile)

features = []

last_extracted = {} # remember the same feature extracted to do faster

# Extract deep features
if is_debug:
    print("Extracting deep features...")

for i, tr, te in test_set:

    if is_debug:
        print("\n\nFor train set of " + str(i))

    train_features = helper.extract_deep_features(tr, is_debug=is_debug, last_extracted=last_extracted)

    if is_debug:
        print("For test set of " + str(i))

    test_features = helper.extract_deep_features(te, is_debug=is_debug, last_extracted=last_extracted)

    if is_debug:
        print("Extracted on " + str(i))

    features.append((i, train_features, test_features))

    if is_debug:
            print("\n\n")
        
    # save features to file
    if save_file_for_report:
        test_set_folder = test_set_path + "\\" + str(i) + "\\"

        if is_debug:
            print("Saving features to file...")

        h5f_data = h5py.File(test_set_folder + "features.h5", 'w')
        h5f_data.create_dataset('TRAIN', data=np.array(train_features))
        h5f_data.create_dataset('TEST', data=np.array(test_features))
        # close H5 writer
        h5f_data.close()

        if is_debug:
            print("Saved features !")

if is_debug:
    print("\n\n")
        
acc_list_linear = []
acc_list_svm_l = []
acc_list_svm_rbf = []

acc_avg_linear = 0
acc_avg_svm_l = 0
acc_avg_svm_rbf = 0

for i, tr, te in test_set:
    if is_debug:
        print("TEST SET " + str(i) + "\n\n")
    index, train_f, test_f = features[int(i)]
    if is_debug:
        print("Lib_linear")
    liblinear_acc = using_liblinear.survey_liblinear(train_f, tr, test_f, te, is_debug=True)
    if is_debug:
        print("\nLib_SVM_linear")
    libsvm_linear_acc = using_libsvm.survey_svm(train_f, tr, test_f, te, is_linear=True,  is_debug=True)
    if is_debug:
        print("\nLib_SVM_RBF")
    libsvm_rbf_acc = using_libsvm.survey_svm(train_f, tr, test_f, te, is_linear=False, is_debug=True)

    """, output_result=True,class_indices_path="ROOT\\DATASET\\class_indices.json",output_folder="ROOT\\output")"""

    acc_list_linear.append(liblinear_acc[0])
    acc_list_svm_l.append(libsvm_linear_acc[0])
    acc_list_svm_rbf.append(libsvm_rbf_acc[0])

    acc_avg_linear += liblinear_acc[0]/100
    acc_avg_svm_l += libsvm_linear_acc[0]/100
    acc_avg_svm_rbf += libsvm_rbf_acc[0] / 100

acc_avg_linear = acc_avg_linear / num_test
acc_avg_svm_l = acc_avg_svm_l / num_test
acc_avg_svm_rbf = acc_avg_svm_rbf / num_test

delta1 = helper.calculate_delta(acc_list_linear, acc_avg_linear)
delta2 = helper.calculate_delta(acc_list_svm_l, acc_avg_svm_l)
delta3 = helper.calculate_delta(acc_list_svm_rbf, acc_avg_svm_rbf)

print("RESULT LIBLINEAR: " + str(acc_avg_linear) + "+-" + str(delta1))
print("RESULT LIBSVM_LINEAR: " + str(acc_avg_svm_l) + "+-" + str(delta2))
print("RESULT LIBSVM_RBF: " + str(acc_avg_svm_rbf) + "+-" + str(delta3))
