from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import SGD, adam, Adagrad
from keras.models import model_from_json
from scipy.io import loadmat, savemat
import csv
import os, struct, re, shutil
import numpy as np
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import statistics
import pyprog
from collections import Counter
from sklearn.utils import class_weight


def create_model():
    print("Create Model")
    model = Sequential()
    model.add(Dense(512, input_dim=4096,init='glorot_normal',W_regularizer=l2(0.001),activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(32,init='glorot_normal',W_regularizer=l2(0.001)))
    model.add(Dropout(0.6))
    model.add(Dense(1,init='glorot_normal',W_regularizer=l2(0.001),activation='sigmoid'))
    return model


class Dataset():
    def __init__(self, root, test_flag, free_flag, num_frames):
        self.root = root                                                                          # root containing if it's train/val/test
        self.free_flag = free_flag                                                                # Unlabeled set flag
        self.test_flag = test_flag
        files = files_test if test_flag else files_train
        classes, class_to_idx = self._find_classes() if not free_flag else (None, None)  # Extract classes from csv files
        self.classes = classes
        self.class_to_idx = class_to_idx
        if test_flag:
            self.samples = [(sample[0], 0 if np.sum(notes_test[i*num_frames:(i*num_frames)+num_frames]) < num_frames/2 else 1) for i, sample in enumerate(files)] if not free_flag else np.concatenate(files)  # Positive segment block if sum > half
        else:
            self.samples = [(sample, int(target)) for sample, target in files] if not free_flag else np.concatenate(files)
        self.targets = [s[1] for s in self.samples] if not free_flag else None

    def _find_classes(self):
        files = files_test if self.test_flag else files_train
        classes = [file[1] for file in files]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):  # When accessing to element in dataset

        if self.free_flag:
            path = self.samples[index]
        else:
            path, target = self.samples[index]

        sample = read_binary_blob(path)  # read binary file from path in sample

        sample_norm = []
        for i, value in enumerate(sample):
            sample_norm.append( ((value - minmax[0][i]) / (minmax[1][i] - minmax[0][i])) if minmax[1][i] > 0 else 0 )  # Normalize sample to previous calculated minmax on training set

        return [sample_norm] if self.free_flag else (sample_norm, target)

    def __len__(self):
        return len(self.samples)


def save_model(model, json_path, weight_path): # Function to save the model
    json_string = model.to_json()
    open(json_path, 'w').write(json_string)
    dict = {}
    i = 0
    for layer in model.layers:
        weights = layer.get_weights()
        my_list = np.zeros(len(weights), dtype=np.object)
        my_list[:] = weights
        dict[str(i)] = my_list
        i += 1
    savemat(weight_path, dict)


def load_model(json_path):  # Function to load the model
    model = model_from_json(open(json_path).read())
    return model


def load_weights(model, weight_path):  # Function to load the model weights
    dict2 = loadmat(weight_path)
    dict = conv_dict(dict2)
    i = 0
    for layer in model.layers:
        weights = dict[str(i)]
        layer.set_weights(weights)
        i += 1
    return model


def conv_dict(dict2):
    dict = {}
    for i in range(len(dict2)):
        if str(i) in dict2:
            if dict2[str(i)].shape == (0, 0):
                dict[str(i)] = dict2[str(i)]
            else:
                weights = dict2[str(i)][0]
                weights2 = []
                for weight in weights:
                    if weight.shape in [(1, x) for x in range(0, 5000)]:
                        weights2.append(weight[0])
                    else:
                        weights2.append(weight)
                dict[str(i)] = weights2
    return dict


def humanSort(text):  # Sort function for strings w/ numbers
    convText = lambda seq: int(seq) if seq.isdigit() else seq.lower()
    arrayKey = lambda key: [convText(s) for s in re.split('([0-9]+)', key)]  # Split numbers and chars, base function for sorted
    return sorted(text, key=arrayKey)


def humanSortSample(text):  # Sort function for strings w/ numbers
    conv_text = lambda seq: int(seq) if seq.isdigit() else seq.lower()
    array_key = lambda key: [conv_text(s) for s in re.split('([0-9]+)', str(key[1]))]  # Split numbers and chars, base function for sorted, using second position of the sample tuple
    return sorted(text, key=array_key)


def read_binary_blob(fn):
    fid = open(fn, 'rb')
    t = struct.unpack('i', fid.read(4))[0]  # unpack byte (4 bits) from fn file for topology
    c = struct.unpack('i', fid.read(4))[0]
    l = struct.unpack('i', fid.read(4))[0]
    h = struct.unpack('i', fid.read(4))[0]
    w = struct.unpack('i', fid.read(4))[0]
    total = t * c * l * h * w

    ret = np.zeros(total, np.float32)
    for i in range(total):
        ret[i] = struct.unpack('f', fid.read(4))[0]  # Extract byte to byte the main info

    fid.close()
    return ret


def load_dataset_train_batch(train_dataset, batchsize):

    n_exp            = int(batchsize/2)  # Number of abnormal and normal videos

    abnormal_samples = [i for i, sample in enumerate(train_dataset.samples) if sample[1] == 1]  # Get indexes of target 1
    normal_samples   = [i for i, sample in enumerate(train_dataset.samples) if sample[1] == 0]  # Get indexes of target 0
    num_abnormal     = len(abnormal_samples)  # Total number of abnormal videos in Training Dataset.
    num_normal       = len(normal_samples)    # Total number of Normal videos in Training Dataset

    abnor_list_iter  = np.random.permutation(num_abnormal)
    abnor_list_iter  = abnor_list_iter[num_abnormal-n_exp:] if num_abnormal >= n_exp else abnor_list_iter # Indexes for randomly selected Abnormal Videos
    norm_list_iter   = np.random.permutation(num_normal)
    norm_list_iter   = norm_list_iter[num_normal-n_exp:]     # Indexes for randomly selected Normal Videos

    all_features      = []  # To store C3D features of a batch
    all_labels        = []  # To store labels of respective batch

    stats = [train_dataset.samples[abnormal_samples[i]][0] for i in abnor_list_iter] + [train_dataset.samples[normal_samples[i]][0] for i in norm_list_iter]

    video_count = 0
    for i in abnor_list_iter:
        if video_count == 0:
            all_features = train_dataset[abnormal_samples[i]][0]
            all_labels   = train_dataset[abnormal_samples[i]][1]
        else:
            all_features = np.vstack((all_features, train_dataset[abnormal_samples[i]][0]))
            all_labels   = np.vstack((all_labels,   train_dataset[abnormal_samples[i]][1]))
        video_count += 1

    for i in norm_list_iter:
        all_features = np.vstack((all_features, train_dataset[normal_samples[i]][0]))
        all_labels   = np.vstack((all_labels,   train_dataset[normal_samples[i]][1]))

    return all_features, all_labels, stats


def auc(model, val_dataset, iteration, aton_iteration):
    scores = []
    gt     = []

    for iv in range(len(val_dataset)):
        inputs = val_dataset[iv]
        predictions = model.predict_on_batch(np.array(inputs[0:1]))  # Get anomaly prediction for each of video segments.
        gt.append(inputs[1])
        scores.append(predictions[0][0])

    AUC = roc_auc_score(np.array(gt), np.array(scores))
    fpr, tpr, thresholds = roc_curve(np.array(gt), np.array(scores))


    savemat(os.path.join('results/strong/VAL', str(aton_iteration) ,'training_AUC_'+str(iteration)+'.mat'), {'AUC': AUC, 'X': fpr, 'Y': tpr, 'scores': scores, 'gt':notes_test})

    return AUC


def read_annotation(path, flag_test):
    global notes_test, notes_train
    notes_test, notes_train = [], []
    with open(path, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            notes_test.append(int(row[0])) if flag_test else notes_train.append(int(row[0]))


def read_file_loader(path, flag_test):
    global files_train, files_test
    files_train = [] if not flag_test else files_train
    files_test  = [] if flag_test else files_test
    with open(path, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            files_test.append(row) if flag_test else files_train.append(row)


def train(path_loader, min_iterations, aton_iteration, val_file, val_notes, num_features, batchsize, save_best):

    read_file_loader(path_loader, 0)                                              # Load train set
    read_file_loader(val_file, 1)                                                 # Load val set
    read_annotation(val_notes, 1)                                                 # Load val annotation

    model   = create_model()
    adagrad = Adagrad(lr=0.01, epsilon=1e-08)
    model.compile(loss='binary_crossentropy', optimizer=adagrad)

    Results_Path      = os.path.join('results/strong/VAL', str(aton_iteration))   # Directory to save val stats in training
    output_dir        = os.path.join('models/strong_model', str(aton_iteration))  # Directory to save models and checkpoints
    model_path        = os.path.join(output_dir, 'model.json')
    best_model_path   = os.path.join(output_dir, 'best_model.json')
    best_weights_path = os.path.join(output_dir, 'best_weights.mat')

    if not os.path.exists(Results_Path):
        os.makedirs(Results_Path)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_dataset    = Dataset('dataset/train', False, False, num_features)  # Create train dataset
    val_dataset      = Dataset('dataset/val', True, False, num_features)     # Create val dataset
    loss_graph       = []
    full_batch_loss  = []
    total_iterations = 0
    bestAUC          = 0
    previousAUC      = [0]

    print('Train dataset: ' + str(len(train_dataset)))

    plt.ion()

    print('Training Strong Classifier...')

    prog = pyprog.ProgressBar("", " AUC - " + str(round(previousAUC[-1], 4))+ '%', total=min_iterations, bar_length=50, complete_symbol="=", not_complete_symbol=" ", wrap_bar_prefix=" [", wrap_bar_suffix="] ", progress_explain="", progress_loc=pyprog.ProgressBar.PROGRESS_LOC_END)
    prog.update()

    while total_iterations != min_iterations:
        inputs, targets, stats = load_dataset_train_batch(train_dataset, batchsize)  # Load normal and abnormal video C3D features
        batch_loss = model.train_on_batch(inputs, targets)

        full_batch_loss.append(float(batch_loss))
        statistics.stats_batch(full_batch_loss, aton_iteration)

        loss_graph = np.hstack((loss_graph, batch_loss))
        total_iterations += 1
        if total_iterations % 20 == 0:
            iteration_path = output_dir + 'Iterations_graph_' + str(total_iterations) + '.mat'
            savemat(iteration_path, dict(loss_graph=loss_graph))                           # Loss checkpoint
            previousAUC.append(auc(model, val_dataset, total_iterations, aton_iteration))  # Validation results

            if previousAUC[-1] > bestAUC and save_best:
                save_model(model, best_model_path, best_weights_path)                       # Best model checkpoint
                bestAUC = previousAUC[-1]

            weights_path = output_dir + 'weightsStrong_' + str(total_iterations) + '.mat'
            save_model(model, model_path, weights_path)                                     # Model checkpoint

        prog.set_suffix(" AUC - " + str(round(previousAUC[-1], 4))+ '% | Best AUC - ' + str(round(bestAUC, 4))+ '%')
        prog.set_stat(total_iterations)
        prog.update()

    prog.end()

    plt.ioff()
    save_model(model, best_model_path, best_weights_path) if not save_best else None  # Save last as best if the best was not kept


def test(test_val_flag, aton_iteration, test_file, test_notes, num_features):
    results = 'FINAL/' if test_val_flag else 'VAL/'

    read_file_loader(test_file,1)                                            # Load test/val set
    read_annotation(test_notes, 1)                                           # Load test/val annotations

    test_dataset = Dataset('dataset/test', True, False, num_features)        # Create test dataset
    Results_Path = os.path.join('results/strong', results, str(aton_iteration))
    model_dir    = os.path.join('models/strong_model', str(aton_iteration))  # Directory to the models
    weights_path = os.path.join(model_dir, 'best_weights.mat')
    model_path   = os.path.join(model_dir, 'best_model.json')                # Path to trained model

    if not os.path.exists(Results_Path):
        os.makedirs(Results_Path)

    model = load_model(model_path)
    load_weights(model, weights_path)

    scores           = []
    gt               = []

    print('Testing Strong Classifier...')

    prog = pyprog.ProgressBar("", "", total=len(test_dataset), bar_length=50, complete_symbol="=", not_complete_symbol=" ", wrap_bar_prefix=" [", wrap_bar_suffix="] ", progress_explain="", progress_loc=pyprog.ProgressBar.PROGRESS_LOC_END)
    prog.update()


    for iv in range(len(test_dataset)):
        inputs = test_dataset[iv]
        predictions = model.predict_on_batch(np.array(inputs[0:1]))  # Get anomaly prediction for each of 16 frame segment.
        gt.append(inputs[1])
        scores.append(predictions[0][0])

        prog.set_stat(iv)
        prog.update()


    prog.end()
    AUC = roc_auc_score(np.array(gt), np.array(scores))
    fpr, tpr, thresholds = roc_curve(np.array(gt), np.array(scores))
    print(AUC)
    savemat(os.path.join(Results_Path, 'eval_AUC_'+str(aton_iteration)+'.mat'), {'AUC': AUC, 'X': fpr, 'Y': tpr, 'scores': scores, 'gt': notes_test})

    return np.array(gt), np.array(scores)


def predict_pattern(test_val_flag, aton_iteration, execute_file, execute_notes, num_frames):

    read_file_loader(execute_file,1)                                                                 # Load test/val set
    read_annotation(execute_notes, 1) if test_val_flag == 0 else None

    test_dataset = Dataset('dataset/test', True, False if test_val_flag == 0 else True, num_frames)  # Create Test/Unlabel dataset
    model_dir    = os.path.join('models/strong_model', str(aton_iteration))                          # Directory to the models
    weights_path = os.path.join(model_dir, 'best_weights.mat')
    model_path   = os.path.join(model_dir, 'best_model.json')                                        # Path to trained model

    model = load_model(model_path)
    load_weights(model, weights_path)

    scores = []
    gt     = []

    text = 'Executing Strong Model Free Dataset...' if test_val_flag == 0 else 'Predict Pattern Strong Model...'
    print(text)

    prog = pyprog.ProgressBar("", "", total=len(test_dataset), bar_length=50, complete_symbol="=", not_complete_symbol=" ", wrap_bar_prefix=" [", wrap_bar_suffix="] ", progress_explain="", progress_loc=pyprog.ProgressBar.PROGRESS_LOC_END)
    prog.update()

    for iv in range(len(test_dataset)):
        inputs = test_dataset[iv]
        predictions = model.predict_on_batch(np.array(inputs[0:1] if test_val_flag == 0 else inputs))  # Get anomaly prediction for each of 16 frame segment.
        gt.append(inputs[1]) if test_val_flag == 0 else None
        scores.append(predictions[0][0])

        prog.set_stat(iv)
        prog.update()

    prog.end()
    AUC = roc_auc_score(np.array(gt), np.array(scores)) if test_val_flag == 0 else None
    print(AUC) if test_val_flag == 0 else None

    print(Counter(np.array(gt)))

    return (np.array(gt) if test_val_flag == 0 else None), (np.array(scores))


def load_norm(norm_file):
    with open(norm_file, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            minmax.append(np.array(row).astype(float))


notes_train, notes_test, files_train, files_test, minmax = [], [], [], [], []






