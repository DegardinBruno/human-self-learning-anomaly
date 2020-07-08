from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adagrad
from scipy.io import loadmat, savemat
from keras.models import model_from_json
import tensorflow as tf
import csv
import os, re
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import statistics
import matplotlib.pyplot as plt
import pyprog

epoch     = tf.Variable(0,  dtype=tf.float32)
batch_obj = tf.Variable(60, dtype=tf.int16)
segmt_obj = tf.Variable(32, dtype=tf.int16)


def create_model():
    print("Create Model")
    model = Sequential()
    model.add(Dense(512, input_dim=4096,init='glorot_normal',W_regularizer=l2(0.001),activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(32,init='glorot_normal',W_regularizer=l2(0.001)))
    model.add(Dropout(0.6))
    model.add(Dense(1,init='glorot_normal',W_regularizer=l2(0.001),activation='sigmoid'))
    return model


def humanSort(text):  # Sort function for strings w/ numbers
    convText = lambda seq: int(seq) if seq.isdigit() else seq.lower()
    arrayKey = lambda key: [convText(s) for s in re.split('([0-9]+)', key)]  # Split numbers and chars, base function for sorted
    return sorted(text, key=arrayKey)


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


def save_model(model, json_path, weight_path):  # Function to save the model
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


def load_dataset_Train_batch(AbnormalPath, NormalPath, batchsize, segments):

    n_exp=int(batchsize/2)  # Number of abnormal and normal videos

    Num_abnormal = len(AbnormalPath)  # Total number of abnormal videos in Training Dataset.
    Num_Normal = len(NormalPath)      # Total number of Normal videos in Training Dataset.

    # We assume the features of abnormal videos and normal videos are located in two different folders.
    Abnor_list_iter = np.random.permutation(Num_abnormal)
    Abnor_list_iter = Abnor_list_iter[Num_abnormal-n_exp:]  # Indexes for randomly selected Abnormal Videos
    Norm_list_iter = np.random.permutation(Num_Normal)
    Norm_list_iter = Norm_list_iter[Num_Normal-n_exp:]      # Indexes for randomly selected Normal Videos


    AllFeatures = []  # To store C3D features of a batch

    for i in Abnor_list_iter:
        f = open(AbnormalPath[i], "r")
        words = f.read().split()
        num_feat = int(len(words) / 4096)
        # Number of features per video to be loaded. In our case num_feat=32, as we divide the video into 32 segments. Note that
        # we have already computed C3D features for the whole video and divide the video features into 32 segments. Please see Save_C3DFeatures_32Segments.m as well

        VideoFeatues = []
        for feat in range(0, num_feat):
            feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])

            VideoFeatues = np.vstack((VideoFeatues, feat_row1)) if len(VideoFeatues) > 0 else feat_row1

        AllFeatures = np.vstack((AllFeatures, VideoFeatues)) if len(AllFeatures) > 0 else VideoFeatues

    for i in Norm_list_iter:
        f = open(NormalPath[i], "r")
        words = f.read().split()
        num_feat = int(len(words) /4096)   # Number of features to be loaded. In our case num_feat=32, as we divide the video into 32 segments.

        VideoFeatues = []
        for feat in range(0, num_feat):

            feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])

            VideoFeatues = np.vstack((VideoFeatues, feat_row1)) if len(VideoFeatues) > 0 else feat_row1

        AllFeatures = np.vstack((AllFeatures, VideoFeatues))

    AllLabels = [0]*(n_exp*segments) + [1]*(n_exp*segments)

    return  AllFeatures,AllLabels


def entropy_tf(x):
    cx = tf.histogram_fixed_width(x, [0, 1], nbins=500)
    c_normalized = cx / tf.reduce_sum(cx)                                                   # Normalize histogram values
    index_nonzero = tf.where(tf.not_equal(c_normalized, tf.constant(0, dtype=tf.float64)))  # Get non zero values indexes
    c_normalized = tf.gather_nd(c_normalized, index_nonzero)                                # Get non zero values
    h = -tf.reduce_sum(c_normalized * tf.math.log(c_normalized))

    h = tf.dtypes.cast(h, dtype=tf.float32)
    return h


def custom_objective(y_true, y_pred):

    with tf.device('/device:XLA_GPU:0'):  # Accelerated Algebra (XLA) is faster, switch to device:GPU:0 if needed
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)

        n_seg = segmt_obj         # Because we have 32 segments per video.
        nvid  = batch_obj  # Batchsize
        n_exp = int(nvid / 2)

        sub_max        = tf.convert_to_tensor([])  # sub_max represents the highest scoring instances in each bag (video).
        sub_sum_labels = tf.convert_to_tensor([])  # Sum the labels in order to distinguish between normal and abnormal videos.
        sub_sum_l1     = tf.convert_to_tensor([])  # For holding the concatenation of summation of scores in the bag.
        sub_l2         = tf.convert_to_tensor([])  # For holding the concatenation of L2 of score in the bag.

        for ii in range(0, nvid, 1):
            # For Labels
            mm             = y_true[ii * n_seg:ii * n_seg + n_seg]
            sub_sum_labels = tf.concat([sub_sum_labels, tf.stack([tf.reduce_sum(mm)])], 0)  # Just to keep track of abnormal and normal vidoes

            # For Features scores
            Feat_Score = y_pred[ii * n_seg:ii * n_seg + n_seg]
            sub_max    = tf.concat([sub_max, tf.stack([tf.math.reduce_max(Feat_Score)])], 0)  # Keep the maximum score of scores of all instances in a Bag (video)
            sub_sum_l1 = tf.concat([sub_sum_l1, tf.stack([tf.reduce_sum(Feat_Score)])],  0)   # Keep the sum of scores of all instances in a Bag (video)

            # Temporal calculation
            z2     = tf.concat([[1], Feat_Score], 0)
            z3     = tf.concat([Feat_Score, [1]], 0)
            z      = z2 - z3
            z      = z[1:n_seg]
            z      = tf.reduce_sum(tf.math.square(z))
            sub_l2 = tf.concat([sub_l2, tf.stack([z])], 0)

        sub_sum_l1 = sub_sum_l1[:n_exp]
        sub_l2     = sub_l2[:n_exp]

        indx_nor = tf.where(tf.equal(sub_sum_labels, n_seg))  # Index of normal videos: Since we labeled 1 for each of 32 segments of normal videos F_labels=32 for normal video
        indx_abn = tf.where(tf.equal(sub_sum_labels, 0))

        Sub_Nor = tf.gather_nd(sub_max, indx_nor)  # Maximum Score for each of abnormal video
        Sub_Abn = tf.gather_nd(sub_max, indx_abn)  # Maximum Score for each of normal video

        z = tf.convert_to_tensor([])
        for ii in range(0, n_exp, 1):
            sub_z = tf.math.maximum(1 - Sub_Abn + Sub_Nor[ii], 0)
            z     = tf.concat([z, tf.stack([tf.reduce_sum(sub_z)])], 0)

        sample_entropy = entropy_tf(y_pred)

    z = (1/((epoch/10)+1)) * (tf.math.reduce_mean(z, axis=-1) + 0.00008 * tf.reduce_sum(sub_sum_l1) + 0.00008 * tf.reduce_sum(sub_l2)) - (tf.math.log(epoch)*30) * sample_entropy  # Final Loss

    return z


def load_dataset_One_Video_Features(Test_Video_Path):

    VideoPath = Test_Video_Path
    f         = open(VideoPath, "r")
    words     = f.read().split()
    num_feat  = int(len(words) / 4096)
    # Number of features per video to be loaded. In our case num_feat=32, as we divide the video into 32 segments. Note that
    # we have already computed C3D features for the whole video and divided the video features into 32 segments.

    VideoFeatues = []
    for feat in range(0, num_feat):
        feat_row1 = np.float32(words[feat * 4096:feat * 4096 + 4096])
        VideoFeatues = np.vstack((VideoFeatues, feat_row1)) if len(VideoFeatues) > 0 else feat_row1
    AllFeatures = VideoFeatues

    return  AllFeatures


def auc(model, iteration, aton_iteration, pred_gap):

    scores = []
    for i, (video, target) in enumerate(files_test):
        inputs      = load_dataset_One_Video_Features(video)                                     # 32 segment features for one testing video
        predictions = model.predict_on_batch(inputs)                                             # Get anomaly prediction for each of 32 video segments.
        preds_video = np.concatenate(([ [preds[0]]*pred_gap for preds in predictions]), axis=0)  # Reshape 32 segments to 480 frames
        scores      = np.concatenate((scores,preds_video), axis=0) if len(scores) > 0 else preds_video

    AUC = roc_auc_score(np.array(notes_test), np.array(scores))
    fpr, tpr, thresholds = roc_curve(np.array(notes_test), np.array(scores))

    savemat(os.path.join('results/weak/VAL', str(aton_iteration), 'training_AUC_'+str(iteration)+'.mat'), {'AUC': AUC, 'X': fpr, 'Y': tpr, 'scores': scores, 'gt':notes_test})

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


def train(path_loader, min_iterations, aton_iteration, val_file, val_notes, batchsize, tot_segments, pred_gap, save_best):
    global epoch, batch_obj, segmt_obj

    read_file_loader(path_loader,0)                                             # Load train set
    read_file_loader(val_file, 1)                                               # Load val set
    read_annotation(val_notes, 1)                                               # Load val annotation

    epoch     = tf.Variable(0,            dtype=tf.float32)                     # Variable to control the weights in the loss function
    batch_obj = tf.Variable(batchsize,    dtype=tf.int16)                       # Variable of the loss function
    segmt_obj = tf.Variable(tot_segments, dtype=tf.int16)                       # Variable of the loss function
    model     = create_model()
    adagrad   = Adagrad(lr=0.01, epsilon=1e-08)
    model.compile(loss=custom_objective, optimizer=adagrad)

    Results_Path      = os.path.join('results/weak/VAL', str(aton_iteration))   # Directory to save val stats in training
    output_dir        = os.path.join('models/weak_model', str(aton_iteration))  # Directory to save models and checkpoints
    model_path        = os.path.join(output_dir, 'model.json')
    best_model_path   = os.path.join(output_dir, 'best_model.json')
    best_weights_path = os.path.join(output_dir, 'best_weights.mat')

    if not os.path.exists(Results_Path):
        os.makedirs(Results_Path)

    if not os.path.exists(output_dir):
           os.makedirs(output_dir)

    abnormalPath     = [file[0] for file in files_train if int(file[1]) == 1]
    normalPath       = [file[0] for file in files_train if int(file[1]) == 0]
    loss_graph       = []
    full_batch_loss  = []
    total_iterations = 0
    bestAUC          = 0
    previousAUC      = [0]

    plt.ion()

    print('Training Weak Classifier...')

    prog = pyprog.ProgressBar("", " AUC - " + str(round(previousAUC[-1], 4))+ '%', total=min_iterations, bar_length=50, complete_symbol="=", not_complete_symbol=" ", wrap_bar_prefix=" [", wrap_bar_suffix="] ", progress_explain="", progress_loc=pyprog.ProgressBar.PROGRESS_LOC_END)
    prog.update()

    while total_iterations != min_iterations:
        inputs, targets = load_dataset_Train_batch(abnormalPath, normalPath, batchsize, tot_segments)  # Load normal and abnormal bags with fixed temporal segments of C3D features
        batch_loss = model.train_on_batch(inputs, targets)

        full_batch_loss.append(float(batch_loss))
        statistics.stats_batch(full_batch_loss, aton_iteration)

        loss_graph = np.hstack((loss_graph, batch_loss))
        if total_iterations % 20 == 0:
            iteration_path = output_dir + 'Iterations_graph_' + str(total_iterations) + '.mat'
            savemat(iteration_path, dict(loss_graph=loss_graph))                        # Loss checkpoint

            previousAUC.append(auc(model, total_iterations, aton_iteration, pred_gap))  # Validation results

            if previousAUC[-1] > bestAUC and save_best:                                 # Best model checkpoint
                bestAUC = previousAUC[-1]
                save_model(model, best_model_path, best_weights_path)

            weights_path = output_dir + 'weightsWeak_' + str(total_iterations) + '.mat'
            save_model(model, model_path, weights_path)                                 # Model checkpoint

        prog.set_suffix(" AUC - " + str(round(previousAUC[-1], 4)) + '% | Best AUC - ' + str(round(bestAUC, 4)) + '%')
        total_iterations += 1
        epoch.assign_add(tf.Variable(1,dtype=tf.float32))                               # Update loss variable
        prog.set_stat(total_iterations)
        prog.update()

    prog.end()

    plt.ioff()
    save_model(model, best_model_path, best_weights_path) if not save_best else None  # Save last as best if the best was not kept


def test(test_val_flag, aton_iteration, test_file, test_notes, pred_gap):
    results = 'FINAL' if test_val_flag else 'VAL'

    read_file_loader(test_file,1)                                          # Load test/val set
    read_annotation(test_notes, 1)                                         # Load test/val annotations

    Results_Path = os.path.join('results/weak', results, str(aton_iteration))
    model_dir    = os.path.join('models/weak_model', str(aton_iteration))  # Directory to the models
    weights_path = os.path.join(model_dir, 'best_weights.mat')
    model_path   = os.path.join(model_dir, 'best_model.json')              # Path to trained model

    if not os.path.exists(Results_Path):
        os.makedirs(Results_Path)

    model = load_model(model_path)
    load_weights(model, weights_path)


    print('Testing Weak Classifier...')

    prog = pyprog.ProgressBar("", "", total=len(files_test), bar_length=50, complete_symbol="=", not_complete_symbol=" ", wrap_bar_prefix=" [", wrap_bar_suffix="] ", progress_explain="", progress_loc=pyprog.ProgressBar.PROGRESS_LOC_END)
    prog.update()

    scores = []
    for i, (video, target) in enumerate(files_test):
        inputs      = load_dataset_One_Video_Features(video)                                     # 32 segment features for one testing video
        predictions = model.predict_on_batch(inputs)                                             # Get anomaly prediction for each of 32 video segments.
        preds_video = np.concatenate(([ [preds[0]]*pred_gap for preds in predictions]), axis=0)  # Reshape 32 segments to 480 frames
        scores      = np.concatenate((scores,preds_video), axis=0) if len(scores) > 0 else preds_video

        prog.set_stat(i+1)
        prog.update()

    prog.end()
    AUC = roc_auc_score(np.array(notes_test), np.array(scores))
    fpr, tpr, thresholds = roc_curve(np.array(notes_test), np.array(scores))
    print(AUC)
    savemat(os.path.join(Results_Path, 'eval_AUC_'+str(aton_iteration)+'.mat'), {'AUC': AUC, 'X': fpr, 'Y': tpr, 'scores': scores, 'gt': notes_test})

    return np.array(notes_test), np.array(scores)


def execute_free(aton_iteration, free_file, features, pred_gap):

    read_file_loader(free_file, 0)

    model_dir    = os.path.join('models/weak_model', str(aton_iteration))  # Model_dir is the folder where we have placed our trained weights
    weights_path = os.path.join(model_dir, 'best_weights.mat')
    model_path   = os.path.join(model_dir, 'best_model.json')              # Path to trained model

    model = load_model(model_path)
    load_weights(model, weights_path)

    print('Executing Weak Model Free Dataset...')

    prog = pyprog.ProgressBar("", "", total=len(files_train), bar_length=50, complete_symbol="=", not_complete_symbol=" ", wrap_bar_prefix=" [", wrap_bar_suffix="] ", progress_explain="", progress_loc=pyprog.ProgressBar.PROGRESS_LOC_END)
    prog.update()


    scores = []
    for i, video in enumerate(files_train):
        inputs        = load_dataset_One_Video_Features(video[0])                                  # 32 segments features for one testing video
        predictions   = model.predict_on_batch(inputs)                                             # Get anomaly prediction for each of 32 video segments.
        preds_video   = np.concatenate(([ [preds[0]]*pred_gap for preds in predictions]), axis=0)  # Reshape 32 segments to 480 frames
        reshape_preds = preds_video.reshape(-1, features).mean(axis=1)                             # Reshape 480 to 30, minimum multiple common between 30 and 32 é 15
        scores        = np.concatenate((scores,reshape_preds), axis=0) if len(scores) > 0 else reshape_preds

        prog.set_stat(i+1)
        prog.update()
    prog.end()
    return scores


notes_train, notes_test, files_train, files_test = [], [], [], []










