import csv, numpy as np
import os
from scipy.io import loadmat, savemat
import weak_classifier as WC
import strong_classifier as SC
import bayesian_classifier as BC
import pattern_classifier as PC
import statistics
import argparse


def read_file(source):
    result = []
    with open(source, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            result.append(row)
    return result


def write_file(source, content):
    with open(source, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(content)


def chosen_from_weak(new_neg_videos, new_pos_videos):  # Annotation from the WS model
    strong_free      = read_file(opt.path_strong_free)
    new_strong_train = read_file(opt.path_strong_train)
    new_selected     = []
    new_free         = []

    repeated         = [i for i, x in enumerate(new_pos_videos) if x in new_neg_videos]
    new_pos_videos   = np.delete(new_pos_videos, repeated)

    # Annotating instances to the SS model
    for i, video in enumerate(strong_free):
        if i in new_neg_videos:
            new_strong_train.append([video[0],0])
            new_selected.append([video[0],0])
        elif i in new_pos_videos:
            new_strong_train.append([video[0],1])
            new_selected.append([video[0],1])
        else:
            new_free.append(video)

    write_file(opt.path_strong_train, new_strong_train)
    write_file(opt.path_strong_selected, new_selected)


def chosen_from_strong(new_neg_videos, new_pos_videos):  # Annotation from the SS model
    weak_free        = read_file(opt.path_weak_free)
    new_weak_train   = read_file(opt.path_weak_train)
    new_strong_train = np.array(read_file(opt.path_strong_train))
    strong_free      = np.array(read_file(opt.path_strong_free))
    new_selected     = []
    new_weak_free    = []
    selected_0       = np.array([])
    selected_1       = np.array([])

    # Annotating instances to the WS model and removing them from the unlabeled set
    for i, video in enumerate(weak_free):
        instances = strong_free[np.arange(i * features_video, i * features_video + features_video, 1)]
        if i in new_neg_videos:
            new_weak_train.append([video[0], 0])
            new_selected.append([video[0],0])
            selected_0 = np.concatenate((selected_0, instances)) if len(selected_0) > 0 else instances
        elif i in new_pos_videos:
            new_weak_train.append([video[0], 1])
            new_selected.append([video[0],1])
            selected_1 = np.concatenate((selected_1, instances)) if len(selected_1) > 0 else instances
        else:
            new_weak_free.append(video)

    # Annotating negative videos to the SS model
    selected_0       = selected_0.reshape((len(selected_0), 1))
    Y0               = np.zeros((len(selected_0), 1), dtype=int)
    selected_0       = np.hstack((selected_0, Y0))
    new_strong_train = np.concatenate((new_strong_train, selected_0)) if len(new_strong_train) > 0 else selected_0

    # Positive videos to be strongly-annotated by the SS model
    selected_1 = selected_1.reshape((len(selected_1),1))

    # Remove labeled instances from the unlabeled set
    delete_strong_free = []
    for i in new_neg_videos:
        delete_strong_free = np.concatenate((delete_strong_free, np.arange(i*features_video,i*features_video+features_video,1)))

    for i in new_pos_videos:
        delete_strong_free = np.concatenate((delete_strong_free, np.arange(i*features_video,i*features_video+features_video,1)))

    strong_free   = np.delete(strong_free, delete_strong_free)
    strong_free   = np.array(strong_free).reshape((len(strong_free),1))

    write_file(opt.path_weak_train, new_weak_train)
    write_file(opt.path_weak_free, new_weak_free)
    write_file(opt.path_weak_selected, new_selected)
    write_file(opt.path_strong_predict, selected_1)
    write_file(opt.path_strong_free, strong_free)
    write_file(opt.path_strong_train, new_strong_train)


def annotate_predict(annotation):  # Annotation of positive videos from the SS model
    new_strong_train = np.array(read_file(opt.path_strong_train))
    new_strong_predict = np.array(read_file(opt.path_strong_predict))

    for i, video in enumerate(new_strong_predict): np.concatenate((new_strong_train, [[video[0],annotation[i]]]))

    write_file(opt.path_strong_train, new_strong_train)


def save_free_scores(source, WSS_iter, free_scores, probs):  # Backup scores
    savemat(os.path.join('results', source, 'VAL', str(WSS_iter), 'free_scores_' + str(WSS_iter) + '.mat'), {'scores': free_scores, 'probs': probs})


def best_model(model, iteration):  # Select the best model based on the validation set
    all_auc = np.array([loadmat(os.path.join('results', model, 'VAL', str(i), 'eval_AUC_'+str(i)+'.mat'))['AUC'][0] for i in range(iteration+1)])
    return np.where(all_auc==max(all_auc))[0][0]


def checkpoint_val(model, iteration):
    file = loadmat(os.path.join('results', model ,'VAL', str(iteration), 'eval_AUC_'+str(iteration)+'.mat'))
    return np.array(file['gt']), np.array(file['scores'][0])


def checkpoint_load(model, iteration):
    file = loadmat(os.path.join('results', model ,'VAL', str(iteration), 'free_scores_'+str(iteration)+'.mat'))
    return np.array(file['scores'][0]), np.array(file['probs'])


def WS_SS(iterations):
    current_iteration = opt.start_iteration
    while current_iteration < iterations:

        ##########################
        #        WS  MODEL       #
        ##########################
        if not opt.weak_free_checkpoint and not opt.strong_free_checkpoint:

            ######## WS Model Training, Evaluation and Statistics ########
            WC.train(opt.path_weak_train, opt.WS_iterations, current_iteration, opt.path_weak_val, opt.path_weak_val_note, opt.batchsize_weak, opt.temp_segments, pred_gap, opt.save_best_weak)

            weak_notes_test, weak_scores_test = WC.test(True, current_iteration, opt.path_weak_test, opt.path_weak_test_note, pred_gap)
            statistics.plot_AUC('weak/FINAL', current_iteration)
            BC.histogram(weak_notes_test, weak_scores_test, 'weak/FINAL', current_iteration)

            notes_val, scores_val = WC.test(False, current_iteration, opt.path_weak_val, opt.path_weak_val_note, pred_gap)
            statistics.plot_AUC('weak/VAL', current_iteration)
            BC.histogram(notes_val, scores_val, 'weak/VAL', current_iteration)
            classes, probs, max_neg_threshold, max_pos_threshold = BC.gaussian_kde(notes_val, scores_val, 'weak', current_iteration)
            ##############################################################


            ######### Best WS Model Execution on Unlabeled Data ##########
            best_iteration        = best_model('weak', current_iteration)
            notes_val, scores_val = checkpoint_val('weak', best_iteration)
            free_scores           = WC.execute_free(best_iteration, opt.path_weak_free, opt.features, pred_gap)  # Execute WS model on unlabeled set

            classes, probs, max_neg_threshold, max_pos_threshold = BC.gaussian_kde(notes_val, scores_val, 'weak', best_iteration, X_test=free_scores)  # Estimate likelihoods
            save_free_scores('weak', current_iteration, free_scores, probs)  # Backup Scores
            ##############################################################

        if opt.weak_free_checkpoint:
            best_iteration     = best_model('weak', current_iteration)
            free_scores, probs = checkpoint_load('weak', best_iteration)
            max_neg_threshold  = (max(probs[:, 0]) // 0.001 / 1000)
            max_pos_threshold  = (max(probs[:, 1]) // 0.001 / 1000)


        if not opt.strong_free_checkpoint:

            ############### Unlabeled scores filtering ###################
            new_neg_videos = np.array([i for i, prob in enumerate(probs) if (prob[0] >= max_neg_threshold-opt.ll_tolerance) and free_scores[i]<opt.WS_neg_threshold])
            new_pos_videos = np.array([i for i, prob in enumerate(probs) if (prob[1] >= max_pos_threshold-opt.ll_tolerance) and free_scores[i]>opt.WS_pos_threshold])


            neg_size = (len(new_neg_videos)*100)/len(probs)
            pos_size = (len(new_pos_videos)*100)/len(probs)

            if neg_size > pos_size*opt.normal_factor:
                oversize       = neg_size-(pos_size*opt.normal_factor)
                rem_length     = int((oversize*len(new_neg_videos))/neg_size)
                random_train   = np.random.permutation(len(new_neg_videos))
                new_neg_videos = new_neg_videos[random_train[rem_length:]]

            neg_size = (len(new_neg_videos)*100)/len(probs)

            print('New negative videos from strong free - ' + str(neg_size) + '%')
            print('New positive videos from strong free - ' + str(pos_size) + '%')

            chosen_from_weak(new_neg_videos, new_pos_videos)
            ##############################################################

        ##########################
        #        SS  MODEL       #
        ##########################
        if not opt.strong_free_checkpoint:

            ######## SS Model Training, Evaluation and Statistics ########
            SC.train(opt.path_strong_train, opt.SS_iterations, current_iteration, opt.path_strong_val, opt.path_strong_val_note, opt.num_features, opt.batchsize_strong, opt.save_best_strong)
            strong_notes_test, strong_scores_test = SC.test(True, current_iteration, opt.path_strong_test, opt.path_strong_test_note, opt.num_features)
            statistics.plot_AUC('strong/FINAL', current_iteration)
            BC.histogram(strong_notes_test, strong_scores_test, 'strong/FINAL', current_iteration)

            strong_notes_val, strong_scores_val = SC.test(False, current_iteration, opt.path_strong_val, opt.path_strong_val_note, opt.num_features)
            statistics.plot_AUC('strong/VAL', current_iteration)
            BC.histogram(strong_notes_val, strong_scores_val, 'strong/VAL', current_iteration)
            classes, probs, max_neg_threshold, max_pos_threshold = BC.gaussian_kde(strong_notes_val, strong_scores_val, 'strong', current_iteration)
            ##############################################################


            ######### Best SS Model Execution on Unlabeled Data ##########
            best_iteration                              = best_model('strong', current_iteration)
            strong_notes_pattern, strong_scores_pattern = checkpoint_val('strong', best_iteration)                           # Load train set for Pattern Classfier with results from val set
            PC.train(strong_notes_pattern, strong_scores_pattern, current_iteration)                                         # Pattern Classifier training with validation scores and GT
            classes, probs = PC.test(strong_scores_pattern, True, current_iteration, y_test=strong_notes_pattern)            # Keep stats

            strong_notes_pattern, strong_scores_pattern = SC.predict_pattern(1, best_iteration, opt.path_strong_free, None, opt.num_features)  # Execute SS model on unlabeled set
            save_free_scores('strong', current_iteration, strong_scores_pattern, [1,2,3])                                    # Backup scores
            classes, probs = PC.test(strong_scores_pattern, False, current_iteration)                                        # Execute Pattern Classifier over unlabeled scores
            save_free_scores('strong', current_iteration, strong_scores_pattern, probs)                                      # Backup scores
            ##############################################################

        if opt.strong_free_checkpoint:
            best_iteration               = best_model('strong', current_iteration)
            strong_scores_pattern, probs = checkpoint_load('strong', best_iteration)
            max_neg_threshold            = (max(probs[:, 0]) // 0.001 / 1000)
            max_pos_threshold            = (max(probs[:, 1]) // 0.001 / 1000)

        ############### Unlabeled scores filtering ###################
        free_stack     = np.array([strong_scores_pattern[i: i+features_video] for i in range(0, len(strong_scores_pattern), features_video)])  # Stack scores per video
        sum_free_stack = np.array([sum(stack) for stack in free_stack])

        new_neg_videos = np.array([i for i, prob in enumerate(probs) if prob[0] <= opt.PC_neg_prob  and sum_free_stack[i] < opt.PC_neg_threshold])
        new_pos_videos = np.array([i for i, prob in enumerate(probs) if prob[0] >= opt.PC_pos_prob  and sum_free_stack[i] > opt.PC_pos_threshold])


        neg_size = (len(new_neg_videos)*100)/len(probs)
        pos_size = (len(new_pos_videos)*100)/len(probs)

        if neg_size > (pos_size*opt.normal_factor):
            oversize       = neg_size-(pos_size*opt.normal_factor)
            rem_length     = int((oversize*len(new_neg_videos))/neg_size)
            random_train   = np.random.permutation(len(new_neg_videos))
            new_neg_videos = new_neg_videos[random_train[rem_length:]]

        neg_size = (len(new_neg_videos)*100)/len(probs)

        print('New negative videos from weak free - ' + str(neg_size) + '%')
        print('New positive videos from weak free - ' + str(pos_size) + '%')

        chosen_from_strong(new_neg_videos, new_pos_videos)
        ##################################################################

        ############# Annotate abnormal videos with SS model #############
        if pos_size > 0:
            strong_notes_free, strong_scores_free                = SC.predict_pattern(2, best_iteration, opt.path_strong_predict, None, opt.num_features)
            classes, probs, max_neg_threshold, max_pos_threshold = BC.gaussian_kde(strong_notes_val, strong_scores_val, 'strong', best_iteration, X_test=strong_scores_free)  # Estimate likelihoods

            new_annotation = np.array(['1' if prob[np.where(classes == 1)] >= max_pos_threshold-opt.ll_tolerance and strong_scores_free[i] > opt.SS_pos_threshold else '0' for i, prob in enumerate(probs)])
            annotate_predict(new_annotation)
        ##################################################################

        current_iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('--path_weak_train',       type=str, default='annotation/weak/train.csv',        help='Train csv file for the WS Model')
    parser.add_argument('--path_weak_val',         type=str, default='annotation/weak/val.csv',          help='Val csv file for the WS Model')
    parser.add_argument('--path_weak_test',        type=str, default='annotation/weak/test.csv',         help='Test csv file for the WS Model')
    parser.add_argument('--path_weak_free',        type=str, default='annotation/weak/free.csv',         help='Unlabeled set csv file for the WS Model')
    parser.add_argument('--path_weak_selected',    type=str, default='annotation/weak/selected.csv',     help='Selected samples csv file from the SS Model to the WS Model in current iteration')
    parser.add_argument('--path_weak_val_note',    type=str, default='annotation/weak/val_notes.csv',    help='Val annotation csv file for the WS Model')
    parser.add_argument('--path_weak_test_note',   type=str, default='annotation/weak/test_notes.csv',   help='Test annotation csv file for the WS Model')
    parser.add_argument('--path_strong_train',     type=str, default='annotation/strong/train.csv',      help='Train csv file for the SS Model')
    parser.add_argument('--path_strong_predict',   type=str, default='annotation/strong/predict.csv',    help='predict csv file for the SS Model, to be annotated for the WS Model')
    parser.add_argument('--path_strong_val',       type=str, default='annotation/strong/val.csv',        help='Val csv file for the SS Model')
    parser.add_argument('--path_strong_test',      type=str, default='annotation/strong/test.csv',       help='Test csv file for the SS Model')
    parser.add_argument('--path_strong_free',      type=str, default='annotation/strong/free.csv',       help='Unlabeled set csv file for the SS Model')
    parser.add_argument('--path_strong_selected',  type=str, default='annotation/strong/selected.csv',   help='Selected samples csv file from the WS Model to the SS Model in current iteration')
    parser.add_argument('--path_strong_val_note',  type=str, default='annotation/strong/val_notes.csv',  help='Val annotation csv file for the SS Model')
    parser.add_argument('--path_strong_test_note', type=str, default='annotation/strong/test_notes.csv', help='Test annotation csv file for the SS Model')

    # Video settings
    parser.add_argument('--num_frame',     type=int, default=480, help='Fixed number of frames of each video')
    parser.add_argument('--features',      type=int, default=16,  help='Number of concatenated frames per feature extracted, ex: C3D use 16 frames')
    parser.add_argument('--temp_segments', type=int, default=32,  help='Total temporal segments per video for the WS Model')

    # Frameworks settings
    parser.add_argument('--start_iteration',        type=int, default=0, help='First iteration number for the WS/SS framework')
    parser.add_argument('--weak_free_checkpoint',   action='store_true', help='Restart system after WS execution on the unsupervised set')
    parser.add_argument('--strong_free_checkpoint', action='store_true', help='Restart system after SS execution on the unsupervised set')
    parser.add_argument('--save_best_weak',         action='store_true', help='Save best WS model based on validation')
    parser.add_argument('--save_best_strong',       action='store_true', help='Save best SS model based on validation')

    # Networks settings
    parser.add_argument('--WS_iterations',        type=int, default=2000,           help='Training iterations for the WS Model')
    parser.add_argument('--SS_iterations',        type=int, default=500,            help='Training iterations for the SS Model')
    parser.add_argument('--WSS_iterations',       type=int, default=50,             help='Total iterations of the framework')
    parser.add_argument('--batchsize_weak',       type=int, default=60,             help='Batch size for the WS Model')
    parser.add_argument('--batchsize_strong',     type=int, default=60,             help='Batch size for the SS Model')
    parser.add_argument('--SS_norm_file',         type=str, default='minmax.csv',   help='File containing the minimum values of each column feature in the first row, and maximum values in second row')

    # Self-annotation settings
    parser.add_argument('--normal_factor',    type=int,   default=2,     help='Limitation of selected normal instances, to avoid extreme inconsistency balance with abnormal instances')
    parser.add_argument('--ll_tolerance',     type=float, default=0.05,  help='Likelihood tolerance for each class relatively to the maximum')
    parser.add_argument('--WS_neg_threshold', type=float, default=0.005, help='Maximum score allowed for negative instances to be chosen by the WS model')
    parser.add_argument('--WS_pos_threshold', type=float, default=0.95,  help='Minimum score allowed for positive instances " "       " "       " "')
    parser.add_argument('--PC_neg_threshold', type=float, default=1.0,   help='Maximum summation score allowed for negative videos to be chosen by the Pattern Classifier')
    parser.add_argument('--PC_pos_threshold', type=float, default=20.0,  help='Minimum summation score allowed for positive " "       " "       " "       " "       " "')
    parser.add_argument('--PC_neg_prob',      type=float, default=0.1,   help='Maximum likelihood allowed for negative videos to be chosen by the Pattern Classifier')
    parser.add_argument('--PC_pos_prob',      type=float, default=0.9,   help='Minimum likelihood allowed for positive " "       " "       " "       " "       " "')
    parser.add_argument('--SS_pos_threshold', type=float, default=0.9,   help='Minimum score to be annotate as positive instance by the SS model')

    opt = parser.parse_args()

    # Directories structure creation

    weak_dir = opt.path_weak_train[:opt.path_weak_train.rfind('/')]
    if not os.path.exists(weak_dir):
        os.makedirs(weak_dir)

    strong_dir = opt.path_strong_train[:opt.path_strong_train.rfind('/')]
    if not os.path.exists(strong_dir):
        os.makedirs(strong_dir)

    weak_dir = os.path.join('models', 'weak_model')
    if not os.path.exists(weak_dir):
        os.makedirs(weak_dir)

    strong_dir = os.path.join('models', 'strong_model')
    if not os.path.exists(strong_dir):
        os.makedirs(strong_dir)

    pattern_dir = os.path.join('models', 'pattern_model')
    if not os.path.exists(pattern_dir):
        os.makedirs(pattern_dir)

    weak_dir = os.path.join('results', 'weak', 'FINAL')
    if not os.path.exists(weak_dir):
        os.makedirs(weak_dir)

    weak_dir = os.path.join('results', 'weak', 'VAL')
    if not os.path.exists(weak_dir):
        os.makedirs(weak_dir)

    weak_dir = os.path.join('results', 'strong', 'FINAL')
    if not os.path.exists(weak_dir):
        os.makedirs(weak_dir)

    weak_dir = os.path.join('results', 'strong', 'VAL')
    if not os.path.exists(weak_dir):
        os.makedirs(weak_dir)

    weak_dir = os.path.join('results', 'pattern', 'VAL')
    if not os.path.exists(weak_dir):
        os.makedirs(weak_dir)


    SC.load_norm(opt.SS_norm_file)

    features_video = opt.num_frame/opt.features       # How many C3D Feature files in one video
    pred_gap       = opt.num_frame/opt.temp_segments  # Reshape number of predictions to number of frames

    WS_SS(opt.WSS_iterations)
