import weak_classifier as WC
import strong_classifier as SC
import bayesian_classifier as BC
import statistics
import argparse, os


def test(model_weak, model_strong, iteration, flag_test, path_test, path_test_note, pred_gap, num_features):

    results        = 'FINAL' if flag_test  else 'VAL'
    select_model   = 'weak'  if model_weak else 'strong'

    ######## Model Evaluation and Statistics ########
    notes, scores = WC.test(flag_test, iteration, path_test, path_test_note, pred_gap) if model_weak else SC.test(flag_test, iteration, path_test, path_test_note, num_features)
    statistics.plot_AUC(os.path.join(select_model,results), iteration)
    BC.histogram(notes, scores, os.path.join(select_model,results), iteration)
    _, _, _, _ = BC.gaussian_kde(notes, scores, select_model, iteration) if not flag_test else None
    ####################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument('--path_test',      type=str, default='annotation/weak/test.csv',       help='Test csv file for the WS Model')
    parser.add_argument('--path_test_note', type=str, default='annotation/weak/test_notes.csv', help='Test annotation csv file for the WS Model')

    # Video settings
    parser.add_argument('--num_frame',     type=int, default=480, help='Fixed number of frames of each video')
    parser.add_argument('--features',      type=int, default=16,  help='Number of concatenated frames per feature extracted, ex: C3D use 16 frames')
    parser.add_argument('--temp_segments', type=int, default=32,  help='Total temporal segments per video for the WS Model')

    # Frameworks settings
    parser.add_argument('--model_iteration', type=int, default=0,            help='Model iteration to evaluate')
    parser.add_argument('--weak_model',      action='store_true',            help='Evaluate WS model')
    parser.add_argument('--strong_model',    action='store_true',            help='Evaluate SS model')
    parser.add_argument('--val',             action='store_true',            help='Evaluation on validation instead of test set')
    parser.add_argument('--SS_norm_file',    type=str, default='minmax.csv', help='File containing the minimum values of each column feature in the first row, and maximum values in second row for the SS model')


    opt = parser.parse_args()
    SC.load_norm(opt.SS_norm_file)

    features_video = int(opt.num_frame/opt.features)       # How many C3D Feature files in one video
    pred_gap       = int(opt.num_frame/opt.temp_segments)  # Reshape number of predictions to number of frames

    test(opt.weak_model, opt.strong_model, opt.model_iteration, not opt.val, opt.path_test, opt.path_test_note, pred_gap, opt.features)
