import os, argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths and settings
    parser.add_argument('--root_video',    type=str, default='PATH_TO_VIDEO',             help='Absolute path to your video')
    parser.add_argument('--root_frames',   type=str, default='PATH_TO_FRAMES',            help='Path to frames (rel/abs)')
    parser.add_argument('--root_C3D_dir',  type=str, default='PATH_TO_C3D_DIRECTORY',     help='Absolute path to the directory where C3Dv1.0 is')
    parser.add_argument('--root_features', type=str, default='PATH_TO_RAW_FEATURES',      help='Absolute path of the directory where to save the features')
    parser.add_argument('--csv_C3D',       type=str, default='demo.csv',                  help='Path to the csv file to be created for the dataloader')
    parser.add_argument('--model_dir',     type=str, default='pretrained',                help='Directory where the model and weights are')
    parser.add_argument('--norm_file',     type=str, default='minmax.csv',                help='Path normalization file which the model was trained with')
    parser.add_argument('--output',        type=str, default='output.mp4',                help='Output video name')

    opt = parser.parse_args()


    os.system("sh inference.sh -a " + str(opt.root_video) +    " -b " + str(opt.root_frames) + " -c " + str(opt.root_C3D_dir) +
                             " -d " + str(opt.root_features) + " -e " + str(opt.csv_C3D) +     " -f " + str(opt.model_dir)+ " -g "+str(opt.norm_file))

