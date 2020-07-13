from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.models import model_from_json
from scipy.io import loadmat, savemat
import numpy as np
import matplotlib.pyplot as plt
import pyprog, argparse, csv, struct, os, re
import matplotlib.animation as animation


def humanSort(text):  # Sort function for strings w/ numbers
    convText = lambda seq: int(seq) if seq.isdigit() else seq.lower()
    arrayKey = lambda key: [convText(s) for s in re.split('([0-9]+)', key)]  # Split numbers and chars, base function for sorted
    return sorted(text, key=arrayKey)


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
    def __init__(self):
        self.samples = [sample[0] for i, sample in enumerate(files_test)]

    def __getitem__(self, index):  # When accessing to element in dataset
        path        = self.samples[index]
        sample      = read_binary_blob(path)  # read binary file from path in sample
        sample_norm = []
        for i, value in enumerate(sample):
            sample_norm.append( ((value - minmax[0][i]) / (minmax[1][i] - minmax[0][i])) if minmax[1][i] > 0 else 0 )  # Normalize sample to previous calculated minmax on training set

        return sample_norm

    def __len__(self):
        return len(self.samples)


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
    i = 0
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


def read_file_loader(path):
    global files_train, files_test
    files_test  = []
    with open(path, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            files_test.append(row)


def execute(path_loader, model_dir, frames_per_feature, output):

    read_file_loader(path_loader)

    test_dataset = Dataset()  # Create Test dataset
    weights_path = os.path.join(model_dir, 'best_weights.mat')
    model_path   = os.path.join(model_dir, 'best_model.json')  # Path to trained model

    model = load_model(model_path)
    load_weights(model, weights_path)

    scores = []

    print('Predicting...')

    prog = pyprog.ProgressBar("", "", total=len(test_dataset), bar_length=50, complete_symbol="=", not_complete_symbol=" ", wrap_bar_prefix=" [", wrap_bar_suffix="] ", progress_explain="", progress_loc=pyprog.ProgressBar.PROGRESS_LOC_END)
    prog.update()

    for iv in range(len(test_dataset)):
        inputs = [test_dataset[iv]]
        predictions = model.predict_on_batch(np.array(inputs))  # Get anomaly prediction for each of 32 video segments.
        preds_video = np.concatenate(([[preds[0]] * frames_per_feature for preds in predictions]), axis=0)
        scores = np.concatenate((scores,preds_video), axis=0) if len(scores) > 0 else preds_video

        prog.set_stat(iv)
        prog.update()

    prog.end()

    savemat(output[:-4] + '.mat',  {'scores': np.array(scores)})
    return np.array(scores)


def demo(path_C3D, path_frames, model_dir, frames_per_feature, fps, output, norm_file):
    global minmax

    with open(norm_file, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            minmax.append(np.array(row).astype(float))

    result = execute(path_C3D, model_dir, frames_per_feature, output)

    left, width        = 0.15, 0.75
    height_1, height_2 = 0.5, 0.35
    bottom_1, bottom_2 = 0.13 + height_2, 0.1

    rect_1 = [left, bottom_1, width, height_1]
    rect_2 = [left, bottom_2, width, height_2]

    fig  = plt.figure()
    axs0 = plt.axes(rect_1)
    axs0.set_yticks([])
    axs0.set_xticks([])

    axs1 = plt.axes(rect_2)
    axs1.set_ylim(-0.05,1.05)
    axs1.grid(True)
    axs1.set_ylabel('Prediction')
    axs1.set_xlabel('Frame')

    impng         = []
    show_result   = []
    y             = []
    last_variance = []


    print('Plotting demo...')
    prog = pyprog.ProgressBar("", "", total=len(result), bar_length=50, complete_symbol="=", not_complete_symbol=" ",
                              wrap_bar_prefix=" [", wrap_bar_suffix="] ", progress_explain="",
                              progress_loc=pyprog.ProgressBar.PROGRESS_LOC_END)
    prog.update()

    frames = [os.path.join(path_frames,frame) for frame in humanSort(os.listdir(path_frames))]

    for i in range(len(result)):

        im1    = axs0.imshow(plt.imread(frames[i]), animated=True,aspect='auto')

        variance = [im1]

        if i%16 == 0:
            show_result.append(result[i])
            y.append(i)
            last_variance = []

            tmp, = axs1.plot(y, show_result, color='black', linewidth=2, alpha=0.7)
            last_variance.append(tmp)

        impng.append(variance+last_variance)

        prog.set_stat(i)
        prog.update()

    prog.end()


    print('Writing video demo...')

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=4000)  # Set bitrate higher if needed
    ani1   = animation.ArtistAnimation(fig, impng, interval=fps)
    ani1.save(output, writer=writer)

    plt.show()


files_test, minmax = [], []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths and settings
    parser.add_argument('--csv_C3D',            type=str, default='demo.csv',                  help='Path to the csv file to be created for the dataloader')
    parser.add_argument('--root_frames',        type=str, default='PATH_TO_FRAMES',            help='Path to frames (rel/abs)')
    parser.add_argument('--model_dir',          type=str, default='PATH_TO_MODEL_AND_WEIGHTS', help='Directory where the model and weights are')
    parser.add_argument('--frames_per_feature', type=int, default=16,                          help='Frames per feature, default: C3D (16)')
    parser.add_argument('--fps',                type=int, default=30,                          help='FPS')
    parser.add_argument('--norm_file',          type=str, default='minmax.csv',                help='Path normalization file which the model was trained with')
    parser.add_argument('--output',             type=str, default='output.mp4',                help='Output video name')

    opt = parser.parse_args()

    demo(opt.csv_C3D, opt.root_frames, opt.model_dir, opt.frames_per_feature, opt.fps, opt.output, opt.norm_file)





