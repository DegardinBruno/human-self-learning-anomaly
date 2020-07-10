import os, re, struct
import numpy as np
import argparse


def read_binary_blob(fn):
    fid = open(fn, 'rb')
    t     = struct.unpack('i', fid.read(4))[0]  # unpack byte (4 bits) from fn file for topology
    c     = struct.unpack('i', fid.read(4))[0]
    l     = struct.unpack('i', fid.read(4))[0]
    h     = struct.unpack('i', fid.read(4))[0]
    w     = struct.unpack('i', fid.read(4))[0]
    total = t * c * l * h * w

    ret = np.zeros(total, np.float32)
    for i in range(total):
        ret[i] = struct.unpack('f', fid.read(4))[0]  # Extract byte to byte the main info

    fid.close()
    return ret


def humanSort(text):  # Sort function for strings w/ numbers
    convText = lambda seq: int(seq) if seq.isdigit() else seq.lower()
    arrayKey = lambda key: [convText(s) for s in re.split('([0-9]+)', key)]  # Split numbers and chars, base function for sorted
    return sorted(text, key=arrayKey)


def convert(video, path_video, size, temp_segments, dest_video, sufix):
    all_feat = [os.path.join(path_video, feat) for feat in humanSort(os.listdir(path_video))]
    feat_vect = np.zeros((len(all_feat), size))

    for i, feat in enumerate(all_feat):
        data = read_binary_blob(feat)
        feat_vect[i, :] = data

    seg_feat = np.zeros((temp_segments, size))
    shots_32 = np.round(np.linspace(1, len(all_feat), num=temp_segments + 1)) - 1

    for i in range(len(shots_32) - 1):
        ss = int(shots_32[i])
        ee = int(shots_32[i + 1] - 1)

        temp_vect = feat_vect[ss, :] if ss == ee else feat_vect[ss, :] if ee < ss else np.mean(feat_vect[ss:ee + 1, :], axis=1)
        temp_vect = temp_vect / np.linalg.norm(temp_vect)
        seg_feat[i, :] = temp_vect

    dest_path = os.path.join(dest_video, video + sufix)
    np.savetxt(dest_path, seg_feat, delimiter=' ', fmt='%.6f')


def C3D_to_fix_temp(root_C3D, dest_SEG, sufix, size, temp_segments, flag_sub):
    for folder in humanSort(os.listdir(root_C3D)):
        path_folder = os.path.join(root_C3D, folder)
        dest_folder = os.path.join(dest_SEG, folder)

        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        for video in humanSort(os.listdir(path_folder)):
            path_video = os.path.join(path_folder, video)
            print(path_video)

            if flag_sub:
                dest_video = os.path.join(dest_folder, video)

                if not os.path.exists(dest_video):
                    os.makedirs(dest_video)

                for seq in humanSort(os.listdir(path_video)):
                    path_seq  = os.path.join(path_video, seq)
                    convert(seq, path_seq, size, temp_segments, dest_video, sufix)

            else:
                convert(video, path_video, size, temp_segments, dest_folder, sufix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths and settings
    parser.add_argument('--root_C3D',        type=str, default='PATH_TO_RAW_C3D_FEATURES',           help='root_C3D -> directory_to_type_video -> full_video -> sub_video -> features')
    parser.add_argument('--dest_SEG',        type=str, default='DEST_PATH_TO_C3D_TEMPORAL_SEGMENTS', help='Destination path to temporal segments')
    parser.add_argument('--sufix',           type=str, default='_C.txt',                             help='Sufix of the temporal segment files')
    parser.add_argument('--size_descriptor', type=int, default=4096,                                 help='Size of the feature descriptor, default: fc6 layer of C3D')
    parser.add_argument('--temp_segments',   type=int, default=32,                                   help='Total temporal segments per video')
    parser.add_argument('--sub_videos',      action='store_true',                                    help='Convert for sub sequence videos, otherwise convert to the original videos')

    opt = parser.parse_args()

    C3D_to_fix_temp(opt.root_C3D, opt.dest_SEG, opt.sufix, opt.size_descriptor, opt.temp_segments, opt.sub_videos)
    
