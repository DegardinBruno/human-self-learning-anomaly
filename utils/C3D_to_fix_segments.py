import os, re, struct
import numpy as np


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


def humanSort(text):  # Sort function for strings w/ numbers
    convText = lambda seq: int(seq) if seq.isdigit() else seq.lower()
    arrayKey = lambda key: [convText(s) for s in re.split('([0-9]+)', key)]  # Split numbers and chars, base function for sorted
    return sorted(text, key=arrayKey)


root_C3D   = 'PATH_TO_RAW_C3D_FEATURES'       # root_C3D -> directory_to_type_video -> full_video -> sub-video -> features
dest_seg   = 'PATH_TO_C3D_TEMPORAL_SEGMENTS'  # the folder with the sub-videos of each
sufix      = '_C.txt'

temporal_segments = 32

folders = [folder for folder in humanSort(os.listdir(root_C3D))]

for folder in humanSort(os.listdir(root_C3D)):
    path_folder = os.path.join(root_C3D, folder)

    for video in humanSort(os.listdir(path_folder)):
        path_video = os.path.join(path_folder, video)
        dest_video = os.path.join(dest_seg, folder, video)
        print(path_video)

        if not os.path.exists(dest_video):
            os.makedirs(dest_video)

        for seq in humanSort(os.listdir(path_video)):
            path_seq = os.path.join(path_video, seq)

            all_feat  = [os.path.join(path_seq, feat) for feat in humanSort(os.listdir(path_seq))]
            feat_vect = np.zeros((len(all_feat), 4096))

            for i, feat in enumerate(all_feat):
                data = read_binary_blob(feat)
                feat_vect[i,:] = data

            seg_feat = np.zeros((temporal_segments,4096))
            shots_32 = np.round(np.linspace(1,len(all_feat),num=temporal_segments+1))-1

            for i in range(len(shots_32)-1):
                ss = int(shots_32[i])
                ee = int(shots_32[i+1]-1)

                temp_vect = feat_vect[ss,:] if ss==ee else feat_vect[ss,:] if ee<ss else np.mean(feat_vect[ss:ee+1,:], axis=1)

                temp_vect = temp_vect/np.linalg.norm(temp_vect)

                seg_feat[i,:] = temp_vect


            dest_path = os.path.join(dest_video, seq+sufix)

            np.savetxt(dest_path, seg_feat, delimiter=' ', fmt='%.6f')