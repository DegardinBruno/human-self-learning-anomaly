import os, re, cv2, argparse, csv
import moviepy.editor as mp


def humanSort(text):  # Sort function for strings w/ numbers
    convText = lambda seq: int(seq) if seq.isdigit() else seq.lower()
    arrayKey = lambda key: [convText(s) for s in re.split('([0-9]+)', key)]  # Split numbers and chars, base function for sorted
    return sorted(text, key=arrayKey)


def convert(file_path, frame_path):

    file = file_path[file_path.rfind('/')+1:]

    vidcap = cv2.VideoCapture(file_path)  # Read Video converted
    totalFrames   = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    success,image = vidcap.read()
    count         = 0

    if not os.path.exists(frame_path):
        os.makedirs(frame_path)

    print(frame_path, totalFrames)
    while success:
      cv2.imwrite(os.path.join(frame_path, file)[:-4]+"_%d.png" % count, image)
      success,image = vidcap.read()
      print(str(success) + ' writing frame: '+ str(count)) if count%100 == 0 else None
      count += 1


def read_video_C3D():

    convert(opt.root_video, opt.root_frames)

    input  = open(os.path.join(opt.root_C3D_dir,'C3D-v1.0/examples/c3d_feature_extraction/prototxt/input_list_video.txt'),'w')
    output = open(os.path.join(opt.root_C3D_dir,'C3D-v1.0/examples/c3d_feature_extraction/prototxt/output_list_video_prefix.txt'), 'w')

    vidcap      = cv2.VideoCapture(opt.root_video)  # Read Video converted
    totalFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))  # Count total number of frames
    total       = int(totalFrames / 16)

    csv_files = []
    for x in range(total):
        input.write(opt.root_video + ' ' + str(x*16) + ' ' + str(0) + '\n')
        output.write(os.path.join(opt.root_features, str(x*16).zfill(6) + '\n'))
        csv_files.append([os.path.join(opt.root_features, str(x*16).zfill(6) + '.fc6-1')])
        if not os.path.exists(opt.root_features):
            os.makedirs(opt.root_features)

    input.close()
    output.close()

    with open(opt.csv_C3D, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(csv_files)  # Write Annotation array


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths and settings
    parser.add_argument('--root_video',    type=str, default='PATH_TO_VIDEO',             help='Absolute path to your video')
    parser.add_argument('--root_features', type=str, default='PATH_TO_RAW_FEATURES',      help='Absolute path of the directory where to save the features')
    parser.add_argument('--root_C3D_dir',  type=str, default='PATH_TO_C3D_DIRECTORY',     help='Absolute path to the directory where C3Dv1.0 is')
    parser.add_argument('--root_frames',   type=str, default='PATH_TO_FRAMES',            help='Path to frames (rel/abs)')
    parser.add_argument('--csv_C3D',       type=str, default='demo.csv',                  help='Path to the csv file to be created for the dataloader')


    opt = parser.parse_args()

    read_video_C3D()

