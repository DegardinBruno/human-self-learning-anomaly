import os, re, csv
import argparse


def humanSort(text):  # Sort function for strings w/ numbers
    convText = lambda seq: int(seq) if seq.isdigit() else seq.lower()
    arrayKey = lambda key: [convText(s) for s in re.split('([0-9]+)', key)]  # Split numbers and chars, base function for sorted
    return sorted(text, key=arrayKey)


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


def normalize(root_csv, dest_csv, num_frames):

    for csv_file in humanSort(os.listdir(root_csv)):
        path_csv_full = os.path.join(root_csv,csv_file)
        frame_full = read_file(path_csv_full)

        print(len(frame_full))
        frame_seq = [frame_full[i: i + num_frames] for i in range(0, len(frame_full), num_frames)]
        print(len(frame_seq))

        csv_solo = csv_file[:-4]
        for i, seq in enumerate(frame_seq): write_file(os.path.join(dest_csv,csv_solo+'_'+str(i)+'.csv'), seq)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths and settings
    parser.add_argument('--root_csv',    type=str, default='PATH_TO_CSVs',     help='Root path of your csvs')
    parser.add_argument('--dest_csv',    type=str, default='PATH_TO_SUB_CSVs', help='Destination path to the normalized csv')
    parser.add_argument('--fps',         type=int, default=30,                 help='FPS of your sub videos')
    parser.add_argument('--duration',    type=int, default=16,                 help='Duration in seconds of your sub videos')

    opt = parser.parse_args()

    num_frames = opt.fps * opt.duration

    normalize(opt.root_videos, opt.root_conv_videos, num_frames)

