import os, cv2, re
import moviepy.editor as mp
import argparse


def humanSort(text):  # Sort function for strings w/ numbers
    convText = lambda seq: int(seq) if seq.isdigit() else seq.lower()
    arrayKey = lambda key: [convText(s) for s in re.split('([0-9]+)', key)]  # Split numbers and chars, base function for sorted
    return sorted(text, key=arrayKey)


def resize(file_path, convPath, width, height, fps):
    clip        = mp.VideoFileClip(file_path)
    newWidth    = (clip.size[0]/clip.size[1])*height                                                                                     # Formula of Aspect Ratio Calculator
    clipResized = clip.resize(width=width) if newWidth >= width else clip.resize(height=height)                                          # Resizing to desired W/H
    margin      = abs(clipResized.size[1] - height) / 2 if newWidth >= width else abs(clipResized.size[0] - width) / 2                   # Margin Calculator
    clipResized = clipResized.margin(color=(255, 255, 255), top=int(margin), bottom=int(margin) if margin % 1 == 0 else int(margin) + 1) if newWidth >= width else \
                  clipResized.margin(color=(255, 255, 255), left=int(margin),right=int(margin) if margin % 1 == 0 else int(margin) + 1)  # Set correct margins accordingly to the new width
    clipResized.write_videofile(convPath[:-4] + '_conv' + convPath[-4:], fps=fps)                                                        # Write converted video


def convert(file_path, conv_path, frame_path, file, width, height, fps):
    if not os.path.exists(conv_path):
        os.makedirs(conv_path)

    resize(file_path,conv_path, width, height, fps)                       # Resize video
    vidcap = cv2.VideoCapture(conv_path[:-4] + '_conv' + conv_path[-4:])  # Read Video converted
    totalFrames   = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    success,image = vidcap.read()
    count         = 0

    if not os.path.exists(frame_path):
        os.makedirs(frame_path)

    print(frame_path, totalFrames)
    while success:
      cv2.imwrite(os.path.join(frame_path, file)[:-4]+"_%d.png" % count, image)
      success,image = vidcap.read()
      print(str(success) + ' Read frame: '+ str(count)) if count%100 == 0 else None
      count += 1



def read_paths(root, root_conv, root_frames, width, height, fps, format, erase_conv):
    for folder in humanSort(os.listdir(root)):
        path = os.path.join(root,folder)
        path_conv = os.path.join(root_conv,folder)
        path_frames = os.path.join(root_frames,folder)
        videos = [video for video in humanSort(os.listdir(path)) if format in video]
        for video in videos:
            video_path = os.path.join(path, video)
            conv_path  = os.path.join(path_conv, video)
            frame_path = os.path.join(path_frames,video)
            convert(video_path, conv_path, frame_path, video, width, height, fps)
            if erase_conv:
                os.remove(conv_path[:-4] + '_conv' + conv_path[-4:])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths and settings
    parser.add_argument('--root_videos',      type=str, default='PATH_TO_TYPE_VIDEOS', help='root_videos -> directory_to_type_video -> video')
    parser.add_argument('--root_conv_videos', type=str, default='PATH_TO_CONV_VIDEOS', help='Destination path to the converted videos')
    parser.add_argument('--root_frames',      type=str, default='PATH_TO_FRAMES',      help='Destination path for frames')
    parser.add_argument('--width',            type=int, default=640,                   help='Width to normalize')
    parser.add_argument('--height',           type=int, default=360,                   help='Height to normalize')
    parser.add_argument('--fps',              type=int, default=30,                    help='FPS to normalize (RECOMMENDED 25~30 FOR C3D!!)')
    parser.add_argument('--format',           type=str, default='.mp4',                help='Video format')
    parser.add_argument('--erase_conv',       action='store_true',                     help='Erase the converted videos')

    opt = parser.parse_args()

    read_paths(opt.root_videos, opt.root_conv_videos, opt.root_frames, opt.width, opt.height, opt.fps, opt.format, opt.erase_conv)







