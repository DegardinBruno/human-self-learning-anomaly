import os, re, shutil
import moviepy.editor as mp
import argparse


def humanSort(text):  # Sort function for strings w/ numbers
    convText = lambda seq: int(seq) if seq.isdigit() else seq.lower()
    arrayKey = lambda key: [convText(s) for s in re.split('([0-9]+)', key)]  # Split numbers and chars, base function for sorted
    return sorted(text, key=arrayKey)


def normDur(dir_path, dest_path, fileName, avg_fps, fps, erase_frames):  # Normalize duration of video, partitioning the video in smaller videos

    print('Normalizing video duration...')
    size = int((len(os.listdir(dir_path)))/avg_fps)                                              # Get duration video to normalize after
    print(size)
    if size >= 1:
        countImage = 0                                                                           # Counter for sub videos frames

        for i in range(size):                                                                    # Create directories for sub videos
            if not os.path.exists(os.path.join(dir_path,str(i))):
                os.makedirs(os.path.join(dir_path,str(i)))

        for i in range((len([x[2] for x in os.walk(dir_path)][0]))):                             # Partitioning all frames into sub videos
            seqPath    = str(int(i/avg_fps))                                                     # Get respective sequence path of current frame

            if int(seqPath) >= size: break
            os.rename(os.path.join(dir_path,fileName + "_" + str(i) + ".png"),
                      os.path.join(dir_path,seqPath,fileName + "_" + str(countImage) + ".png"))  # Move frames to respective sequence directory

            countImage = countImage + 1 if countImage < avg_fps - 1 else 0                       # Update counter

        for extra in [file for file in os.listdir(dir_path) if os.path.isfile(dir_path+'/'+file)]:
            os.remove(os.path.join(dir_path, extra))                                             # Remove extra frames

        for i in range(size):                                                                    # Create all sub videos of each sequence path
            imgs = [os.path.join(dest_path,str(i), img) for img in humanSort(os.listdir(os.path.join(dest_path, str(i)))) if '.png' in img]  # Get list of full path of each images in each sub seq folder

            clips = [mp.ImageClip(m).set_duration(1 / fps-0.0000001) for m in imgs]              # Create ImageClip with duration per image

            concat_clip = mp.concatenate_videoclips(clips,  method="compose")
            concat_clip.write_videofile(os.path.join(dest_path, fileName + "_" + str(i) + '.mp4'), fps=fps)
            shutil.rmtree(os.path.join(dest_path, str(i))) if erase_frames else None

        print('Successfully normalized!')
    else:
        too_short.append(dir_path)



def read_paths(root_frames, root_dest, avg_fps, fps, erase_frames):
    for folder in humanSort(os.listdir(root_frames)):
        path_type_frames = os.path.join(root_frames, folder)
        path_dest        = os.path.join(root_dest, folder)

        for file in humanSort(os.listdir(path_type_frames)):
            dir_path  = os.path.join(path_type_frames, file)
            dest_path = os.path.join(path_dest, file)
            normDur(dir_path, dest_path, file, avg_fps, fps, erase_frames)

    print(too_short)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Paths and settings
    parser.add_argument('--root_frames',     type=str, default='PATH_TO_FRAMES',          help='root_frames -> directory_to_type_video -> full_video -> frames')
    parser.add_argument('--root_sub_videos', type=str, default='DEST_PATH_TO_SUB_VIDEOS', help='Destination path to sub videos')
    parser.add_argument('--duration',        type=int, default=16,                        help='Seconds, recommended to be multiple of 8, for consistency')
    parser.add_argument('--fps',             type=int, default=30,                        help='FPS to normalize (RECOMMENDED 25~30 FOR C3D!!)')
    parser.add_argument('--erase_frames',    action='store_true',                         help='Erase directories with frames')

    opt = parser.parse_args()

    avg_fps = opt.fps * opt.duration

    read_paths(opt.root_frames, opt.root_sub_videos, avg_fps, opt.fps, opt.erase_frames)


too_short = []

