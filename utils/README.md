## Preprocessing Dataset
In order to employ the WS/SS framework and apply self-supervision to your dataset, normalization is required since we are working with coupled deep learning networks working at different levels. Follow the next steps to successfully preprocessing your dataset.


1. **Video Conversion**: Convert videos to frames. Videos inside their type directory.
```bash
python3 video_frames.py --root_videos path/to/videos --root_conv_videos destination/path/conv/videos --root_frames destination/path/frames
```

2. **Normalize Durations**: Convert frames from the previous step to fixed number of sub videos. Also, normalize your .csv annotation files of your testing and validation sets to the corresponding duration of sub videos.
```bash
python3 normalize_videos.py --root_frames path/to/frames --root_sub_videos destination/path/sub/videos --erase_frames  # Change settings if needed
```

3. **Feature Extraction**: Extract the [C3Dv1.0](https://github.com/facebookarchive/C3D) features of your dataset, with the default settings. Only fc6 layer needed.

4. **Feature Conversion**: Convert C3D raw features from the previous step to fixed temporal segments. To employ the WS/SS framework, fixed length is needed, so add the flag ```--sub_videos``` if you already converted your videos to trimmed sub_videos (Step 2).
```bash
python3 normalize_C3D.py --root_C3D path/to/raw/features --dest_SEG destination/path/segments --sub_videos
```
