## Preprocessing Dataset
In order to employ the WS/SS framework and apply self-supervision to your dataset, normalization is required since we are working with coupled deep learning networks working at different levels. Follow the next steps to successfully preprocessing your dataset.


1. **Video Conversion**: Convert videos to frames. Videos inside their type directory.
```bash
python3 video_frames.py --root_videos path/to/videos --root_conv_videos destination/path/conv/videos --root_frames destination/path/frames
```

2. **Normalize Durations**: Convert frames from the previous step to fixed number of sub videos. Also, normalize your .csv annotation files of your testing and validation sets to the corresponding duration of sub videos (Step 3).
```bash
python3 normalize_videos.py --root_frames path/to/frames --root_sub_videos destination/path/sub/videos --erase_frames  # Change settings if needed
```

3. **Normalize Annotations**: Convert .csv file annotations (testing sets and validation sets) accordingly to the sub videos from the previous step. Each row of the .csv file corresponds to one temporal frame.
```bash
python3 normalize_notes.py --root_csv path/to/csv/files --dest_csv destination/path/sub/csv/files --fps 30 --duration 16  # Change settings if needed
```

4. **Feature Extraction**: Extract the [C3Dv1.0](https://github.com/facebookarchive/C3D) features of your dataset, with the default settings. Only fc6 layer needed.

5. **Feature Conversion**: Convert C3D raw features from the previous step to fixed temporal segments. To employ the WS/SS framework, fixed length is needed, so add the flag ```--sub_videos``` if you already converted your videos to trimmed sub_videos (Step 2).
```bash
python3 normalize_C3D.py --root_C3D path/to/raw/features --dest_SEG destination/path/segments --sub_videos
```

6. **CSV File Format**: Each row in a .csv file should contain the temporal segment's path and corresponding weak annotation (0 if the video does not contain any anomaly, and 1 if the video contains an anomaly somewhere). Example:\[path_to_temporal_segment_file, weak_flag\], where weak_flag is the weak annotation.
