

## Preprocessing Dataset
In order to employ the WS/SS framework and apply self-supervision to your dataset, normalization is required since we are working with coupled deep learning networks working at different levels. Follow the next steps to successfully preprocessing your dataset.


1. **Video Conversion**: Convert videos to frames. Videos inside their type directory.
```bash
python3 video_frames.py --root_videos path/to/videos --root_conv_videos destination/path/conv/videos --root_frames destination/path/frames
```

2. **Normalize Durations**: Convert frames from the previous step to a fixed number of sub videos. Also, normalize your .csv annotation files of your testing and validation sets to the corresponding duration of sub videos (Step 3).
```bash
python3 normalize_videos.py --root_frames path/to/frames --root_sub_videos destination/path/sub/videos --erase_frames  # Change settings if needed
```

3. **Normalize Annotations**: Convert .csv file annotations (testing sets and validation sets) accordingly to the sub videos from the previous step. Each row of the .csv file corresponds to one temporal frame. After normalizing annotations concatenate each .csv video annotation file into one for testing and other for validation, corresponding to the *test_notes.csv* and *val_notes.csv* in the [Dataset and Directories Structures](https://github.com/DegardinBruno/human_activity_anomaly_IJCB20#dataset-and-directories-structure).
```bash
python3 normalize_notes.py --root_csv path/to/csv/files --dest_csv destination/path/sub/csv/files --fps 30 --duration 16  # Change settings if needed
```

4. **Feature Extraction**: Extract the [C3Dv1.0](https://github.com/facebookarchive/C3D) features of your dataset, with the default settings. Only the fc6 layer needed.

5. **Feature Conversion**: Convert C3D raw features from the previous step to fixed temporal segments. To employ the WS/SS framework, fixed length is needed, so add the flag ```--sub_videos``` if you already converted your videos to trimmed sub_videos (Step 2).
```bash
python3 normalize_C3D.py --root_C3D path/to/raw/features --dest_SEG destination/path/segments --sub_videos
```

6. **CSV File Format**: Each row in a .csv file in the [Dataset and Directories Structures](https://github.com/DegardinBruno/human_activity_anomaly_IJCB20#dataset-and-directories-structure) should contain the temporal segment's path and corresponding weak annotation (0 if the video does not contain any anomaly, and 1 if the video contains an anomaly somewhere). Example: ```[path_to_temporal_segment_file, weak_flag]```, where weak_flag is the weak annotation. For the SS model, the rows should contain the C3D raw segment from the videos and corresponding weak annotation for the test and validation .csv files. The unlabeled .csv files should only contain the path to the file in each row in the network's respective format. 

7. **IMPORTANT!!** The unlabeled datasets from the weak and strong folder inside the annotation directory should start synchronized with each other, which means every corresponding raw feature in the strong folder should be in the same order as the corresponding video in the weak one.
