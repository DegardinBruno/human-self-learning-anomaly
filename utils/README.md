## Preprocessing Dataset

- **Feature Conversion**: Convert raw features from C3D to fixed temporal segments. To employ the WS/SS framework add the flag ```--sub_videos``` if you already converted your videos to trimmed sub_videos.
```bash
python3 C3D_to_fixed_seg.py --root_C3D path/to/raw/features --dest_SEG destination/path/segments --sub_videos
```
