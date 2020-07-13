#!/bin/bash

while getopts ":a:b:c:d:e:f:g:" opt; do
  case $opt in
    a) root_video="$OPTARG"
    ;;
    b) root_frames="$OPTARG"
    ;;
    c) root_C3D_dir="$OPTARG"
    ;;
    d) root_features="$OPTARG"
    ;;
    e) csv_C3D="$OPTARG"
    ;;
    f) model_dir="$OPTARG"
    ;;
    g) norm_file="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done


python3 input_C3D.py --root_video "$root_video" --root_frames "$root_frames" --root_C3D_dir "$root_C3D_dir" --root_features "$root_features" --csv_C3D "$csv_C3D"

(cd "$root_C3D_dir"/C3D-v1.0/examples/c3d_feature_extraction/ && GLOG_logtosterr=1 ../../build/tools/extract_image_features.bin prototxt/c3d_sport1m_feature_extractor_video.prototxt conv3d_deepnetA_sport1m_iter_1900000 0 50 50000 prototxt/output_list_video_prefix.txt fc6-1)

python3 scores.py --csv_C3D "$csv_C3D" --root_frames "$root_frames" --model_dir "$model_dir" --norm_file "$norm_file"