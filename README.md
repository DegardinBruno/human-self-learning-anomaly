# Human Activity Analysis: Iterative Weak/Self-Supervised Learning Frameworks for Detecting Abnormal Events

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/iterative-weak-self-supervised-classification/semi-supervised-anomaly-detection-on-ubi)](https://paperswithcode.com/sota/semi-supervised-anomaly-detection-on-ubi?p=iterative-weak-self-supervised-classification)

This repository contains the code for the IJCB'20 paper ["Human Activity Analysis: Iterative Weak/Self-Supervised Learning Frameworks for Detecting Abnormal Events"](https://ieeexplore.ieee.org/document/9304905)

<div align="center">
    <img src="data/WSS_Schema.png", width=100%">
</div>

## Installation
```bash
pip install -r requirements.txt  # use flag --user if permission needed
```

##### Libraries version:
* Keras: 2.3.1
* TensorFlow: 2.2.0
* Scipy: 1.4.1
* Scikit-learn: 0.23.1
* NumPy: 1.19.0

## Preprocessing Dataset << highly recommended
In order to employ the WS/SS framework and apply self-supervision to your dataset, normalization is required since we are working with coupled deep learning networks working at different levels. Follow the steps inside the [utils](utils) folder to successfully normalize your dataset.

## Dataset and Directories Structure

The [**UBI-Fights Dataset**](http://socia-lab.di.ubi.pt/EventDetection) provides a wide diversity in fighting scenarios with 1000 videos, where 784 are normal daily life situations and 216 contain a fighting event. This dataset is **fully-annotated** at the **frame-level**.<br>
To train your model and employ self-supervision for each network, the following directory structure is created:

```
|-- **annotation
|   |-- strong                  // Insert here your .csv files to be used by the SS Model
|   |   |-- train.csv           // empty, to be filled by the WS Model and Bayesian Classifier
|   |   |-- test.csv
|   |   |-- val.csv
|   |   |-- unlabeled_set.csv   // Same remaining videos unlabeled from the weak, but in C3D raw segment format
|   |   |-- test_notes.csv
|   |   `-- val_notes.csv
|   `-- weak                    // Insert here your .csv files to be used by the WS Model
|       |-- train.csv           // Small percentage (i.e., 30%) of the original training set
|       |-- test.csv
|       |-- val.csv
|       |-- unlabeled_set.csv   // Remaining videos are unlabeled for self-supervision purposes
|       |-- test_notes.csv
|       `-- val_notes.csv
|-- models
|   |-- pattern_model
|   |   |-- 0                   // WSS framework iterations
|   |   |-- 1
|   |   |-- ...
|   |-- strong_model
|   |   |-- 0                   // WSS framework iterations
|   |   |-- 1
|   |   |-- ...
|   `-- weak_model
|       |-- 0                   // WSS framework iterations
|       |-- 1
|       |-- ...
`-- results         
    |-- pattern
    |   `-- VAL                 // Directory for validation stats
    |-- strong
    |   |-- FINAL               // Directory for test stats
    |   `-- VAL
    `-- weak
        |-- FINAL
        `-- VAL
    
```


## Quick Start
- **Training**: Start framework training.
```bash
python3 ws_ss.py --save_best_weak --save_best_strong    # Set flag settings accordingly to the ws_ss.py file
```

- **Restart from checkpoint**: To analyse the unlabeled scores, after a network executed over the unlabeled set, select the previous iteration and network.
```bash
python3 ws_ss.py --start_iteration 0 --weak_free_checkpoint --save_best_weak --save_best_strong    # Set flag settings accordingly to the ws_ss.py file
```

- **Testing**: To evaluate the WS or SS model in the specific iteration over the testing data or validation data.
```bash
python3 test.py --strong_model --model_iteration 0 --path_test annotation/strong/test.csv --path_test_note annotation/strong/test_notes.csv   # Example to evaluate the SS model at iteration 0 of the WS/SS framework in the testing set
```

- **Inference**: Visualization of the response scores from a video. Install [C3D](https://github.com/facebookarchive/C3D) first and download their C3D model.
```bash
python3 inference.py --root_video abs/path/video --root_frames path/save/frames --root_C3D_dir abs/path/C3D --root_features abs/path/save/features  # Example to obtain a video visualizing the response scores of your model
```

## Citation
Please cite this paper in your publications if it helps your research:

    @inproceedings{degardin2020human,
      title={Human Activity Analysis: Iterative Weak/Self-Supervised Learning Frameworks for Detecting Abnormal Events},
      author={Degardin, Bruno and Proen{\c{c}}a, Hugo},
      booktitle={2020 IEEE International Joint Conference on Biometrics (IJCB)},
      pages={1--7},
      organization={IEEE}
    }

    
