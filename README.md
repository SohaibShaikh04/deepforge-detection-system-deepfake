# Deepfake Video Detection System

This project aims to detect deepfake videos using a hybrid deep learning approach combining Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). The model extracts spatial features from video frames and captures temporal patterns across sequences to identify manipulations.

## Features

- CNN for spatial feature extraction
- RNN (GRU) for temporal sequence analysis
- K-Fold Cross Validation for performance robustness
- Achieves 96% accuracy with 5-fold testing for 5 epochs
- User-friendly UI for video upload and prediction


## How It Works

1. Upload a video via the interface.
2. The video is split into frames.
3. CNN extracts features from each frame.
4. RNN processes the sequence of features.
5. The model outputs the probability of the video being deepfake.


## Future Scope

1. Improve detection of advanced deepfake types like lip-syncing and face morphing
2. Add audio-visual inconsistency checks for better accuracy
3. Optimize the model for faster, real-time detection
4. Expand to work on different platforms and adapt to new deepfake techniques
