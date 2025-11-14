PyTorch Audio Classification CNN
This repository contains a complete Python script for audio classification using a Convolutional Neural Network (CNN) built with PyTorch. The script handles everything from data loading and preprocessing to model training, validation, and generating predictions for a test set.

It uses Librosa to convert audio files into Mel spectrograms, which are then fed into the CNN for classification.

✨ Features
End-to-End Pipeline: Includes data loading, preprocessing, training, evaluation, and test prediction.

Mel Spectrograms: Uses librosa to transform raw audio into 2D Mel spectrograms, a standard and effective representation for audio tasks.

CNN Model: Implements a simple but effective 4-layer CNN designed for 2D image-like data (spectrograms).

Data Handling:

Automatically discovers classes from the training folder structure.

Uses torch.utils.data.Dataset and DataLoader for efficient batching.

Includes custom collate_fn to handle potential loading errors in a batch.

Best Model Saving: Automatically saves the model checkpoint with the highest validation accuracy.

Prediction Generation: Generates a predictions.csv file for the test dataset, formatted with ID and Class.

📂 Data Structure
The script expects a specific folder structure to locate the training and testing data.

Training Data
The script will automatically detect class_A, class_B, and class_C as the output categories.

Test Data
The test audio files should be in a separate folder.

🚀 Usage
1. Requirements
You can install the necessary Python packages using pip:

Bash

pip install torch numpy librosa scikit-learn
2. Configuration
All major parameters are defined as global constants at the top of the script. You can adjust these to tune performance or fit your dataset.

SAMPLE_RATE = 22050: The target sample rate for all audio files.

CLIP_DURATION = 4.0: The fixed length (in seconds) all clips will be padded or truncated to.

N_MELS = 128: The number of Mel-frequency bins to use for the spectrogram.

N_FFT = 1024: The size of the Fast Fourier Transform window.

HOP_LENGTH = 512: The number of samples between successive frames.

BATCH_SIZE = 32: The number of samples per batch during training.

NUM_EPOCHS = 200: The total number of training epochs.

TEST_SIZE = 0.2: The proportion of the training data to hold out for validation.

3. Run the Script
To run the entire pipeline, simply execute the Python script from your terminal:

Bash

python your_script_name.py
The script will:

Start Training: Load data from TRAIN_DATA_PATH, split it, and begin training for NUM_EPOCHS.

Log Progress: Print the training loss, validation loss, and validation accuracy for each epoch.

Save Model: Save the model weights to MODEL_SAVE_PATH (best_audio_classifier.pth) whenever a new best validation accuracy is achieved.

Start Testing: Once training is complete, it will load the saved best model.

Generate Predictions: Process all files in TEST_DATA_PATH and save the class predictions to OUTPUT_CSV_PATH (predictions.csv).

🔧 How It Works
Preprocessing Pipeline
Each audio file is processed through the following steps:

Load: Loaded using librosa.load, resampled to SAMPLE_RATE, and converted to mono.

Pad/Truncate: Standardized to a fixed length of NUM_SAMPLES (4.0 seconds) using librosa.util.fix_length. Shorter clips are padded with silence, and longer clips are truncated.

Mel Spectrogram: Transformed into a Mel spectrogram using librosa.feature.melspectrogram.

Log-Scale: Converted to decibels (dB) using librosa.power_to_db for a more perceptually relevant scale.

Tensor: Converted to a PyTorch tensor and a channel dimension is added, resulting in a shape of [1, N_MELS, TimeSteps].

Model Architecture (CNNClassifier)
The model is a 4-layer Convolutional Neural Network:

Input: A 1-channel (grayscale) Mel spectrogram of shape [Batch, 1, 128, ~173].

Four Convolutional Blocks:

Conv2d -> ReLU -> MaxPool2d -> Dropout

The channel depth increases with each block: 1 -> 16 -> 32 -> 64 -> 128.

Global Pooling: An nn.AdaptiveAvgPool2d(1) layer reduces the spatial dimensions of each feature map to 1x1, effectively summarizing the features.

Flatten: The output is flattened to [Batch, 128].

Fully Connected Head: A 2-layer classifier (nn.Linear(256, 512) -> ReLU -> Dropout -> nn.Linear(512, num_classes)) produces the final logits for classification.
