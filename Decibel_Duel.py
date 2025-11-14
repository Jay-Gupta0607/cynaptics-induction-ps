import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split

#Constants
SAMPLE_RATE = 22050
CLIP_DURATION = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)

#Spectrogram parameters
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

BATCH_SIZE = 32
NUM_EPOCHS = 200
TEST_SIZE = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

CATEGORIES = []

def load_data(data_dir):
    audio_paths = []
    labels = []
    category_to_index = {}
    index_counter = 0

    for root, dirs, files in os.walk(data_dir):
        category_name = os.path.basename(root)

        if category_name != os.path.basename(data_dir) and category_name:
            if category_name not in category_to_index:
                category_to_index[category_name] = index_counter
                index_counter += 1

            label_index = category_to_index[category_name]

            for filename in files:
                if filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    full_path = os.path.join(root, filename)
                    audio_paths.append(full_path)
                    labels.append(label_index)

    categories = sorted(category_to_index, key=category_to_index.get)

    if not audio_paths:
        print("Warning: No audio files found. Check folder path and extensions.")

    return audio_paths, labels, categories

def load_test_files(data_dir):
    audio_paths = []
    print(f"Scanning test data in: {data_dir}")
    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                full_path = os.path.join(root, filename)
                audio_paths.append(full_path)
    print(f"Found {len(audio_paths)} test files.")
    return sorted(audio_paths)


class AudioDataset(Dataset):
    def __init__(self, audio_paths, labels, target_sample_rate=SAMPLE_RATE, num_samples=NUM_SAMPLES, n_mels=N_MELS):
        self.audio_paths = audio_paths
        self.labels = labels
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.n_mels = n_mels

    def _pad_or_truncate_librosa(self, waveform):
        return librosa.util.fix_length(
            data=waveform,
            size=self.num_samples,
            mode='constant'
        )

    def _transform_to_spectrogram(self, waveform):
        melspec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.target_sample_rate,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=self.n_mels
        )
        db_melspec = librosa.power_to_db(melspec, ref=np.max)
        return db_melspec

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        label = self.labels[index]

        try:
            waveform, sr = librosa.load(
                audio_path,
                sr=self.target_sample_rate,
                mono=True
            )

            waveform = self._pad_or_truncate_librosa(waveform)
            spectrogram_np = self._transform_to_spectrogram(waveform)

            spectrogram = torch.tensor(spectrogram_np, dtype=torch.float32).unsqueeze(0)
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            return spectrogram, label_tensor

        except Exception as e:
            print(f"Skipping file {audio_path} due to error (Librosa/NumPy): {e}")
            return None

class TestAudioDataset(Dataset):
    def __init__(self, audio_paths, target_sample_rate=SAMPLE_RATE, num_samples=NUM_SAMPLES, n_mels=N_MELS):
        self.audio_paths = audio_paths
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.n_mels = n_mels

    def _pad_or_truncate_librosa(self, waveform):
        return librosa.util.fix_length(
            data=waveform,
            size=self.num_samples,
            mode='constant'
        )

    def _transform_to_spectrogram(self, waveform):
        melspec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.target_sample_rate,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=self.n_mels
        )
        db_melspec = librosa.power_to_db(melspec, ref=np.max)
        return db_melspec

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        audio_path = self.audio_paths[index]

        try:
            waveform, sr = librosa.load(
                audio_path,
                sr=self.target_sample_rate,
                mono=True
            )

            waveform = self._pad_or_truncate_librosa(waveform)
            spectrogram_np = self._transform_to_spectrogram(waveform)

            spectrogram = torch.tensor(spectrogram_np, dtype=torch.float32).unsqueeze(0)
            
            return spectrogram, audio_path

        except Exception as e:
            print(f"Skipping file {audio_path} due to error (Librosa/NumPy): {e}")
            return None

class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return logits

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.empty(0, 1, N_MELS, 0), torch.empty(0, dtype=torch.long)
    return torch.utils.data.dataloader.default_collate(batch)

def collate_fn_test(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.empty(0, 1, N_MELS, 0), []

    spectrograms, paths = zip(*batch)
    
    spectrograms_stacked = torch.stack(spectrograms)
    
    return spectrograms_stacked, list(paths)


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        if inputs.size(0) == 0:
            continue
            
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            if inputs.size(0) == 0:
                continue

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    val_loss = running_loss / (total_samples if total_samples > 0 else 1)
    val_accuracy = correct_predictions / (total_samples if total_samples > 0 else 1)
    return val_loss, val_accuracy

def generate_test_predictions(model_path, test_data_dir, output_csv_path, num_classes, categories):
    model = CNNClassifier(num_classes=num_classes).to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Cannot run testing.")
        return
    model.eval()
    print(f"Loaded model from {model_path}")

    test_audio_paths = load_test_files(test_data_dir)
    if not test_audio_paths:
        print("No test audio files found. Aborting prediction.")
        return

    test_dataset = TestAudioDataset(test_audio_paths)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn_test
    )

    results = []
    with torch.no_grad():
        for inputs, paths in test_loader:
            if inputs.size(0) == 0:
                continue

            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            
            _, predicted_indices = torch.max(outputs.data, 1)
            
            for i in range(len(paths)):
                file_path = paths[i]
                filename = os.path.basename(file_path)
                predicted_index = predicted_indices[i].item()
                predicted_class = categories[predicted_index]
                results.append((filename, predicted_class))
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Class'])
            writer.writerows(results)
        print(f"Successfully saved {len(results)} predictions to {output_csv_path}")
    except Exception as e:
        print(f"Error writing to CSV {output_csv_path}: {e}")

def main(data_folder_path, model_save_path):
    global CATEGORIES

    all_audio_paths, all_labels, dynamic_categories = load_data(data_folder_path)

    if not all_audio_paths:
        print("Training aborted: No audio files found.")
        return 0.0, []

    CATEGORIES = dynamic_categories
    NUM_CLASSES = len(CATEGORIES)
    print(f"\n--- Found {len(all_audio_paths)} files across {NUM_CLASSES} classes: {CATEGORIES} ---\n")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_audio_paths, all_labels, test_size=TEST_SIZE, random_state=42, stratify=all_labels
    )
    print(f"Train samples: {len(train_paths)}, Validation samples: {len(val_labels)}")

    train_dataset = AudioDataset(train_paths, train_labels)
    val_dataset = AudioDataset(val_paths, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = CNNClassifier(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    best_accuracy = 0.0
    print("Starting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_model(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, DEVICE)

        print(f"Epoch {epoch}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print("  -> Model saved (New best accuracy).")

    print("\nTraining complete.")
    print(f"Best Validation Accuracy achieved: {best_accuracy:.4f}")
    
    return best_accuracy, CATEGORIES


if __name__ == '__main__':
    
    TRAIN_DATA_PATH = "the-frequency-quest/train"
    TEST_DATA_PATH = "the-frequency-quest/test"
    
    MODEL_SAVE_PATH = 'best_audio_classifier.pth'
    OUTPUT_CSV_PATH = 'predictions.csv'

    print("--- STARTING TRAINING PHASE ---")
    best_val_accuracy, trained_categories = main(
        data_folder_path=TRAIN_DATA_PATH,
        model_save_path=MODEL_SAVE_PATH
    )

    if not trained_categories:
        print("Training failed or found no data. Exiting.")
    else:
        print(f"\nTraining complete. Best Validation Acc: {best_val_accuracy:.4f}")
        
        print("\n--- STARTING TESTING PHASE ---")
        generate_test_predictions(
            model_path=MODEL_SAVE_PATH,
            test_data_dir=TEST_DATA_PATH,
            output_csv_path=OUTPUT_CSV_PATH,
            num_classes=len(trained_categories),
            categories=trained_categories
        )
        
        print("\nScript finished.")