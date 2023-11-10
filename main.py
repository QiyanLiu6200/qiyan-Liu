import os
import warnings

import dill
import librosa
import librosa.feature
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from xgboost import XGBClassifier

warnings.simplefilter("ignore")  # Ignore warnings
warnings.filterwarnings("ignore")  # Ignore warnings
#plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # Used to normally display Chinese labels
plt.rcParams["axes.unicode_minus"] = False  # Used to normally display the negative sign

IDX2LABELS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}  # Dictionary mapping numbers to emotions in the RAVDESS dataset
LABELS = list(IDX2LABELS.values())  # Emotion labels


def save_pkl(filepath, data):
    # Save the model
    with open(filepath, "wb") as fw:
        dill.dump(data, fw)
    print(f"[{filepath}] data saving...")


def load_pkl(filepath):
    # Load the model
    with open(filepath, "rb") as fr:
        data = dill.load(fr, encoding="utf-8")
    print(f"[{filepath}] data loading...")
    return data


def save_txt(filepath, data):
    # Save text
    with open(filepath, "w", encoding="utf-8") as fw:
        fw.write(data)
    print(f"{filepath} saving...")


def standard_scaler(values, scaler_path, mode="train"):
    # Standardize data
    if mode == "train":
        scaler = StandardScaler()  # Define standardization model
        scaler.fit(values)  # Train the model
        save_pkl(scaler_path, scaler)  # Save the model
    else:
        scaler = load_pkl(scaler_path)  # Load the model
    return scaler.transform(values)  # Transform the data


def save_evaluate(y_test, y_test_pred, outputs_path):
    # Save performance metrics and confusion matrix
    test_report = classification_report(
        y_test,
        y_test_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        digits=4,
    )  # Calculate performance metrics for test set including precision/recall/f1-score/accuracy
    test_matrix = confusion_matrix(y_test, y_test_pred)  # Calculate test set confusion matrix

    results = "test classification report:\n" + test_report + "\nconfusion matrix\n" + str(test_matrix)  # Concatenate performance metrics and confusion matrix
    save_txt(outputs_path, results)  # Save performance metrics and confusion matrix
    print(results)  # Print performance metrics and confusion matrix


def plot_confusion_matrix(y_test, y_test_pred, output_path):
    # Plot confusion matrix
    matrix = confusion_matrix(y_test, y_test_pred)  # Calculate confusion matrix
    matrix = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]  # Normalize

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 4.8), dpi=100)  # Define canvas
    sns.heatmap(matrix, annot=True, fmt=".2f", linewidths=0.5, square=True, cmap="Blues", ax=ax)  # Plot heatmap
    ax.set_title("Confusion Matrix Visualization")  # Title
    ax.set_xlabel("True Labels")  # x-axis label
    ax.set_ylabel("Predicted Labels")  # y-axis label
    ax.set_xticks([x + 0.5 for x in range(len(LABELS))], LABELS, rotation=0)  # x-axis ticks
    ax.set_yticks([x + 0.5 for x in range(len(LABELS))], LABELS, rotation=0)  # y-axis ticks
    plt.savefig(output_path)  # Save image
    # plt.show()  # Display image
    plt.close()  # Close image


def plot_roc(y_test, y_test_pred_score, output_path):
    # Plot ROC curve
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.4, 4.8), dpi=100)  # Define canvas
    for label_idx, label_name in enumerate(LABELS):  # Iterate through all labels
        false_positive_rate, true_positive_rate, _ = roc_curve(
            [1 if y == label_idx else 0 for y in y_test],
            y_test_pred_score[:, label_idx],
        )  # Calculate ROC values
        roc_auc = auc(false_positive_rate, true_positive_rate)  # Calculate AUC
        ax.plot(false_positive_rate, true_positive_rate, label=f"{label_name}, AUC = {roc_auc:0.4f}")  # Plot line graph
    ax.plot([0, 1], [0, 1], "r--")  # Plot diagonal line
    ax.set_xlabel("False Positive Rate")  # x-axis label
    ax.set_ylabel("True Positive Rate")  # y-axis label
    ax.set_title("ROC curve based on RF model")  # Title
    plt.legend(loc="lower right")  # Show legend
    plt.savefig(output_path)  # Save image
    # plt.show()  # Display image
    plt.close()  # Close image


def build_datasets(voice_root_dir, data_path, outputs_dir):
    # Extract features from audio data and build datasets
    data = []  # Dataset
    for voice_dir_idx, voice_dir in enumerate(os.listdir(voice_root_dir)):  # Iterate through subdirectories in the root directory of the audio data
        for voice_path_idx, voice_path in enumerate(os.listdir(os.path.join(voice_root_dir, voice_dir))):  # Iterate through audio files in subdirectories
            print(f"Processing the {voice_path_idx + 1}th audio file in the {voice_dir_idx + 1}th subdirectory...")

            label = IDX2LABELS[voice_path.split(".")[0].split("-")[2]]  # Emotion label
            gender = 0 if int(voice_path.split(".")[0].split("-")[6]) & 1 == 0 else 1  # Gender  0: female  1: male

            # Read audio file
            voice, sr = librosa.load(os.path.join(voice_root_dir, voice_dir, voice_path), sr=None)  # sr=None keeps the original sampling rate

            # log-mel spectrogram feature extraction
            mel_spect = librosa.feature.melspectrogram(y=voice, sr=sr, n_fft=2048, hop_length=512, n_mels=128)  # Mel spectrogram
            log_mel_spect = librosa.power_to_db(mel_spect)  # Convert to dB
            log_mel_features = np.mean(log_mel_spect, axis=1)  # Calculate mean along the spectrum axis

            # MFCC feature extraction
            mfcc = librosa.feature.mfcc(y=voice, sr=sr, n_mfcc=40)  # Mel-frequency cepstral coefficients
            mfcc_features = np.mean(mfcc, axis=1)  # Calculate mean along the spectrum axis

            # Combine features
            sample = {"label": label, "gender": gender}
            sample.update({f"log_mel_{idx}": value for idx, value in enumerate(log_mel_features)})
            sample.update({f"mfcc_{idx}": value for idx, value in enumerate(mfcc_features)})
            data.append(sample)

            if voice_dir_idx == 0 and voice_path_idx == 0:
                librosa.display.waveshow(voice, sr=sr)  # Plot waveform
                # plt.show()  # Display image
                plt.savefig(os.path.join(outputs_dir, "voice.png"))  # Save image
                plt.close()  # Close image

                librosa.display.specshow(log_mel_spect, sr=sr, x_axis="time", y_axis="mel")  # Plot Mel spectrogram
                # plt.show()  # Display image
                plt.savefig(os.path.join(outputs_dir, "voice_log_mel.png"))  # Save image
                plt.close()  # Close image

                librosa.display.specshow(mfcc, sr=sr, x_axis="time")  # Plot Mel-frequency cepstral coefficients
                # plt.show()  # Display image
                plt.savefig(os.path.join(outputs_dir, "voice_mfcc.png"))  # Save image
                plt.close()  # Close image

    data = pd.DataFrame(data)  # Convert to DataFrame
    data.to_csv(data_path, index=False)  # Save dataset


if __name__ == "__main__":
    voice_root_dir = rf"voice-data"  # Root directory of audio data
    data_path = rf"data.csv"  # Dataset path
    outputs_dir = rf"outputs"  # Output directory
    os.makedirs(outputs_dir, exist_ok=True)  # Create output directory if it doesn't exist

    if not os.path.exists(data_path):  # If the dataset does not exist
        build_datasets(voice_root_dir, data_path, outputs_dir)  # Build the dataset

    data = pd.read_csv(data_path)  # Read the dataset
    X = data.drop(columns="label").values  # Features
    y = data["label"].apply(lambda x: LABELS.index(x)).values  # Labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split the dataset into training and test sets (8:2)
    X_train = standard_scaler(X_train, os.path.join(outputs_dir, "continuous_scaler.pkl"), mode="train")  # Standardize features for the training set
    X_test = standard_scaler(X_test, os.path.join(outputs_dir, "continuous_scaler.pkl"), mode="test")  # Standardize features for the test set

    model = XGBClassifier(n_jobs=-1, random_state=42)  # Define the model
    model.fit(X_train, y_train)  # Train the model
    save_pkl(os.path.join(outputs_dir, "model.pkl"), model)  # Save the model

    y_test_pred = model.predict(X_test)  # Predict on the test set
    save_evaluate(y_test, y_test_pred, os.path.join(outputs_dir, "evaluate.txt"))  # Save evaluation metrics
    plot_confusion_matrix(y_test, y_test_pred, os.path.join(outputs_dir, "confusion_matrix.png"))  # Plot confusion matrix

    y_test_pred_score = model.predict_proba(X_test)  # Calculate scores for the test set
    plot_roc(y_test, y_test_pred_score, os.path.join(outputs_dir, "roc.png"))  # Plot ROC curve
