import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from ultralytics import YOLO


def create_spectrogram(signal, sample_rate, output_path):
    """
    Generate and save a spectrogram image for a given signal.
    Args:
        signal (np.ndarray): The input signal.
        sample_rate (int): The sample rate of the signal.
        output_path (str): Path to save the spectrogram image.
    """
    # Calculate the spectrogram
    f, t, Sxx = spectrogram(signal, fs=sample_rate, nperseg=128, noverlap=64)

    # Convert to decibel scale
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # Plot and save the spectrogram
    plt.figure(figsize=(4, 4))
    plt.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


class Model:
    def __init__(self):
        self.clf = None

    def predict(self, input_array, standarize = True):
        """
        Predict if the same anomaly is present in both signals for multiple measurements.
        Args:
            input_array (np.ndarray): A numpy array of shape (x, 200, 2),
                                      where each entry represents two signals to analyze.

        Returns:
            np.ndarray: A numpy array of shape (x,) with predictions for each measurement,
                        where each value is in the range [0, 1].
        """
        if len(input_array.shape) != 3 or input_array.shape[1:] != (200, 2):
            raise ValueError(f"Expected input shape (x, 200, 2), got {input_array.shape}.")

        if self.clf is None:
            raise ValueError("Model is not loaded. Call `load()` first.")

        if standarize:
            # standarize input data
            stds = np.std(input_array, axis=-1)[:, :, np.newaxis]
            input_array = input_array/stds
            #input_array = np.swapaxes(background, 1, 2)

        sample_rate = 4096
        predictions = []
        for measurement_idx, signals in enumerate(input_array):
            # Extract the two signals
            signal_1, signal_2 = signals[:, 0], signals[:, 1]

            # Create spectrograms for the two signals
            spectrogram_paths = []
            for i, signal in enumerate([signal_1, signal_2]):
                 output_path = f"temp_spectrogram_{measurement_idx}_{i}.png"
                 create_spectrogram(signal, sample_rate, output_path)
                 spectrogram_paths.append(output_path)

            # YOLO predictions for each spectrogram
            preds = self.clf.predict(spectrogram_paths, verbose=False)

            # Clean up temporary spectrogram files
            for path in spectrogram_paths:
                if os.path.exists(path):
                    os.remove(path)

            # Aggregate probabilities for this measurement
            class_ = 1 # model.clf.names -> {0: 'background', 1: 'bbh'}
            probabilities = [float(pred.probs.data[class_]) for pred in preds]
            combined_prob = min(probabilities)  # Use the minimum probability as the combined score

            predictions.append(combined_prob)

        return predictions

    def load(self):
        """
        Load the pre-trained YOLO model.
        """
        #parent_dir_abspath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        #model_path = os.path.join(parent_dir_abspath, "train7/weights/best.pt")

        dir_abspath = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.join(dir_abspath, "best.pt")
        self.clf = YOLO(model_path)
        print(f"model path: {model_path}")

