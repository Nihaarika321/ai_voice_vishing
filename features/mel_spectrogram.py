"""
Mel Spectrogram Feature Extraction
==================================

Project:
    AI Voice Vishing Detection

Purpose:
    Extract log-Mel spectrogram features from short speech segments.

Input:
    Audio file or already-loaded audio waveform

Output:
    Log-Mel spectrogram

Configuration:
    Sample rate   : 16000 Hz
    Mel bands     : 64
    FFT size      : 512
    Hop length    : 160
    Window length : 400
    Duration      : 1 second
"""

import os

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000

N_MELS = 64

N_FFT = 512

HOP_LENGTH = 160

WIN_LENGTH = 400

DURATION = 1.0

TARGET_SAMPLES = int(
    SAMPLE_RATE * DURATION
)


# ============================================================
# LOAD AUDIO FROM FILE
# ============================================================

def load_audio(
    audio_path,
    sample_rate=SAMPLE_RATE
):
    """
    Load an audio file.

    Parameters
    ----------
    audio_path : str
        Path to audio file.

    sample_rate : int
        Target sampling rate.

    Returns
    -------
    y : np.ndarray
        Audio waveform.

    sr : int
        Sampling rate.
    """

    if not os.path.exists(audio_path):

        raise FileNotFoundError(
            f"Audio file not found:\n{audio_path}"
        )

    y, sr = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=True
    )

    return y, sr


# ============================================================
# FIX AUDIO LENGTH
# ============================================================

def fix_audio_length(
    y,
    target_samples=TARGET_SAMPLES
):
    """
    Make audio exactly one second long.

    If longer:
        Keep the first one second.

    If shorter:
        Zero-pad to one second.
    """

    if len(y) > target_samples:

        y = y[:target_samples]

    elif len(y) < target_samples:

        padding = (
            target_samples - len(y)
        )

        y = np.pad(
            y,
            (0, padding),
            mode="constant"
        )

    return y


# ============================================================
# EXTRACT MEL FROM AUDIO WAVEFORM
# ============================================================

def extract_mel_from_audio(
    y,
    sr,
    n_mels=N_MELS,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH
):
    """
    Extract a log-Mel spectrogram directly
    from an already-loaded audio waveform.

    Parameters
    ----------
    y : np.ndarray
        Audio waveform.

    sr : int
        Sampling rate.

    n_mels : int
        Number of Mel frequency bands.

    n_fft : int
        FFT size.

    hop_length : int
        Number of samples between frames.

    win_length : int
        Analysis window size.

    Returns
    -------
    mel_db : np.ndarray

        Log-Mel spectrogram.

        Expected shape:
            (64, 101)
    """

    # --------------------------------------------------------
    # Make audio exactly one second
    # --------------------------------------------------------

    y = fix_audio_length(
        y
    )

    # --------------------------------------------------------
    # Create Mel spectrogram
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        power=2.0
    )

    # --------------------------------------------------------
    # Convert power to decibels
    # --------------------------------------------------------

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db


# ============================================================
# EXTRACT MEL FROM AUDIO FILE
# ============================================================

def extract_mel_spectrogram(
    audio_path,
    sample_rate=SAMPLE_RATE,
    n_mels=N_MELS,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH
):
    """
    Load an audio file and extract its
    log-Mel spectrogram.
    """

    # Load audio
    y, sr = load_audio(
        audio_path,
        sample_rate
    )

    # Extract Mel spectrogram
    mel_db = extract_mel_from_audio(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length
    )

    return mel_db


# ============================================================
# CONVERT MEL TO FEATURE VECTOR
# ============================================================

def mel_to_feature_vector(
    mel_db
):
    """
    Flatten a Mel spectrogram.

    Example:

        (64, 101)
             ↓
        (6464,)
    """

    feature_vector = (
        mel_db.flatten()
    )

    return feature_vector


# ============================================================
# TEST MEL EXTRACTION
# ============================================================

def test_mel_extraction(
    audio_path
):
    """
    Test the complete Mel extraction pipeline
    using one audio file.
    """

    print("=" * 60)

    print(
        "MEL SPECTROGRAM FEATURE EXTRACTION TEST"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Audio path
    # --------------------------------------------------------

    print()

    print(
        "Audio file:"
    )

    print(
        audio_path
    )

    # --------------------------------------------------------
    # Load original audio
    # --------------------------------------------------------

    y, sr = load_audio(
        audio_path
    )

    print()

    print(
        f"Sample rate: {sr} Hz"
    )

    print(
        f"Original audio samples: {len(y)}"
    )

    print(
        f"Original duration: "
        f"{len(y) / sr:.3f} seconds"
    )

    # --------------------------------------------------------
    # Fix audio length
    # --------------------------------------------------------

    y_fixed = fix_audio_length(
        y
    )

    print()

    print(
        f"Fixed audio samples: "
        f"{len(y_fixed)}"
    )

    print(
        f"Fixed duration: "
        f"{len(y_fixed) / sr:.3f} seconds"
    )

    # --------------------------------------------------------
    # Extract Mel spectrogram
    # --------------------------------------------------------

    mel_db = extract_mel_from_audio(
        y_fixed,
        sr
    )

    print()

    print(
        f"Mel spectrogram shape: "
        f"{mel_db.shape}"
    )

    print(
        f"Number of Mel bands: "
        f"{mel_db.shape[0]}"
    )

    print(
        f"Number of time frames: "
        f"{mel_db.shape[1]}"
    )

    # --------------------------------------------------------
    # Verify expected shape
    # --------------------------------------------------------

    if mel_db.shape == (64, 101):

        print()

        print(
            "Shape check: PASS"
        )

    else:

        print()

        print(
            "Shape check: WARNING"
        )

        print(
            "Expected: (64, 101)"
        )

        print(
            f"Received: {mel_db.shape}"
        )

    # --------------------------------------------------------
    # Convert to feature vector
    # --------------------------------------------------------

    feature_vector = (
        mel_to_feature_vector(
            mel_db
        )
    )

    print()

    print(
        f"Feature vector shape: "
        f"{feature_vector.shape}"
    )

    print(
        f"Feature vector length: "
        f"{len(feature_vector)}"
    )

    print()

    print(
        f"Feature minimum: "
        f"{feature_vector.min():.4f}"
    )

    print(
        f"Feature maximum: "
        f"{feature_vector.max():.4f}"
    )

    print(
        f"Feature mean: "
        f"{feature_vector.mean():.4f}"
    )

    print(
        f"Feature standard deviation: "
        f"{feature_vector.std():.4f}"
    )

    # --------------------------------------------------------
    # Plot Mel spectrogram
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel"
    )

    plt.colorbar(
        format="%+2.0f dB"
    )

    plt.title(
        "Log-Mel Spectrogram"
    )

    plt.tight_layout()

    plt.show()

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print()

    print("=" * 60)

    print(
        "TEST COMPLETED"
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    AUDIO_PATH = (
        r"D:\ai_voice_vishing\data\raw\LA\LA"
        r"\ASVspoof2019_LA_train\flac"
        r"\LA_T_1000137.flac"
    )

    test_mel_extraction(
        AUDIO_PATH
    )