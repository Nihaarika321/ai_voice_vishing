import numpy as np
import librosa


def extract_mfcc(
    audio,
    sr,
    n_mfcc=13,
    n_fft=400,
    hop_length=160
):
    """
    Extract MFCC features from an audio waveform.

    Parameters
    ----------
    audio : np.ndarray
        Audio waveform.

    sr : int
        Sampling rate of the audio.

    n_mfcc : int
        Number of MFCC coefficients.

    n_fft : int
        FFT window size.

    hop_length : int
        Number of samples between successive frames.

    Returns
    -------
    mfcc : np.ndarray
        MFCC matrix with shape:
        (n_mfcc, time_frames)
    """

    audio = np.asarray(audio, dtype=np.float32)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )

    return mfcc


def mfcc_statistics(mfcc):
    """
    Convert the variable-length MFCC matrix into
    a fixed-length feature vector using mean and
    standard deviation across time.

    Returns
    -------
    features : np.ndarray
        Fixed-length feature vector.
    """

    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)

    features = np.concatenate([mean, std])

    return features


if __name__ == "__main__":

    print("Testing MFCC extraction...")

    # Generate 1 second of test audio.
    # This is ONLY a software test.
    sr = 16000
    duration = 1.0

    audio = np.random.randn(
        int(sr * duration)
    ).astype(np.float32)

    # Extract MFCC
    mfcc = extract_mfcc(
        audio,
        sr
    )

    # Convert MFCC to fixed-length vector
    features = mfcc_statistics(mfcc)

    print("Sample rate :", sr)
    print("Audio samples:", len(audio))
    print("MFCC shape   :", mfcc.shape)
    print("Feature shape:", features.shape)

    print("MFCC extraction test complete.")