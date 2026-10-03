from pathlib import Path
import pandas as pd
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt


# ============================================================
# SETTINGS
# ============================================================

METADATA = Path(
    "data/metadata/asvspoof2019_la_metadata.csv"
)

INPUT_DIR = Path(
    "data/raw/LA/LA/ASVspoof2019_LA_eval/flac"
)

TELEPHONE_DIR = Path(
    "data/processed/phone_eval"
)

CODEC_DIR = Path(
    "data/processed/phone_codec_eval"
)

SAMPLES_PER_CLASS = 500

TARGET_SR = 8000

LOW_FREQ = 300
HIGH_FREQ = 3400


# ============================================================
# TELEPHONE FILTER
# ============================================================

def telephone_filter(audio, sr):

    sos = butter(
        6,
        [LOW_FREQ, HIGH_FREQ],
        btype="bandpass",
        fs=sr,
        output="sos"
    )

    return sosfilt(sos, audio)


# ============================================================
# NORMALIZE
# ============================================================

def normalize(audio):

    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak * 0.95

    return audio


# ============================================================
# TELEPHONE SIMULATION
# ============================================================

def simulate_phone(input_file, output_file):

    audio, sr = librosa.load(
        input_file,
        sr=None,
        mono=True
    )

    # Telephone frequency band
    audio = telephone_filter(
        audio,
        sr
    )

    # 8 kHz telephone sampling
    audio = librosa.resample(
        audio,
        orig_sr=sr,
        target_sr=TARGET_SR
    )

    audio = normalize(audio)

    sf.write(
        output_file,
        audio,
        TARGET_SR
    )


# ============================================================
# G.711 A-LAW ENCODER
# ============================================================

def alaw_encode(samples):

    samples = np.asarray(
        samples,
        dtype=np.int16
    )

    encoded = np.zeros(
        len(samples),
        dtype=np.uint8
    )

    for i, sample in enumerate(samples):

        sample = int(sample)

        sign = 0x80 if sample < 0 else 0x00

        value = abs(sample)

        if value > 32635:
            value = 32635

        if value >= 256:

            exponent = 7
            mask = 0x4000

            while (
                exponent > 0
                and (value & mask) == 0
            ):
                exponent -= 1
                mask >>= 1

            mantissa = (
                value >> (exponent + 3)
            ) & 0x0F

            alaw = (
                sign
                | (exponent << 4)
                | mantissa
            )

        else:

            alaw = (
                sign
                | (value >> 4)
            )

        encoded[i] = alaw ^ 0x55

    return encoded


# ============================================================
# G.711 A-LAW DECODER
# ============================================================

def alaw_decode(encoded):

    encoded = np.asarray(
        encoded,
        dtype=np.uint8
    )

    decoded = np.zeros(
        len(encoded),
        dtype=np.int16
    )

    for i, byte in enumerate(encoded):

        byte = int(byte)

        byte ^= 0x55

        sign = byte & 0x80

        exponent = (
            byte >> 4
        ) & 0x07

        mantissa = byte & 0x0F

        if exponent == 0:

            value = (
                mantissa << 4
            ) + 8

        else:

            value = (
                (mantissa << 4)
                + 0x108
            ) << (exponent - 1)

        if sign:
            value = -value

        decoded[i] = np.clip(
            value,
            -32768,
            32767
        )

    return decoded


# ============================================================
# APPLY G.711 A-LAW
# ============================================================

def apply_alaw(
    input_file,
    output_file
):

    audio, sr = sf.read(
        input_file
    )

    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1
        )

    audio_int16 = np.int16(
        np.clip(
            audio,
            -1.0,
            1.0
        ) * 32767
    )

    encoded = alaw_encode(
        audio_int16
    )

    decoded = alaw_decode(
        encoded
    )

    output_audio = (
        decoded.astype(
            np.float32
        ) / 32767.0
    )

    sf.write(
        output_file,
        output_audio,
        TARGET_SR
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("BUILDING BALANCED PHONE ROBUSTNESS DATASET")
    print("=" * 60)

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    df = pd.read_csv(
        METADATA
    )

    print(
        f"\nTotal metadata rows: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Use EVAL split only
    # --------------------------------------------------------

    eval_df = df[
        df["split"].str.lower() == "eval"
    ].copy()

    print(
        f"Evaluation files: "
        f"{len(eval_df):,}"
    )

    # --------------------------------------------------------
    # Balanced sampling
    # --------------------------------------------------------

    bonafide = eval_df[
        eval_df["label"].str.lower()
        == "bonafide"
    ]

    spoof = eval_df[
        eval_df["label"].str.lower()
        == "spoof"
    ]

    bonafide = bonafide.sample(
        n=SAMPLES_PER_CLASS,
        random_state=42
    )

    spoof = spoof.sample(
        n=SAMPLES_PER_CLASS,
        random_state=42
    )

    selected = pd.concat(
        [
            bonafide,
            spoof
        ],
        ignore_index=True
    )

    selected = selected.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    print("\nSelected dataset:")
    print(
        selected["label"].value_counts()
    )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    TELEPHONE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CODEC_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Generate audio
    # --------------------------------------------------------

    successful = 0
    failed = 0

    print(
        "\nGenerating telephone audio..."
    )

    for index, row in selected.iterrows():

        file_id = str(
            row["file"]
        ).strip()

        input_file = (
            INPUT_DIR
            / f"{file_id}.flac"
        )

        phone_file = (
            TELEPHONE_DIR
            / f"{file_id}_phone.wav"
        )

        codec_file = (
            CODEC_DIR
            / f"{file_id}_phone.wav"
        )

        try:

            if not input_file.exists():

                print(
                    f"Missing: {input_file}"
                )

                failed += 1
                continue

            # Telephone
            simulate_phone(
                input_file,
                phone_file
            )

            # Telephone + G.711
            apply_alaw(
                phone_file,
                codec_file
            )

            successful += 1

            if (
                successful % 100 == 0
            ):

                print(
                    f"Processed "
                    f"{successful}/"
                    f"{len(selected)}"
                )

        except Exception as e:

            print(
                f"ERROR: {file_id}"
            )

            print(e)

            failed += 1

    # --------------------------------------------------------
    # Save selected metadata
    # --------------------------------------------------------

    selected.to_csv(
        "data/metadata/"
        "phone_robustness_eval_metadata.csv",
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PHONE ROBUSTNESS DATASET COMPLETE")
    print("=" * 60)

    print(
        f"\nRequested files: "
        f"{len(selected)}"
    )

    print(
        f"Successful: "
        f"{successful}"
    )

    print(
        f"Failed: "
        f"{failed}"
    )

    print(
        "\nClass distribution:"
    )

    print(
        selected["label"].value_counts()
    )

    print(
        "\nTelephone directory:"
    )

    print(TELEPHONE_DIR)

    print(
        "\nTelephone + G.711 directory:"
    )

    print(CODEC_DIR)

    print(
        "\nMetadata:"
    )

    print(
        "data/metadata/"
        "phone_robustness_eval_metadata.csv"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()