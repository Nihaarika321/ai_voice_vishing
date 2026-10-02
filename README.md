# AI Voice Vishing Detection

## Real-Time Detection of AI-Cloned Voices in Banking Vishing Calls

A research project for detecting AI-generated / AI-cloned voices used in fraudulent banking and vishing calls.

The project focuses on **short-duration telephone audio**, realistic banking-vishing scenarios, and robustness under telephone-channel conditions. The system is designed to produce an AI-voice probability that can be aggregated over consecutive audio windows to support a simulated fraud warning.

> **Important:** This project is a research prototype. It does not connect to real banking systems, automatically terminate calls, or make financial decisions.

---

## 1. Project Objective

Voice cloning and speech synthesis can be used by attackers to impersonate bank representatives during vishing attacks.

The proposed pipeline is:

```text
AI-cloned bank representative
            ↓
     Fraudulent bank call
            ↓
       Incoming audio
            ↓
       Preprocessing
            ↓
     Short audio window
            ↓
      Feature extraction
            ↓
     AI voice detector
            ↓
     AI-voice probability
            ↓
    Temporal aggregation
            ↓
      Fraud warning
```

Example warning:

```text
Potential AI-generated voice detected.
Verify the caller through an official banking channel
before sharing OTP, PIN, password, or other sensitive information.
```

---

# 2. Project Architecture

```text
                         ┌──────────────────────┐
                         │   Incoming Call      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Preprocessing     │
                         │                      │
                         │ Stereo → Mono        │
                         │ Resampling            │
                         │ Telephone simulation  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  1-second Window     │
                         │  0.5-second Hop      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Feature Extraction   │
                         │                      │
                         │ MFCC                 │
                         │ Mel Spectrogram      │
                         │ Pitch / F0           │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ AI Voice Detection   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Temporal Aggregation │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Simulated Warning    │
                         └──────────────────────┘
```

---

# 3. Dataset Strategy

The project uses different datasets for different purposes.

| Dataset          | Purpose                             |
| ---------------- | ----------------------------------- |
| ASVspoof 2019 LA | Main model training/development     |
| ASVspoof 2021 DF | Held-out evaluation                 |
| VISHGUARD        | External banking-vishing evaluation |
| BANKING77        | Banking-domain text/scenario design |

### Important

BANKING77 is a **text-only banking intent dataset**. It is not used as an audio dataset and is not used to train the voice detector.

It is used to identify realistic banking-related intents and construct banking-vishing scenarios.

VISHGUARD is kept as an **external evaluation dataset** rather than being used to train the detector.

---

# 4. ASVspoof 2019 LA

ASVspoof 2019 Logical Access is the primary audio dataset for training and development.

Current dataset statistics:

```text
Total audio files : 121,461

Train : 25,380
Dev   : 24,844
Eval  : 71,237

Bona fide : 12,483
Spoof     : 108,978
```

Audio characteristics:

```text
Sample rate : 16 kHz
Channels    : Mono
Format      : FLAC
```

Official protocol files are used for labels rather than inferring labels from filenames.

### Short-window availability

The dataset was analyzed for fixed-duration windows.

| Window | Available files |
| -----: | --------------: |
|  3.0 s |          60,137 |
|  2.0 s |          98,113 |
|  1.5 s |         112,488 |
|  1.0 s |         120,184 |
| 0.75 s |         121,290 |
|  0.5 s |         121,460 |

The project uses short windows to investigate low-latency voice-spoof detection.

---

# 5. Telephone-Channel Simulation

Real vishing calls commonly pass through telephone-bandwidth conditions.

The preprocessing pipeline therefore supports:

### Clean

```text
Original audio
     ↓
16 kHz mono
```

### Telephone

```text
Original audio
     ↓
300–3400 Hz bandpass
     ↓
8 kHz
     ↓
Peak normalization
```

### Telephone + G.711 A-law

```text
Original audio
     ↓
300–3400 Hz bandpass
     ↓
8 kHz
     ↓
G.711 A-law simulation
     ↓
Decoded audio
```

The processed audio is generated dynamically rather than storing multiple copies of the dataset.

---

# 6. VISHGUARD Banking-Vishing Evaluation

VISHGUARD is used as an external real-world-oriented vishing evaluation resource.

The project selected a controlled subset of:

```text
Total calls       : 40
Fraudulent/spoof  : 20
Legitimate        : 20
Language          : English
```

The selected calls are focused on banking, card, account, transaction, transfer, identity-verification, and related financial-call scenarios.

### Original VISHGUARD audio characteristics

The selected 40 calls contain:

```text
24 kHz
44.1 kHz
48 kHz
```

and both mono and stereo recordings.

Observed distribution:

```text
Sample rate
24 kHz   : 23
44.1 kHz : 14
48 kHz   : 3

Channels
Stereo : 36
Mono   : 4
```

Duration:

```text
Mean       : 21.006 s
Minimum    : 11.938 s
Maximum    : 28.896 s
```

The original VISHGUARD recordings are preserved.

---

# 7. VISHGUARD Short-Window Evaluation

VISHGUARD calls are converted logically into fixed windows using metadata.

No physical audio snippets are created.

Configuration:

```text
Window size : 1.0 second
Hop size    : 0.5 second
```

Result:

```text
Total windows : 1,617

Spoof windows    : 999
Bonafide windows : 618

Calls represented : 40 / 40
```

The difference in window counts is expected because the selected spoof and bonafide calls have different durations.

### Important evaluation rule

Overlapping windows from the same call must **not** be treated as independent calls.

Predictions are aggregated using the original:

```text
call_id
```

to obtain call-level evaluation results.

---

# 8. VISHGUARD Dynamic Preprocessing

VISHGUARD audio is dynamically standardized before feature extraction.

```text
Original WAV
     ↓
Read at original sample rate
     ↓
Stereo → Mono
     ↓
Resample
     ↓
┌────────────────────────────┐
│ Clean                      │
│ → 16 kHz                   │
├────────────────────────────┤
│ Telephone                  │
│ → 300–3400 Hz              │
│ → 8 kHz                    │
├────────────────────────────┤
│ Telephone + G.711 A-law   │
│ → telephone processing     │
│ → G.711 A-law simulation   │
│ → 8 kHz                    │
└────────────────────────────┘
     ↓
1-second window
     ↓
Model input
```

The loader was tested using different VISHGUARD audio files.

Final verification:

```text
Windows tested : 10
Conditions     : 3
Tests passed   : 30
Tests failed   : 0
```

Validated outputs:

```text
Clean              → 16,000 samples @ 16 kHz
Telephone          →  8,000 samples @  8 kHz
Telephone + codec  →  8,000 samples @  8 kHz
```

---

# 9. BANKING77

BANKING77 is used for banking-domain language and scenario development.

The dataset contains 77 banking-related intents.

Relevant categories used during scenario construction include:

```text
card_payment_not_recognised
cash_withdrawal_not_recognised
compromised_card
beneficiary_not_allowed
declined_transfer
failed_transfer
pending_transfer
transfer_not_received_by_recipient
verify_my_identity
unable_to_verify_identity
verify_source_of_funds
verify_top_up
top_up_failed
cash_withdrawal_charge
card_payment_fee_charged
extra_charge_on_statement
direct_debit_payment_not_recognised
Refund_not_showing_up
request_refund
card_not_working
card_swallowed
lost_or_stolen_card
pin_blocked
change_pin
```

A banking-vishing scenario set was constructed from these intents.

Current scenario set:

```text
18 scenarios
90 utterances
5 utterances per scenario
```

Examples include:

```text
Suspicious card transaction
Suspicious cash withdrawal
Compromised card
Failed bank transfer
Pending bank transfer
Transfer not received
Identity verification
Source-of-funds verification
Failed account top-up
Unknown direct debit
Unexpected account charge
Card not working
Lost or stolen card
PIN blocked
Refund missing
Refund request
Beneficiary problem
```

---

# 10. Repository Structure

```text
ai_voice_vishing/
│
├── data/
│   ├── raw/
│   │   ├── LA/
│   │   │   └── ASVspoof2019_LA/
│   │   │
│   │   ├── ASVspoof2021_DF_eval/
│   │   │
│   │   ├── banking_text/
│   │   │
│   │   ├── banking_vishing/
│   │   │   ├── audio/
│   │   │   │   ├── spoof/
│   │   │   │   └── bonafide/
│   │   │   │
│   │   │   └── scripts/
│   │   │       ├── banking_call_scripts.csv
│   │   │       └── vishing_scenarios.csv
│   │   │
│   │   └── vishguard/
│   │
│   ├── metadata/
│   │   ├── asvspoof2019_la_metadata.csv
│   │   ├── short_window_metadata.csv
│   │   ├── phone_conditions_metadata.csv
│   │   ├── banking_vishing_metadata.csv
│   │   ├── vishguard_final_metadata.csv
│   │   ├── vishguard_short_window_metadata.csv
│   │   └── vishguard_audio_validation.csv
│   │
│   └── processed/
│
├── preprocessing/
│   ├── audio_loader.py
│   ├── vishguard_audio_loader.py
│   ├── phone_simulator.py
│   ├── codec_simulator.py
│   ├── create_phone_metadata.py
│   ├── create_banking_scripts.py
│   ├── create_banking_audio_manifest.py
│   ├── create_vishguard_windows.py
│   ├── select_vishguard_subset.py
│   ├── validate_datasets.py
│   ├── validate_banking_audio.py
│   ├── validate_vishguard_audio.py
│   ├── inspect_vishguard.py
│   ├── analyze_vishguard.py
│   ├── test_metadata_loader.py
│   ├── test_window_conditions.py
│   ├── test_vishguard_loader.py
│   └── verify_vishguard_windows.py
│
├── features/
│
├── models/
│
├── evaluation/
│
├── results/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 11. Preprocessing Validation Status

### ASVspoof

```text
Dataset metadata       ✓
Official labels        ✓
Short-window metadata  ✓
Telephone simulation   ✓
G.711 A-law simulation ✓
Dynamic loader         ✓
Loader tests           ✓
```

### VISHGUARD

```text
40-call selection          ✓
Audio extraction           ✓
Audio validation            ✓
Final metadata              ✓
Short-window metadata      ✓
Dynamic audio loader       ✓
Window verification        ✓
```

Final VISHGUARD validation:

```text
40 / 40 audio files valid
20 / 20 spoof
20 / 20 bonafide
30 / 30 loader/window tests passed
```

---

# 12. Experimental Protocol

The intended experimental protocol is:

```text
                    TRAINING
                       │
                       ▼
               ASVspoof 2019
                train split
                       │
                       ▼
              Feature extraction
                       │
                       ▼
                    Model
                       │
                       ▼
                    Tuning
                       │
                       ▼
               ASVspoof dev
                       │
                       ▼
                 Freeze model
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
 ASVspoof evaluation        VISHGUARD evaluation
          │                         │
          ▼                         ▼
  Dataset performance       External performance
```

VISHGUARD should not be used to train the primary detector.

This separation allows the project to evaluate whether a model trained on a standard spoofing dataset transfers to banking-vishing audio.

---

# 13. Short-Window / Streaming Concept

The system is designed around short windows rather than complete calls.

For example:

```text
Incoming audio

0.0 ───── 1.0 s
      ↓
0.5 ───── 1.5 s
      ↓
1.0 ───── 2.0 s
      ↓
1.5 ───── 2.5 s
      ↓
...
```

Each window produces an AI-voice probability:

```text
Window 1 → 0.72
Window 2 → 0.81
Window 3 → 0.76
Window 4 → 0.88
...
```

A temporal aggregation layer can then combine consecutive predictions into a call-level risk signal.

This supports the project's focus on **low-latency detection**.

---

# 14. Planned Features

The planned acoustic features include:

### MFCC

Mel-frequency cepstral coefficients for compact spectral representation.

### Mel Spectrogram

Time-frequency representation useful for neural-network-based audio classification.

### Pitch / F0

Fundamental-frequency information that may capture differences in speech production.

### Additional acoustic features

Depending on the final model design:

```text
Spectral centroid
Spectral bandwidth
Spectral rolloff
Zero-crossing rate
RMS energy
```

Feature extraction must use the same preprocessing and feature definitions across training and evaluation datasets.

---

# 15. Planned Models

The model component is being developed separately from the dataset/preprocessing component.

Potential model architecture:

```text
Audio
  ↓
Short window
  ↓
MFCC / Mel / Pitch
  ↓
CNN / lightweight classifier
  ↓
Spoof probability
```

The final model will be evaluated using both conventional classification metrics and short-window/call-level metrics.

---

# 16. Evaluation Metrics

Planned metrics include:

```text
Accuracy
Precision
Recall
F1-score
ROC-AUC
EER
Confusion matrix
```

For short-window evaluation:

```text
Window-level performance
Call-level performance
Detection latency
False alarm rate
```

For VISHGUARD, call-level aggregation is particularly important because multiple overlapping windows originate from the same call.

---

# 17. Reproducibility

Python environment:

```text
Python 3.12
```

Core packages currently used:

```text
numpy
pandas
librosa
soundfile
scipy
scikit-learn
tqdm
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

Run preprocessing validation:

```bash
python -m preprocessing.validate_datasets
```

Run VISHGUARD audio validation:

```bash
python -m preprocessing.validate_vishguard_audio
```

Run VISHGUARD loader validation:

```bash
python -m preprocessing.test_vishguard_loader
```

Run VISHGUARD short-window verification:

```bash
python -m preprocessing.verify_vishguard_windows
```

---

# 18. Data and Licensing

Large third-party datasets are **not intended to be redistributed through this Git repository**.

In particular:

* ASVspoof 2019 files should be obtained from the official dataset source.
* ASVspoof 2021 files should be obtained from the official dataset source.
* BANKING77 should be obtained from its official dataset source.
* VISHGUARD should be obtained according to its official dataset license and distribution terms.

Only project-generated metadata, scripts, and permitted evaluation data should be committed to this repository.

Before redistributing any third-party audio, verify the corresponding dataset license and terms.

---

# 19. Current Project Status

```text
┌────────────────────────────────────────────┐
│ Component                    │ Status       │
├──────────────────────────────┼──────────────┤
│ ASVspoof metadata            │ COMPLETE     │
│ ASVspoof short windows       │ COMPLETE     │
│ Telephone simulation         │ COMPLETE     │
│ G.711 A-law simulation       │ COMPLETE     │
│ ASVspoof dynamic loader      │ COMPLETE     │
│ BANKING77 scenario design    │ COMPLETE     │
│ Banking audio manifest       │ COMPLETE     │
│ VISHGUARD selection          │ COMPLETE     │
│ VISHGUARD audio extraction   │ COMPLETE     │
│ VISHGUARD validation         │ COMPLETE     │
│ VISHGUARD metadata           │ COMPLETE     │
│ VISHGUARD windows            │ COMPLETE     │
│ VISHGUARD loader             │ COMPLETE     │
│ Window verification          │ COMPLETE     │
│ Feature extraction           │ NEXT         │
│ Model training               │ NEXT         │
│ Streaming aggregation        │ PLANNED      │
│ Final evaluation             │ PLANNED      │
└──────────────────────────────┴──────────────┘
```

---

# 20. Research Contribution

The project investigates a practical pipeline for detecting AI-generated voices in banking-vishing scenarios by combining:

1. Standardized anti-spoofing audio data.
2. Short-duration audio windows.
3. Telephone-channel simulation.
4. Codec-aware preprocessing.
5. Banking-specific scenario design.
6. External evaluation on VISHGUARD.
7. Call-level temporal aggregation.
8. Low-latency detection considerations.

The goal is to study the gap between performance on standard voice-spoofing datasets and performance under realistic banking-vishing conditions.

---

## Disclaimer

This project is intended for academic and research purposes. It does not replace banking security systems, caller authentication mechanisms, or official fraud-detection infrastructure.
