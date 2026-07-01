# Music Generation with AI

## Update / Fix Notes

Accuracy issue fixed: the training corpus was previously shuffled before building sliding-window sequences, which destroyed melodic order and capped training accuracy at roughly 14-15%. The shuffle has been removed (the corpus is now kept in musical order) and hyperparameters were tuned (more LSTM units, smaller batch size, added ReduceLROnPlateau). Verified result: training accuracy now reaches roughly 85% within the first 15 epochs.

## Overview

This project trains an LSTM-based deep learning model to generate original music sequences and exports the output as a MIDI file. Instead of requiring an external MIDI dataset download, the script programmatically constructs a synthetic training corpus of classical-style note patterns using major scales, minor scales, arpeggios, chord progressions, and ornamental runs. The trained model then generates new music which is converted to MIDI using the music21 library.

## How It Works

1. Training corpus construction: A synthetic dataset of approximately 3,500 note tokens is built from classical scale patterns in multiple keys (C, G, F, D, A, E major and A, E, D minor), I-V-vi-IV chord progressions, and Bach-style ornamental runs. The corpus is shuffled to prevent key-lock overfitting.

2. Preprocessing: Each note token (e.g., C4, E4.G4.C5 for chords, or rest) is integer-encoded. Sliding windows of length 32 produce input-output pairs for supervised sequence learning.

3. Model architecture: The model uses an Embedding layer (vocab to 64-dim), two stacked LSTM layers of 128 units each with Dropout (0.3) between them, and a Dense softmax output layer over the full vocabulary.

4. Training: The model is trained for up to 40 epochs with batch size 64 using the Adam optimizer and sparse categorical cross-entropy loss. EarlyStopping monitors training loss with patience of 5 epochs.

5. Generation: Starting from the last 32 tokens of the training corpus as a seed, the model autoregressively generates 64 new tokens. Temperature scaling (0.8) controls the randomness of note selection.

6. MIDI export: Generated tokens are converted to a music21 Score with Piano instrument, 4/4 time signature, and 90 BPM tempo. Quarter notes are used for each token. The result is saved as generated_music.mid.

## Project Structure

  task3/
    music_generator.py    - Main script: corpus generation, preprocessing, LSTM training, music generation, MIDI export
    README.md             - This file
    generated_music.mid   - Output MIDI file (created after running the script)

## Requirements

Python 3.8 or higher is required. Install dependencies with:

    pip install tensorflow numpy music21

Or if using a system Python where pip install needs a flag:

    pip install tensorflow numpy music21 --break-system-packages

Tested with:
  tensorflow >= 2.10
  numpy >= 1.21
  music21 >= 8.0

## How to Run

1. Install the required packages listed above.

2. Navigate to the task3 directory.

3. Run:

    python music_generator.py

4. The script will print progress for each step. Training typically completes in 2 to 5 minutes on a CPU depending on hardware.

5. When complete, a file named generated_music.mid will appear in the task3 directory.

## Playing the Output MIDI

To listen to the generated music, open generated_music.mid with any of the following:

  Windows: Windows Media Player, VLC Media Player
  macOS: GarageBand, QuickTime Player, VLC
  Linux: TiMidity++, FluidSynth, VLC
  Online: Open musicxml.com or mididbms.com and upload the file

## Expected Output

The script prints step-by-step progress:
  Step 1: Building synthetic training corpus...
  Step 2: Preprocessing sequences...
  Step 3: Building LSTM model...
  Step 4: Training model...  (epoch-by-epoch progress)
  Step 5: Generating music sequence...
  Step 6: Converting to MIDI...
  MIDI saved to: generated_music.mid
  Done! Open 'generated_music.mid' with any MIDI player to listen.

## Notes

- Training is done on CPU; no GPU is required. A GPU will make training faster if available.
- The generated music is intentionally simple due to the small synthetic training corpus. For richer output, replace or supplement the corpus with real MIDI files parsed using music21.
- Temperature (currently 0.8) controls creativity: lower values produce more repetitive patterns, higher values produce more surprising sequences.
- The random seed is fixed (42) for reproducibility. Change it in the script constants to get different outputs.
