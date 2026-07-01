"""
Music Generation with AI - LSTM-based MIDI music generator.

This script:
1. Generates a synthetic classical-style MIDI training dataset using music21
2. Preprocesses note sequences for LSTM input
3. Builds and trains an LSTM model using TensorFlow/Keras
4. Generates new music sequences from the trained model
5. Saves the output as a MIDI file

No external MIDI dataset download is required; training data is programmatically
constructed from classical scale patterns, arpeggios, and chord progressions.
"""

import os
import random
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from music21 import stream, note, chord, tempo, meter, instrument

# ── Reproducibility ────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ── Constants ──────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 24        # notes fed into LSTM per training sample
EPOCHS = 60
BATCH_SIZE = 32
LSTM_UNITS = 256
GENERATE_LENGTH = 64        # notes to generate
OUTPUT_MIDI = "generated_music.mid"

# ── 1. Build synthetic training dataset ───────────────────────────────────────

def build_training_corpus():
    """
    Returns a long list of note/chord tokens representing classical-style music.
    Tokens are strings like 'C4', 'E4.G4.C5', 'rest'.
    """
    corpus = []

    # Major scales in several keys
    major_scales = {
        "C": ["C4","D4","E4","F4","G4","A4","B4","C5"],
        "G": ["G4","A4","B4","C5","D5","E5","F#5","G5"],
        "F": ["F4","G4","A4","A#4","C5","D5","E5","F5"],
        "D": ["D4","E4","F#4","G4","A4","B4","C#5","D5"],
        "A": ["A3","B3","C#4","D4","E4","F#4","G#4","A4"],
        "E": ["E4","F#4","G#4","A4","B4","C#5","D#5","E5"],
    }

    # Minor scales
    minor_scales = {
        "Am": ["A3","B3","C4","D4","E4","F4","G4","A4"],
        "Em": ["E4","F#4","G4","A4","B4","C5","D5","E5"],
        "Dm": ["D4","E4","F4","G4","A4","A#4","C5","D5"],
    }

    # Chord progressions (I-V-vi-IV style)
    chord_progressions = [
        ["C4.E4.G4", "G4.B4.D5", "A4.C5.E5", "F4.A4.C5"],   # C major
        ["G4.B4.D5", "D4.F#4.A4", "E4.G4.B4", "C4.E4.G4"],   # G major
        ["F4.A4.C5", "C4.E4.G4", "D4.F4.A4", "A#4.D5.F5"],   # F major
        ["A3.C4.E4", "E3.G3.B3", "F3.A3.C4", "G3.B3.D4"],    # Am
    ]

    def scale_pattern(scale):
        """Ascending, descending and arpeggio patterns from a scale."""
        seq = []
        # Ascending
        seq += scale
        # Descending
        seq += list(reversed(scale))
        # Skip pattern (1 3 5 8)
        arp = [scale[0], scale[2], scale[4], scale[7],
               scale[7], scale[4], scale[2], scale[0]]
        seq += arp
        # Repeated motif
        motif = [scale[0], scale[2], scale[4], scale[2]] * 2
        seq += motif
        return seq

    # Add scale patterns
    for key, sc in {**major_scales, **minor_scales}.items():
        for _ in range(6):
            corpus += scale_pattern(sc)
            # Occasional rest
            corpus += [sc[0], "rest", sc[2], sc[4]]

    # Add chord progressions
    for prog in chord_progressions:
        for _ in range(20):
            corpus += prog
            # Interleave with scale notes
            key = list(major_scales.values())[chord_progressions.index(prog) % len(major_scales)]
            corpus += [key[0], key[2], key[4], key[0]]

    # Add Bach-like ornamental runs
    ornament_base = ["C4","D4","E4","D4","C4","E4","G4","E4",
                     "C5","B4","A4","G4","F4","E4","D4","C4"]
    for _ in range(30):
        corpus += ornament_base
        corpus += list(reversed(ornament_base))

    # NOTE: corpus is intentionally kept in musical order (NOT shuffled).
    # An LSTM learns sequential/melodic dependencies; shuffling token order
    # destroys that structure and was the root cause of the model being
    # unable to exceed ~15% training accuracy in earlier versions of this
    # script. Each musical phrase/pattern block above is already internally
    # ordered and self-contained, so concatenating blocks in this fixed
    # order preserves learnable structure while still giving variety.
    return corpus


# ── 2. Preprocessing ──────────────────────────────────────────────────────────

def prepare_sequences(corpus, seq_len):
    """Convert token corpus to integer-encoded X, y pairs for LSTM training."""
    vocab = sorted(set(corpus))
    tok2idx = {t: i for i, t in enumerate(vocab)}
    idx2tok = {i: t for t, i in tok2idx.items()}

    encoded = [tok2idx[t] for t in corpus]
    X, y = [], []
    for i in range(len(encoded) - seq_len):
        X.append(encoded[i:i + seq_len])
        y.append(encoded[i + seq_len])

    X = np.array(X)
    y = np.array(y)
    return X, y, tok2idx, idx2tok


# ── 3. Build LSTM model ───────────────────────────────────────────────────────

def build_model(vocab_size, seq_len):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=64, input_length=seq_len),
        LSTM(LSTM_UNITS, return_sequences=True),
        Dropout(0.2),
        LSTM(LSTM_UNITS),
        Dropout(0.2),
        Dense(vocab_size, activation="softmax"),
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return model


# ── 4. Generate note sequence ─────────────────────────────────────────────────

def generate_sequence(model, seed, tok2idx, idx2tok, length, temperature=0.8):
    """
    Generate 'length' tokens from a seed sequence.
    Temperature controls randomness: lower = more deterministic.
    """
    vocab_size = len(tok2idx)
    generated = list(seed)
    current_seq = list(seed)

    for _ in range(length):
        x = np.array(current_seq[-SEQUENCE_LENGTH:])
        x = np.pad(x, (max(0, SEQUENCE_LENGTH - len(x)), 0), constant_values=0)
        x = x.reshape(1, SEQUENCE_LENGTH)

        preds = model.predict(x, verbose=0)[0]
        # Temperature scaling
        preds = np.log(preds + 1e-8) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))
        next_idx = np.random.choice(vocab_size, p=preds)

        generated.append(next_idx)
        current_seq.append(next_idx)

    return [idx2tok[i] for i in generated]


# ── 5. Convert tokens to MIDI via music21 ────────────────────────────────────

def tokens_to_midi(tokens, output_path):
    """Convert list of note/chord/rest tokens to a MIDI file."""
    s = stream.Score()
    part = stream.Part()
    part.insert(0, instrument.Piano())
    part.insert(0, meter.TimeSignature("4/4"))
    part.insert(0, tempo.MetronomeMark(number=90))

    quarter = 1.0   # quarter note duration

    for token in tokens:
        if token == "rest":
            r = note.Rest(quarterLength=quarter)
            part.append(r)
        elif "." in token:
            # Chord: tokens like "C4.E4.G4"
            pitches = token.split(".")
            c = chord.Chord(pitches, quarterLength=quarter)
            part.append(c)
        else:
            # Single note
            n = note.Note(token, quarterLength=quarter)
            part.append(n)

    s.append(part)
    s.write("midi", fp=output_path)
    print(f"MIDI saved to: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Step 1: Building synthetic training corpus...")
    corpus = build_training_corpus()
    print(f"  Corpus size: {len(corpus)} tokens")

    print("Step 2: Preprocessing sequences...")
    X, y, tok2idx, idx2tok = prepare_sequences(corpus, SEQUENCE_LENGTH)
    vocab_size = len(tok2idx)
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Training samples: {len(X)}")

    print("Step 3: Building LSTM model...")
    model = build_model(vocab_size, SEQUENCE_LENGTH)
    model.summary()

    print("Step 4: Training model...")
    early_stop = EarlyStopping(monitor="loss", patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1)
    history = model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )
    final_acc = history.history["accuracy"][-1]
    final_loss = history.history["loss"][-1]
    print(f"  Final loss: {final_loss:.4f} | Final accuracy: {final_acc:.4f}")

    print("Step 5: Generating music sequence...")
    # Use last SEQUENCE_LENGTH tokens of corpus as seed
    seed_tokens = corpus[-SEQUENCE_LENGTH:]
    seed_indices = [tok2idx[t] for t in seed_tokens]
    generated_tokens = generate_sequence(
        model, seed_indices, tok2idx, idx2tok,
        length=GENERATE_LENGTH, temperature=0.8
    )
    print(f"  Generated {len(generated_tokens)} tokens")

    print("Step 6: Converting to MIDI...")
    tokens_to_midi(generated_tokens, OUTPUT_MIDI)

    print("\nDone! Open 'generated_music.mid' with any MIDI player to listen.")


if __name__ == "__main__":
    main()
