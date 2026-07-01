# AI and Web Projects Repository

This repository contains four independent projects built as part of an AI internship at CodeAlpha. Each project lives in its own folder and is self-contained. The four projects are a Language Translation Tool, a FAQ Chatbot, a Music Generation AI, and an Object Detection and Tracking system.

---

## Repository Structure

    /
    |-- task1/  (Language Translation Tool)
    |   |-- language-translation-tool/
    |       |-- index.html
    |       |-- script.js
    |       |-- style.css
    |       |-- README.md
    |
    |-- task2/  (FAQ Chatbot)
    |   |-- task2/
    |       |-- chatbot.py
    |       |-- index.html
    |       |-- README.md
    |
    |-- task3/  (Music Generation with AI)
    |   |-- task3/
    |       |-- music_generator.py
    |       |-- README.md
    |
    |-- task4/  (Object Detection and Tracking)
    |   |-- task4/
    |       |-- object_detection_tracking.py
    |       |-- README.md
    |
    |-- README.md  (this file)

---

## Task 1: Language Translation Tool

### What It Does

A browser-based language translation tool that accepts text input, lets the user choose a source language and a target language from a dropdown of 30 languages, and returns the translated text using the MyMemory Translation API. It also supports text-to-speech playback for both the original and translated text, copying the translation to the clipboard, swapping the source and target languages, and a live character counter capped at 1000 characters.

### Files

    index.html   Markup and layout. Three-panel structure: source text box, swap button, target text box.
    script.js    All application logic. Handles the translate button click, fetch call to MyMemory API, swap button, copy button, speak buttons, and character count update.
    style.css    Styling. Responsive two-column layout using flexbox, collapses to single column on screens narrower than 768px. Uses Font Awesome icons (loaded from cdnjs CDN) for the speaker and copy buttons.

### API Used

MyMemory Translation API (free, no registration required).
Endpoint: https://api.mymemory.translated.net/get
Parameters: q (URL-encoded source text), langpair (format: sourcecode|targetcode, e.g. en|fr).
Response field used: data.responseData.translatedText.
Free tier allows approximately 1000 words per day per IP address without any API key.

### Languages Supported

English, Arabic, Bengali, Chinese, Czech, Danish, Dutch, Finnish, French, German, Greek, Hebrew, Hindi, Hungarian, Indonesian, Italian, Japanese, Korean, Malay, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Spanish, Swedish, Thai, Turkish, Vietnamese.

### How to Run

No installation required. Open index.html directly in any modern browser. If you get CORS issues in some browsers when opening a local file, serve it with:

    python -m http.server 8000

Then open http://localhost:8000 in your browser.

### Dependencies

No npm packages or build steps. Font Awesome is loaded from a CDN link in the HTML head. An internet connection is required at runtime for the translation API call and the Font Awesome CDN.

### How the Code Works (step by step)

1. On DOMContentLoaded, all DOM element references are cached and the character counter is initialized.
2. The user types in the left textarea. The input event fires and updates the character counter span.
3. The user clicks "Translate Text". The click handler reads sourceText.value, builds the API URL, sets the button to disabled and shows "Translating..." in the output textarea, then calls fetch().
4. The fetch response is parsed as JSON. data.responseData.translatedText is written into the target textarea.
5. Error handling: if fetch throws (network offline) or the response is not ok, a user-friendly error message is shown.
6. The swap button exchanges the values of the two language dropdowns and the two textareas simultaneously using destructuring assignment.
7. The copy button writes targetText.value to the clipboard using navigator.clipboard.writeText() and briefly changes the icon to a checkmark for visual feedback.
8. The speak buttons call speechSynthesis.speak() with a SpeechSynthesisUtterance constructed from the textarea value and language code.

---

## Task 2: FAQ Chatbot

### What It Does

A tech-topic FAQ chatbot that matches a user's natural-language question to the closest pre-stored FAQ answer using TF-IDF vectorization and cosine similarity. No external NLP libraries are required. The project ships in two forms: a Python terminal chatbot (chatbot.py) and a browser-based chat UI (index.html) where both versions implement the exact same NLP pipeline independently in Python and JavaScript.

### Files

    chatbot.py   Python terminal chatbot. Contains the full FAQ dataset, preprocessing pipeline (tokenize, stopword removal, TF-IDF, cosine similarity), and a REPL loop for terminal interaction.
    index.html   Self-contained browser chat UI. Contains the same FAQ dataset and the same NLP pipeline re-implemented in plain JavaScript. No server needed.

### FAQ Dataset

30 question-answer pairs covering: Python, JavaScript, React, machine learning, deep learning, neural networks, LSTM, NLP, supervised vs unsupervised learning, overfitting, APIs, REST, GET vs POST, Docker, Git, SQL vs NoSQL databases, algorithms, data structures, hash tables, recursion, OOP, compilers, cloud computing, blockchain, large language models, and basic conversational intents (hello, thank you, what can you do).

### NLP Pipeline (same in both Python and JS versions)

Step 1: Tokenization. Input text is lowercased. All punctuation is stripped. The result is split on whitespace into a list of tokens.

Step 2: Stopword removal. A hardcoded set of approximately 55 common English stopwords (a, an, the, is, it, by, to, for, what, how, etc.) is used to filter the token list. No external stopword library is used.

Step 3: TF-IDF (Term Frequency - Inverse Document Frequency). At startup, all FAQ question and answer texts are concatenated and preprocessed into token lists. IDF is computed over this corpus using the formula log((N+1)/(df+1))+1 where N is the number of documents and df is the number of documents containing the term, with +1 smoothing to avoid division by zero. At query time, TF-IDF vectors are computed for both the query and every FAQ document.

Step 4: Cosine similarity. The query TF-IDF vector is compared against each FAQ TF-IDF vector. The dot product divided by the product of the vector magnitudes gives the cosine similarity score between 0 and 1.

Step 5: Match selection. The FAQ with the highest cosine similarity score above a minimum threshold of 0.05 is returned as the answer. If no FAQ clears the threshold, a fixed fallback message is returned.

### How to Run

Terminal version (Python):

    python chatbot.py

Type any question at the "You:" prompt. Type quit or exit to stop. Requires Python 3.6 or higher. No pip installs needed (only standard library: string, math).

Browser version:

    Open index.html directly in any browser, or serve with: python -m http.server 8001

No internet connection required. The entire NLP pipeline and FAQ dataset are embedded in the page.

### Edge Cases Handled

- Empty input (whitespace only): returns "Please type a question." without crashing.
- All-stopword input (e.g. "the a is"): tokenizer returns empty list, TF-IDF vector is empty dict, cosine similarity returns 0.0 without division by zero, fallback message is shown.
- Very long input: no length limit, processes normally.
- Gibberish input: cosine similarity scores are all below threshold, fallback message is shown.

---

## Task 3: Music Generation with AI

### What It Does

Trains a two-layer LSTM neural network on a programmatically generated synthetic corpus of classical-style MIDI note sequences, then generates new music using the trained model and saves it as a playable MIDI file (generated_music.mid). No external MIDI dataset download is required; the training data is built from scale patterns, chord progressions, and ornamental runs inside the script itself.

### File

    music_generator.py   Single script containing: corpus generation, preprocessing, model definition, training, sequence generation, and MIDI export.

### How It Works (step by step)

Step 1: Corpus generation. A synthetic training corpus of approximately 3544 tokens is built programmatically using music21. Tokens are strings representing musical events: single notes (e.g. C4, F#5), chords (e.g. C4.E4.G4 where dot separates pitches), and rests. The corpus includes ascending and descending scale patterns in 6 major keys (C, G, F, D, A, E) and 3 minor keys (Am, Em, Dm), skip/arpeggio patterns (root, third, fifth, octave), I-V-vi-IV chord progressions in 4 keys, motif repetition patterns, and Bach-style ornamental runs. The corpus is kept in musical sequence order (not shuffled) so that the LSTM can learn temporal patterns.

Step 2: Preprocessing. The token list is integer-encoded into a vocabulary (34 unique tokens in the default corpus). Sliding windows of length 24 are applied to produce (input_sequence, next_token) training pairs. This yields 3520 training samples.

Step 3: Model architecture. The model is a Keras Sequential model with: Embedding layer (vocab_size input, 64-dimensional output, sequence length 24), LSTM layer with 256 units returning sequences, Dropout 0.2, second LSTM layer with 256 units, Dropout 0.2, Dense output layer with softmax activation over the full vocabulary. Compiled with Adam optimizer (learning rate 0.001) and sparse categorical crossentropy loss.

Step 4: Training. Up to 60 epochs with batch size 32. Two callbacks: EarlyStopping (monitors training loss, patience 8, restores best weights) and ReduceLROnPlateau (monitors loss, factor 0.5, patience 4, minimum LR 1e-5). With the corpus in correct sequential order, the model reaches approximately 85% training accuracy by epoch 13 to 15 on CPU.

Step 5: Generation. The last 24 tokens of the training corpus are used as the seed sequence. At each step, the seed is padded to length 24, passed to model.predict(), the output probability distribution is temperature-scaled (temperature 0.8), and the next token is sampled. 64 new tokens are generated autoregressively. Temperature 0.8 balances variety and coherence; lower values are more repetitive, higher values are more random.

Step 6: MIDI export. Generated tokens are converted to a music21 Score with a Piano instrument, 4/4 time signature, and 90 BPM tempo. Each token becomes a quarter note. Single notes become Note objects, dot-separated tokens become Chord objects, and "rest" tokens become Rest objects. The score is written to generated_music.mid.

### How to Run

    pip install tensorflow numpy music21
    python music_generator.py

The script prints step-by-step progress. On CPU (no GPU), training takes roughly 2 to 10 minutes depending on hardware. When finished, open generated_music.mid with VLC, Windows Media Player, GarageBand, or any MIDI player.

### Requirements

Python 3.8 or higher. tensorflow 2.10 or higher. numpy 1.21 or higher. music21 8.0 or higher.

### Known Fix Applied

An earlier version of this script shuffled the corpus before building training sequences (random.shuffle(corpus)). This destroyed all melodic/temporal order and was the root cause of training accuracy being stuck at approximately 14%. The shuffle has been removed. With sequential order preserved, accuracy reaches approximately 85% in the same number of epochs on the same hardware.

---

## Task 4: Object Detection and Tracking

### What It Does

Performs real-time object detection and multi-object tracking on a live webcam feed or a video file. Each video frame is passed to a YOLOv8 model for detection. Detected bounding boxes are then passed to a from-scratch SORT (Simple Online and Realtime Tracking) implementation that assigns persistent integer track IDs to objects across frames using a Kalman filter for motion prediction and IoU-based greedy matching for data association. Results are rendered on each frame with colored bounding boxes, track ID, class label, and confidence score.

### File

    object_detection_tracking.py   Single script containing: SORT tracker (KalmanBoxTracker and Sort classes), YOLOv8 detector (via ultralytics), YOLOv3 fallback detector (via OpenCV DNN with auto-download), drawing utilities, and the main video loop with argument parsing.
    yolov8n.pt                     YOLOv8 nano weights (included). Used if you pass --model yolov8n.pt to avoid downloading the default yolov8s weights.

### Detection Backend

Primary: YOLOv8 via the ultralytics Python package. Default model is yolov8s.pt (small variant) which downloads automatically on first run (approximately 22 MB). The small variant is meaningfully more accurate than the nano variant (yolov8n.pt) for distinguishing visually similar small objects such as phones, remotes, books, and wallets.

Fallback: If ultralytics is not installed, the script automatically falls back to YOLOv3 via OpenCV's built-in DNN module. On first run it downloads the YOLOv3 config, COCO class names, and weights (approximately 236 MB) from public Darknet sources.

Both backends detect 80 object classes from the COCO dataset.

### SORT Tracker (implemented from scratch, no external tracking library needed)

KalmanBoxTracker: Each tracked object has its own 7-dimensional Kalman filter. The state vector is [center_x, center_y, scale, aspect_ratio, velocity_cx, velocity_cy, velocity_scale]. A constant-velocity motion model is used (state transition matrix F). The measurement matrix H maps the full state to the 4-dimensional observation [cx, cy, s, r]. Process noise Q and measurement noise R are diagonal matrices with hand-tuned values. Prediction uses the standard Kalman predict step (x = F*x, P = F*P*F^T + Q). Update uses the standard Kalman update step with the innovation, Kalman gain, and posterior covariance.

Sort: Manages a list of KalmanBoxTracker instances. On each frame: (1) all existing trackers predict their next bounding box. (2) Incoming detections are matched to predicted boxes using a greedy IoU-based assignment (pairs sorted by IoU descending; accepted if IoU >= iou_threshold). (3) Matched trackers are updated with the detected box. (4) Unmatched detections create new trackers. (5) Trackers not matched for more than max_age frames are removed. Track IDs are monotonically increasing integers and are never reused.

### How to Run

Install dependencies (once):

    pip install opencv-python ultralytics numpy

Run with webcam (camera index 0):

    python object_detection_tracking.py --source 0

Run with a video file:

    python object_detection_tracking.py --source path/to/video.mp4

Run with webcam and save output to file:

    python object_detection_tracking.py --source 0 --output result.mp4

Use the included yolov8n weights (no download needed):

    python object_detection_tracking.py --source 0 --model yolov8n.pt

Run headless (no display window, for servers):

    python object_detection_tracking.py --source path/to/video.mp4 --output result.mp4 --headless

Press ctrl+c at any time to stop.

### Command-Line Arguments

    --source      Video source. Integer for webcam index (0 is default camera), or path to a video file. Default: 0
    --output      Path to save annotated output video (e.g. result.mp4). Optional. If omitted, no file is saved.
    --conf        Detection confidence threshold, 0.0 to 1.0. Default: 0.45. Lower to detect more objects (more false positives). Raise to reduce false positives (may miss some objects).
    --iou         NMS IoU threshold for suppressing duplicate overlapping boxes. Default: 0.45.
    --imgsz       Inference image size. Default: 640. Larger values improve detection of small objects but are slower.
    --model       YOLOv8 weights file. Default: yolov8s.pt (downloads if not cached). Options: yolov8n.pt (fastest, less accurate), yolov8s.pt (default, good balance), yolov8m.pt (more accurate, slower).
    --headless    Flag. Run without opening a display window.

### Requirements

Python 3.8 or higher. opencv-python 4.5 or higher. numpy 1.21 or higher. ultralytics (for YOLOv8; if not installed the script falls back to YOLOv3 automatically). On Linux servers without a display, use opencv-python-headless instead of opencv-python.

### Known Fix Applied

An earlier version defaulted to yolov8n.pt with no explicit imgsz or NMS IoU parameters. This caused frequent misclassification of small, flat, rectangular handheld objects (items being labelled "cell phone" regardless of what they actually were). The fix: default model changed to yolov8s.pt, explicit imgsz=640 and iou=0.45 passed to every inference call, and agnostic_nms set to False to keep NMS class-aware so two different objects overlapping (e.g. hand and phone) are not merged into one box.

---

## Global Requirements Summary

    Task 1:   Any modern web browser. Internet connection for API calls.
    Task 2:   Any modern web browser (browser version). Python 3.6+ with no extra packages (terminal version).
    Task 3:   Python 3.8+, tensorflow, numpy, music21.
    Task 4:   Python 3.8+, opencv-python, numpy, ultralytics.

---

## How to Install All Python Dependencies at Once

    pip install tensorflow numpy music21 opencv-python ultralytics

---

## Notes

- All four tasks are independent. You can run any one of them without the others.
- Tasks 1 and 2 (browser versions) require no installation and no terminal commands; just open the HTML file.
- Task 3 generates a new MIDI file every time you run it. The output file is always named generated_music.mid and is saved in the task3 folder.
- Task 4 requires a working camera or a video file. It does not work with image files.
- The MyMemory API used in Task 1 is free but rate-limited to approximately 1000 words per day per IP address without registration. For heavier use, register a free account at mymemory.translated.net.
