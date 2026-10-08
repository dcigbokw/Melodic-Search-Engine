# 🎵 Co-Creative Music AI Engine & Multi-Composer Melodic Search

A high-performance backend and interactive front end for **human–AI co-creative music making**. The project combines an algorithmic SATB composer (Markov chains + pruned DFS backtracking) with a multi-dimensional approximate search engine, trained on a six-composer corpus: **Bach, Beethoven, Chopin, Tchaikovsky, Mozart, and Handel**.

It is deployed as a REST API and demonstrates scalable system design, algorithmic optimization, and strict test-driven development.

---

## 🚀 Core Features

### 1. The Co-Creative AI Engine (Core Architecture)

* **Turn-Based Handoffs (Mixed-Initiative UI):** A collaborative SATB workspace where the user inputs/edits notes and the AI generates the next sequence.
* **Combinatorial Creativity (Multi-Model Blending):** Backend corpus support for 6 composers (Bach, Beethoven, Chopin, Tchaikovsky, Mozart, Handel).
* **Dynamic Style Weighting UI:** A percentage-based slider interface that mathematically interpolates the Markov matrices in real time.
* **Robust Input Parsing & Rest Handling:** A custom regex note parser (`note_parser.py`) that is resilient against missing octaves, random spaces, unicode accidentals (♭, ♯), and explicit rests. Accepts scientific pitch notation (`"C4"`, `"F#4"`, `"Bb3"`) or raw MIDI integers (`"60"`).
* **Asynchronous Asset Management:** Memory-safe temporary `.mid` file generation with non-blocking, delayed cleanup routines to prevent local disk bloat and file-lock crashes.

### 2. Advanced Approximate Search (Multi-Dimensional)

* **Visual Sheet Music Results:** Raw JSON payloads are replaced with dynamically rendered sheet music snippets (`<midi-visualizer type="staff">`).
* **Horizontal Search:** Traces a single melody over time across different composers using trigram indexing.
* **Algorithm Experimentation:** Dynamic Time Warping (DTW) is fully implemented and benchmarked against Levenshtein Edit Distance on the backend.
* **Selectable Playback Instrumentation:** On-the-fly injection of music21 MIDI program changes (Harpsichord, Strings, Pipe Organ, etc.) for historically accurate playback in search results.
* **Composer & Distance Filtering:** Configurable UI thresholds to restrict search spaces and distance tolerances.
* **"Sounds Like This" Queries:** Transposition-tolerant search, achieved by comparing horizontal melodic intervals rather than exact pitches.
* **Vertical (Harmonic) Search:** Extracts and searches for specific vertical chord progressions and exact voicings using bass-relative spacing calculation, housed in a dedicated UI tab.
* **Dual-Mode Harmonic Search:** A toggle for *Exact Chord Pitch* vs. *Transposition-Invariant (relative)* matching in the Vertical tab.
* **Simultaneous SATB Playback Rendering:** The `/search` endpoint compiles vertical matches into single 4-voice block chords (capped at 4 notes) for simultaneous Tone.js playback.
* **Selectable Playback Style:** A UI toggle to render vertical harmonic matches as either a simultaneous block chord or a sequential arpeggio.

### 3. Controllable Generation System & Constraints (Phase 12)

* **Constraint-Guided Co-Creation & Pre-Flight Auditing:** The `/extend` endpoint was refactored so the AI generates candidates, passes them through strict DFS rules, prunes invalid ones, and dynamically evaluates/rejects invalid human edits before generating.
* **Strict Error Handling:** No silent failures. Specific HTTP 400/500 errors are raised (e.g., `"Invalid note: H4"` or `"Rule violation"`).
* **Fuzzy Fallback Smoothing Algorithm:** Multi-layered fallback logic that matches outer voices or melody lines so the Markov engine never freezes on out-of-key or heavily altered chords missing from the exact historical training data.

### 4. Expanded Music Theory & Expression

* **Musical Instructions (Expression Engine):** UI dropdowns for dynamics (velocity) and tempo (BPM) that physically alter the generated MIDI.

---

## 🧠 Algorithmic Optimization (Big-O)

The generative engine uses a **Pruned DFS Backtracking** algorithm to tame the exponential time complexity of brute-force music generation.

* **Time Complexity:** A naive brute-force approach testing all combinations of a chord sequence with branching factor `k` yields **O(k^n)**. This engine applies early pruning, discarding invalid harmonic branches immediately instead of generating and checking every combination. `benchmark.py` runs an honest brute-force (via `itertools.product`) against the pruned engine on a small sequence length so the comparison actually completes.
* **Space Complexity:** DFS backtracking stores only the current active path, giving a lean **O(n)** space footprint.
* **Search Complexity:** The melodic search engine uses a trigram inverted index to narrow the corpus to candidate phrases in roughly O(1) lookups per trigram, then runs the expensive distance metric (Levenshtein, O(n·m); or DTW) only on that narrowed set rather than on every phrase in the corpus.

---

## 🛠️ Tech Stack

* **Language:** Python 3
* **API Framework:** FastAPI, Uvicorn
* **Music Processing:** music21
* **Front End Playback/Rendering:** Tone.js, `<midi-visualizer>` staff rendering
* **Testing & Validation:** pytest, httpx (FastAPI `TestClient`)

---

## 💻 Local Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/dcigbokw/melodic-search-engine.git
   cd melodic-search-engine
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models (one-time, offline step)**

   These scripts parse the corpus, transpose everything to a common key (C major / A minor) to densify the Markov states, and serialize the results to disk. Re-run only if you want to retrain on a different corpus or delete the cached files.
   ```bash
   python train.py          # builds bach_matrices.pkl (chord Markov chains)
   python build_index.py    # builds search_index.pkl (melodic search trigram index)
   ```

4. **Run the API server**
   ```bash
   uvicorn main:app --reload
   ```

5. **Access the documentation**

   Open <http://127.0.0.1:8000/docs> to interact with the endpoints via the Swagger UI.

---

## 🔌 API Overview

* **`POST /generate`** – Generates a 4-part chorale and returns a playable MIDI file.
* **`POST /extend`** – Co-creative continuation: audits the human's input against the DFS rules, rejects invalid edits with explicit errors, then generates the next sequence in the blended style.
* **`POST /search`** – Horizontal (melodic) and vertical (harmonic) approximate search across the corpus, with composer and distance filters, selectable distance algorithm (Levenshtein / DTW), instrumentation, and playback style.

Notes can be given as scientific pitch notation or raw MIDI integers. A melody needs at least 4 notes so it can be reduced to at least one trigram for the index lookup.

---

## 🧪 Testing

The pytest suite covers the rules engine, the chord generator (including dynamic key detection), the rhythm/passing-tone engine, note parsing, the search pipeline, and the FastAPI endpoints (with heavy corpus-dependent calls mocked out):
```bash
pytest
```