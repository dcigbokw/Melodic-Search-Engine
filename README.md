# 🎵 Co-Creative Music AI Engine & Multi-Composer Melodic Search

A high-performance backend and interactive front end for **human–AI co-creative music making**. The project combines an algorithmic SATB composer (Markov chains + pruned DFS backtracking) with a multi-dimensional approximate search engine, trained on a six-composer corpus: **Bach, Beethoven, Chopin, Tchaikovsky, Mozart, and Handel**. 

It is deployed as a REST API and demonstrates scalable system design, algorithmic optimization, rigorous empirical benchmarking, and strict test-driven development.

---

## 🚀 Core Features

### 1. The Co-Creative AI Engine (Core Architecture)
* **Turn-Based Handoffs (Mixed-Initiative UI):** A collaborative SATB workspace where the user inputs/edits notes and the AI generates the next sequence.
* **Combinatorial Creativity (Multi-Model Blending):** Backend corpus support for 6 distinct historical composers.
* **Dynamic Style Weighting UI:** A percentage-based slider interface that mathematically interpolates the Markov matrices in real time.
* **Robust Input Parsing & Rest Handling:** A custom regex note parser (`note_parser.py`) resilient against missing octaves, random spaces, unicode accidentals (♭, ♯), and explicit rests. Accepts scientific pitch notation (`"C4"`, `"F#4"`, `"Bb3"`) or raw MIDI integers (`"60"`).
* **Asynchronous Asset Management:** Memory-safe temporary `.mid` file generation with non-blocking, delayed cleanup routines to prevent local disk bloat and file-lock crashes.

### 2. Advanced Approximate Search (Multi-Dimensional)
* **Visual Sheet Music Results:** Raw JSON payloads are replaced with dynamically rendered sheet music snippets (`<midi-visualizer type="staff">`).
* **Horizontal Search:** Traces a single melody over time across different composers using O(1) trigram indexing.
* **Vertical (Harmonic) Search:** Extracts and searches for specific vertical chord progressions and exact voicings using bass-relative spacing calculation, housed in a dedicated UI tab.
* **Dual-Mode Harmonic Search:** Toggle between *Exact Chord Pitch* and *Transposition-Invariant (relative)* matching in the Vertical tab.
* **Algorithm Experimentation:** Dynamic Time Warping (DTW) is fully implemented and benchmarked against Levenshtein Edit Distance on the backend.
* **Simultaneous SATB Playback Rendering:** The `/search` endpoint dynamically compiles vertical matches into single 4-voice block chords (capped at 4 notes) for simultaneous Tone.js playback.
* **Selectable Playback Style:** A UI toggle allowing users to choose how vertical harmonic matches are rendered (either as a simultaneous block chord or a sequential arpeggio).
* **Selectable Playback Instrumentation:** On-the-fly injection of music21 MIDI program changes (Harpsichord, Strings, Pipe Organ, etc.) for historically accurate playback.
* **Composer & Distance Filtering:** Configurable UI thresholds to restrict search spaces and distance tolerances.

### 3. Controllable Generation System & Constraints
* **Constraint-Guided Co-Creation & Pre-Flight Auditing:** The `/extend` endpoint generates candidates, passes them through strict DFS rules, prunes invalid ones, and dynamically evaluates/rejects invalid human edits before generating.
* **Strict Error Handling:** Eliminates silent failures by throwing specific HTTP 400/500 errors (e.g., `"Invalid note: H4"` or `"Rule violation"`).
* **Fuzzy Fallback Smoothing Algorithm:** Multi-layered fallback logic that matches outer voices or melody lines to prevent the Markov engine from freezing when encountering out-of-key or heavily altered chords missing from the historical training data.

### 4. Expanded Music Theory & Expression
* **Musical Instructions (Expression Engine):** UI dropdowns for dynamics (velocity) and tempo (BPM) that physically alter the generated MIDI output.

### 5. Quantitative Evaluation Suite & Academic Polish
* **Algorithmic Efficiency Benchmarking:** Automated simulations comparing the state-dependent DFS search engine against honest brute-force baselines to track nodes explored, max depth, pruning ratio, and latency.
* **Quantitative Creativity Metrics:** Empirical validation of generated sequences measuring:
  * **Negative Log-Likelihood (NLL) & Perplexity** against a strict held-out test split using raw event-weighted transition counts.
  * **Phrase Memorization (LCS)** to measure contiguous training overlap and prevent strict plagiarism.
  * **Nearest-Neighbor Novelty** using length-normalized Levenshtein distance against the training corpus.
  * **Structural Diversity** across generated collections (measuring distinct states and bigram entropy).
* **Search Retrieval Validation:** Mean Reciprocal Rank (MRR) tracking for melodic search queries against labeled ground truths.
* **Rhythmic Meter Validation:** Strict mathematical boundary checks confirming multi-meter alignment (e.g., 4/4, 3/4, 6/8).

---

## 🧠 Algorithmic Optimization (Big-O)

The generative engine uses a **Pruned DFS Backtracking** algorithm to tame the exponential time complexity of brute-force music generation.

* **Time Complexity:** A naive brute-force approach testing all combinations of a chord sequence with branching factor $k$ yields $O(k^n)$. This engine applies early pruning, discarding invalid harmonic branches immediately instead of generating and checking every combination. The built-in benchmark suite runs an honest brute-force (via `itertools.product`) against the pruned engine on a small sequence length so the algorithmic savings can be directly measured.
* **Space Complexity:** DFS backtracking stores only the current active path, giving a lean $O(n)$ space footprint.
* **Search Complexity:** The melodic search engine uses an inverted index to narrow the corpus to candidate phrases in roughly $O(1)$ lookups per trigram, then runs the expensive distance metric (Levenshtein, $O(n \cdot m)$; or DTW) only on that narrowed candidate set rather than scanning every phrase in the entire database.

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
   git clone [https://github.com/dcigbokw/melodic-search-engine.git](https://github.com/dcigbokw/melodic-search-engine.git)
   cd melodic-search-engine
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models (one-time, offline step)**

  These scripts parse the corpus, perform a 70/15/15 train/evaluator/test split, transpose everything to a common key (C major / A minor) to densify the Markov states, and build the search index.
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
* **`POST /harmonize`** – Top-down harmony generation preserving a user-provided soprano melody line.
* **`POST /search`** – Horizontal (melodic) and vertical (harmonic) approximate search across the corpus, with composer and distance filters, selectable distance algorithm (Levenshtein / DTW), instrumentation, and playback style.

Notes can be given as scientific pitch notation or raw MIDI integers. A melody needs at least 4 notes so it can be reduced to at least one trigram for the index lookup.

---

## 🧪 Testing & Research Evaluation

The pytest suite covers the rules engine, the chord generator (including dynamic key detection), the rhythm/passing-tone engine, note parsing, the search pipeline, and the FastAPI endpoints (with heavy corpus-dependent calls mocked out):
```bash
pytest
```
To generate the empirical research data for academic evaluation, run the standalone benchmark suites locally:
```bash
python benchmark.py             # Algorithmic efficiency, NLL test-split evals, constraints, and MRR search
python creativity_metrics.py    # Cross-composition novelty, memorization, and structural diversity
```