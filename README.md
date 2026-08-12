# 🎵 Bach Generative AI & Melodic Search Engine

A high-performance backend architecture featuring a dual-engine system: an Algorithmic AI Composer and a Fuzzy-Search Melodic Engine, trained on the corpus of J.S. Bach.

This project is deployed as a REST API and demonstrates scalable system design, algorithmic optimization, and strict test-driven development.

## 🚀 Core Features

* **Algorithmic Composer (DFS Backtracking):** Generates 4-part SATB counterpoint. Uses 2nd-Order Markov Chains for stylistic probability and a custom Early-Pruning Depth-First Search to enforce strict music theory rules (parallel motion, leading tones, spacing). The `/generate` endpoint returns a playable `.mid` file, complete with generated rhythms and diatonic passing tones.
* **Fuzzy-Search Melodic Engine:** Converts melodies into interval vectors and searches the Bach corpus for melodic matches within a configurable edit distance, using a trigram inverted index to filter candidates before running the expensive Levenshtein comparison.
* **Offline Training Pipeline:** Markov matrices and the search index are trained/built once, offline, and serialized to disk (`bach_matrices.pkl`, `search_index.pkl`). The API loads them at startup instead of retraining on every launch.
* **Flexible Note Input:** Melodies can be submitted as scientific pitch notation (`"C4"`, `"F#4"`, `"Bb3"`) or raw MIDI integers (`"60"`).
* **Stateless REST API:** Built with FastAPI, decoupling the heavy mathematical engines from the client-facing architecture. Includes a Master Retry Loop to ensure reliable generation.

## 🧠 Algorithmic Optimization (Big-O)

The Generative AI uses a **Pruned DFS Backtracking** algorithm to solve the exponential time complexity of brute-force music generation.

* **Time Complexity:** A naive brute-force approach testing all combinations of a chord sequence with branching factor `k` yields **O(k^n)**. This engine applies early pruning, destroying invalid harmonic branches instantly instead of generating and checking every combination. `benchmark.py` runs an honest brute-force (via `itertools.product`) against the pruned engine on a small sequence length so the comparison actually completes.
* **Space Complexity:** By utilizing DFS backtracking, the engine only stores the current active path in memory, reducing space complexity to an extremely lean **O(n)**.
* **Search Complexity:** The melodic search engine uses a trigram inverted index to narrow the corpus down to candidate phrases in roughly O(1) lookups per trigram, before running Levenshtein distance (O(n·m)) only on that narrowed set — rather than against every phrase in the corpus.

## 🛠️ Tech Stack
* **Language:** Python 3
* **API Framework:** FastAPI, Uvicorn
* **Music Processing:** music21
* **Testing & Validation:** pytest, httpx (FastAPI `TestClient`)

## 💻 Local Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/melodic-search-engine.git
   cd melodic-search-engine
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models (one-time, offline step)**

   These scripts parse the Bach corpus, transpose everything to a common key (C major / A minor) to densify the Markov states, and serialize the results to disk. This only needs to be re-run if you want to retrain on a different corpus size or delete the cached files.
   ```bash
   python train.py          # builds bach_matrices.pkl (chord Markov chains)
   python build_index.py    # builds search_index.pkl (melodic search trigram index)
   ```

4. **Run the API server**
   ```bash
   uvicorn main:app --reload
   ```
   On first launch, if no rhythm matrix is cached in memory yet, the server will also train the rhythm model automatically (a few seconds).

5. **Access the documentation**

   Open your browser and navigate to <http://127.0.0.1:8000/docs> to interact with the endpoints via the Swagger UI.

## 🔌 API Endpoints

### `POST /generate`
Generates a 4-part chorale and returns a playable MIDI file.

```json
{
  "num_chords": 16,
  "top_k": 5,
  "tonic_pc": 0
}
```
`tonic_pc` is a pitch class (0 = C, 1 = C#/Db, ... 11 = B). The engine only starts from a trained chord whose bass note matches this tonic, so the generated chorale actually resolves in the requested key.

### `POST /search`
Fuzzy-searches the Bach corpus for melodic phrases similar to a query melody.

```json
{
  "melody": ["C4", "D4", "E4", "F4"],
  "max_distance": 1
}
```
Notes can be given as scientific pitch notation or raw MIDI integers. A melody needs at least 4 notes so it can be reduced to at least one trigram for the index lookup.

## 🧪 Testing

The project has a full pytest suite covering the rules engine, the chord generator (including dynamic key detection), the rhythm/passing-tone engine, note parsing, the search pipeline, and the FastAPI endpoints themselves (with the heavy corpus-dependent calls mocked out). To run it:
```bash
pytest
```