# 🎵 Bach Generative AI & Melodic Search Engine

A high-performance backend architecture featuring a dual-engine system: an Algorithmic AI Composer and a Fuzzy-Search Melodic Engine, trained on the corpus of J.S. Bach. 

This project is deployed as a REST API and demonstrates scalable system design, algorithmic optimization, and strict test-driven development.

## 🚀 Core Features

* **Algorithmic Composer (DFS Backtracking):** Generates 4-part SATB counterpoint. Uses 2nd-Order Markov Chains for stylistic probability and a custom Early-Pruning Depth-First Search to enforce strict music theory rules (parallel motion, leading tones, spacing).
* **Fuzzy-Search Melodic Engine:** Converts standard MIDI notes into interval vectors and performs a sliding-window fuzzy search across the Bach corpus to find melodic matches with a configurable edit distance.
* **Stateless REST API:** Built with FastAPI, decoupling the heavy mathematical engines from the client-facing architecture. Includes a Master Retry Loop to ensure 100% valid JSON payload delivery.

## 🧠 Algorithmic Optimization (Big-O)

The Generative AI uses a **Pruned DFS Backtracking** algorithm to solve the exponential time complexity of brute-force music generation. 

* **Time Complexity:** A naive brute-force approach testing all combinations of a 16-chord sequence with a branching factor of 5 yields **O(k^n)** (152 billion combinations). This engine applies early pruning, destroying invalid harmonic branches instantly and solving the sequence in <0.01 seconds.
* **Space Complexity:** By utilizing DFS backtracking, the engine only stores the current active path in memory, reducing space complexity to an extremely lean **O(n)**.

## 🛠️ Tech Stack
* **Language:** Python 3
* **API Framework:** FastAPI, Uvicorn
* **Music Processing:** music21
* **Testing & Validation:** pytest

## 💻 Local Installation & Usage

1. **Clone the repository**
   ```bash
   git clone [https://github.com/yourusername/melodic-search-engine.git](https://github.com/yourusername/melodic-search-engine.git)
   cd melodic-search-engine

2. **Install Dependencies**
   pip install fastapi uvicorn music21 pytest

3. **Run the API server**
   uvicorn main:app --reload

4. **Access the Documentation**
   Open your browser and navigate to http://127.0.0.1:8000/docs to interact with the endpoints via the Swagger UI.

## 🧪 Testing
The rules engine is fully covered by an automated test suite. To verify the counterpoint mathematics, run:
pytest
