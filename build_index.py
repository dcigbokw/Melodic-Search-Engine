from music21 import corpus, chord
import pickle
import os
from collections import defaultdict

INDEX_FILE = "search_index.pkl"
WINDOW_SIZE = 4 

def build_multidimensional_index():
    print("Building the Multi-Tier Search Index...")
    composers = ['bach', 'beethoven', 'chopin', 'mozart', 'tchaikovsky', 'handel']
    database = {}
    inverted_index = defaultdict(list)
    phrase_id = 0
    
    for comp in composers:
        print(f"\nIndexing corpus: {comp.upper()}")
        bundles = corpus.getComposer(comp) 
        
        for idx, score_path in enumerate(bundles):
            try:
                score = corpus.parse(score_path)
                chordified = score.chordify()
                score_chords = [c for c in chordified.flatten().getElementsByClass(chord.Chord) if len(c.pitches) >= 2]
                if not score_chords: continue
                
                title = score.metadata.title or os.path.basename(str(score_path))
                
                # --- TIER 1: HORIZONTAL (Full Continuous Melody) ---
                melody_line = [max(c.pitches).midi for c in score_chords]
                horizontal_intervals = [melody_line[j] - melody_line[j-1] for j in range(1, len(melody_line))]
                
                database[phrase_id] = {
                    "type": "horizontal",
                    "title": title,
                    "composer": comp,
                    "raw_notes": melody_line,
                    "horizontal_intervals": horizontal_intervals
                }
                
                # Trigram offset posting for O(1) local window lookup
                for i in range(len(horizontal_intervals) - 2):
                    trigram = tuple(horizontal_intervals[i : i + 3])
                    inverted_index[trigram].append((phrase_id, i))
                    
                phrase_id += 1
                
                # --- TIER 2: VERTICAL (4-Chord Blocks) ---
                midi_sequence = [[p.midi for p in c.pitches] for c in score_chords]
                for i in range(len(midi_sequence) - WINDOW_SIZE + 1):
                    window = midi_sequence[i : i + WINDOW_SIZE]
                    raw_notes = [note for chord_block in window for note in chord_block]
                    first_chord = sorted(window[0])
                    bass = first_chord[0]
                    vertical_intervals = tuple([n - bass for n in first_chord])
                    exact_vertical_notes = tuple(first_chord)
                    
                    database[phrase_id] = {
                        "type": "vertical",
                        "title": title,
                        "composer": comp,
                        "raw_notes": raw_notes,
                        "vertical_intervals": list(vertical_intervals),
                        "exact_vertical_notes": list(exact_vertical_notes) 
                    }
                    phrase_id += 1
                    
            except Exception:
                pass 

    print("\nSaving Multi-Tier Index to disk...")
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump({"database": database, "inverted_index": dict(inverted_index)}, f)
        
    print(f"Success! Indexed {len(database)} entities.")

if __name__ == "__main__":
    build_multidimensional_index()