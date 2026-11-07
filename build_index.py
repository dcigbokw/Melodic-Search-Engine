from music21 import corpus, chord
import pickle
import os
from collections import defaultdict

INDEX_FILE = "search_index.pkl"
WINDOW_SIZE = 4 # 4 notes produce 3 intervals (a trigram)

def build_multidimensional_index():
    print("Building the Multi-Corpus Search Index...")
    composers = ['bach', 'beethoven', 'chopin', 'mozart', 'tchaikovsky', 'handel']
    database = {}
    inverted_index = defaultdict(set)
    phrase_id = 0
    
    for comp in composers:
        print(f"\nIndexing corpus: {comp.upper()}")
        bundles = corpus.getComposer(comp) 
        
        for idx, score_path in enumerate(bundles):
            try:
                score = corpus.parse(score_path)
                chordified = score.chordify()
                score_chords = [c for c in chordified.flatten().getElementsByClass(chord.Chord) if len(c.pitches) >= 2]
                midi_sequence = [[p.midi for p in c.pitches] for c in score_chords]
                
                for i in range(len(midi_sequence) - WINDOW_SIZE + 1):
                    window = midi_sequence[i : i + WINDOW_SIZE]
                    
                    raw_notes = [note for chord_block in window for note in chord_block]
                    
                    # Extract the Soprano line and calculate horizontal intervals
                    melody_line = [max(chord_block) for chord_block in window]
                    horizontal_intervals = tuple([melody_line[j] - melody_line[j-1] for j in range(1, len(melody_line))])
                    
                    first_chord = sorted(window[0])
                    bass = first_chord[0]
                    vertical_intervals = tuple([n - bass for n in first_chord])
                    exact_vertical_notes = tuple(first_chord)
                    
                    # Save to database
                    database[phrase_id] = {
                        "title": score.metadata.title or os.path.basename(str(score_path)),
                        "composer": comp,
                        "raw_notes": raw_notes,
                        "horizontal_intervals": list(horizontal_intervals),
                        "vertical_intervals": list(vertical_intervals),
                        "exact_vertical_notes": list(exact_vertical_notes) 
                    }
                    
                    # POPULATE THE INVERTED INDEX (Trigram -> Set of Phrase IDs)
                    inverted_index[horizontal_intervals].add(phrase_id)
                    
                    phrase_id += 1
            except Exception:
                pass 

    print("\nSaving Multi-Corpus Index to disk...")
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump({"database": database, "inverted_index": dict(inverted_index)}, f)
        
    print(f"Success! Indexed {len(database)} phrases.")

if __name__ == "__main__":
    build_multidimensional_index()