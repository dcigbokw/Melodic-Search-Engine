from music21 import corpus, chord
import pickle
import os

INDEX_FILE = "search_index.pkl"
WINDOW_SIZE = 4 # Size of the searchable musical chunks

def build_multidimensional_index():
    print("Building the Multi-Corpus Search Index...")
    composers = ['bach', 'beethoven', 'chopin', 'mozart', 'tchaikovsky', 'handel']
    database = {}
    phrase_id = 0
    
    for comp in composers:
        print(f"\nIndexing corpus: {comp.upper()}")
        bundles = corpus.getComposer(comp) 
        
        for idx, score_path in enumerate(bundles):
            try:
                score = corpus.parse(score_path)
                # Compress all instruments into vertical block chords
                chordified = score.chordify()
                
                # Filter out single notes (only keep actual chords with 2+ notes)
                score_chords = [c for c in chordified.flatten().getElementsByClass(chord.Chord) if len(c.pitches) >= 2]
                
                # Convert to raw MIDI integer arrays
                midi_sequence = [[p.midi for p in c.pitches] for c in score_chords]
                
                # Create sliding windows of data
                for i in range(len(midi_sequence) - WINDOW_SIZE + 1):
                    window = midi_sequence[i : i + WINDOW_SIZE]
                    
                    # 1. Exact Mode (Raw Notes)
                    raw_notes = [note for chord_block in window for note in chord_block]
                    
                    # 2. Horizontal Mode (Melodic Intervals of the Soprano Voice)
                    melody_line = [max(chord_block) for chord_block in window]
                    horizontal_intervals = [melody_line[j] - melody_line[j-1] for j in range(1, len(melody_line))]
                    
                    # 3. Vertical Mode (Bass-Relative Harmonic Intervals of the first chord)
                    first_chord = sorted(window[0])
                    bass = first_chord[0]
                    vertical_intervals = [n - bass for n in first_chord]
                    exact_vertical_notes = first_chord
                    
                    # Save to database
                    database[phrase_id] = {
                        "title": score.metadata.title or os.path.basename(str(score_path)),
                        "composer": comp,
                        "raw_notes": raw_notes,
                        "horizontal_intervals": horizontal_intervals,
                        "vertical_intervals": vertical_intervals,
                        "exact_vertical_notes": exact_vertical_notes 
                    }
                    phrase_id += 1
            except Exception as e:
                pass # Skip files that fail to parse gracefully

    print("\nSaving Multi-Corpus Index to disk...")
    with open(INDEX_FILE, 'wb') as f:
        # We save an empty inverted index to prevent breaking your search engine's load step
        pickle.dump({"database": database, "inverted_index": {}}, f)
        
    print(f"Success! Indexed {len(database)} phrases.")

if __name__ == "__main__":
    build_multidimensional_index()