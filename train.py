from music21 import corpus, chord, interval, pitch
import pickle

MATRIX_FILE = "multi_composer_matrices.pkl"

def build_multi_matrices():
    composers = ['bach', 'beethoven', 'chopin', 'mozart', 'tchaikovsky', 'handel']
    all_corpora = {}
    
    for comp in composers:
        print(f"\n--- Training Corpus: {comp.upper()} ---")
        transition_counts = {}
        transition_counts_2nd_order = {}
        transition_counts_3rd_order = {} # NEW: 3rd order counts
        
        # 1. Fetch built-in scores 
        bundles = corpus.getComposer(comp)
        
        for idx, score_path in enumerate(bundles):
            print(f"Processing {idx+1}/{len(bundles)}: {score_path}")
            try:
                score = corpus.parse(score_path)
                
                # 2. Normalize the key to C Major / A Minor
                key = score.analyze('key')
                target_pitch = pitch.Pitch('C') if key.mode == 'major' else pitch.Pitch('A')
                transpose_interval = interval.Interval(key.tonic, target_pitch)
                transposed_score = score.transpose(transpose_interval)
                
                # 3. Extract 4-part chords
                raw_chords = transposed_score.chordify().flatten().getElementsByClass(chord.Chord)
                clean_chords = []
                
                for c in raw_chords:
                    midi_array = [int(p.ps) for p in c.pitches]
                    if len(midi_array) == 4:
                        midi_array.reverse()
                        clean_chords.append(tuple(midi_array))
                        
                # 4. Build 1st-Order Counts
                for i in range(len(clean_chords) - 1):
                    curr = clean_chords[i]
                    nxt = clean_chords[i + 1]
                    if curr not in transition_counts: transition_counts[curr] = {}
                    transition_counts[curr][nxt] = transition_counts[curr].get(nxt, 0) + 1
                    
                # 5. Build 2nd-Order Counts
                for i in range(len(clean_chords) - 2):
                    state = (clean_chords[i], clean_chords[i + 1])
                    nxt = clean_chords[i + 2]
                    if state not in transition_counts_2nd_order: transition_counts_2nd_order[state] = {}
                    transition_counts_2nd_order[state][nxt] = transition_counts_2nd_order[state].get(nxt, 0) + 1
                    
                # NEW: Build 3rd-Order Counts
                for i in range(len(clean_chords) - 3):
                    state = (clean_chords[i], clean_chords[i + 1], clean_chords[i + 2])
                    nxt = clean_chords[i + 3]
                    if state not in transition_counts_3rd_order: transition_counts_3rd_order[state] = {}
                    transition_counts_3rd_order[state][nxt] = transition_counts_3rd_order[state].get(nxt, 0) + 1
                    
            except Exception as e:
                print(f"Skipping {score_path} due to parsing error: {e}")

        print(f"Converting {comp.capitalize()} counts to probabilities...")
        
        # 6. Convert to Probabilities
        transition_matrix = {}
        for curr, nxt_dict in transition_counts.items():
            total = sum(nxt_dict.values())
            transition_matrix[curr] = {k: v / total for k, v in nxt_dict.items()}
            
        transition_matrix_2nd = {}
        for state, nxt_dict in transition_counts_2nd_order.items():
            total = sum(nxt_dict.values())
            transition_matrix_2nd[state] = {k: v / total for k, v in nxt_dict.items()}
            
        # NEW: Convert 3rd-Order to Probabilities
        transition_matrix_3rd = {}
        for state, nxt_dict in transition_counts_3rd_order.items():
            total = sum(nxt_dict.values())
            transition_matrix_3rd[state] = {k: v / total for k, v in nxt_dict.items()}

        # Store in the global dictionary
        all_corpora[comp] = {
            "first_order": transition_matrix,
            "second_order": transition_matrix_2nd,
            "third_order": transition_matrix_3rd
        }

    # 7. Serialize and Save
    with open(MATRIX_FILE, 'wb') as f:
        pickle.dump(all_corpora, f)
        
    print(f"\nSuccess! Multi-composer models saved to {MATRIX_FILE}.")

if __name__ == "__main__":
    build_multi_matrices()