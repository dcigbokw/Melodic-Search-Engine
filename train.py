from music21 import corpus, chord, interval, pitch
import pickle

MATRIX_FILE = "bach_matrices.pkl"

def build_matrices():
    print("Training AI: Transposing corpus to C Major / A Minor to densify Markov states...")
    transition_counts = {}
    transition_counts_2nd_order = {}
    
    # We will process 150 chorales to balance training time and matrix density
    bach_bundles = corpus.getComposer('bach')[:500]
    
    for idx, score_path in enumerate(bach_bundles):
        print(f"Processing {idx+1}/{len(bach_bundles)}: {score_path}")
        score = corpus.parse(score_path)
        
        # 1. NORMALIZE THE KEY (Fixes the Sparse Data!)
        key = score.analyze('key')
        target_pitch = pitch.Pitch('C') if key.mode == 'major' else pitch.Pitch('A')
        
        # Calculate the distance from the original key to C/A and transpose it
        transpose_interval = interval.Interval(key.tonic, target_pitch)
        transposed_score = score.transpose(transpose_interval)
        
        # 2. EXTRACT CHORDS
        raw_chords = transposed_score.chordify().flatten().getElementsByClass(chord.Chord)
        clean_chords = []
        
        for c in raw_chords:
            midi_array = [int(p.ps) for p in c.pitches]
            if len(midi_array) == 4:
                midi_array.reverse()
                clean_chords.append(tuple(midi_array))
                
        # 3. BUILD 1ST-ORDER COUNTS
        for i in range(len(clean_chords) - 1):
            curr = clean_chords[i]
            nxt = clean_chords[i + 1]
            if curr not in transition_counts: transition_counts[curr] = {}
            transition_counts[curr][nxt] = transition_counts[curr].get(nxt, 0) + 1
            
        # 4. BUILD 2ND-ORDER COUNTS
        for i in range(len(clean_chords) - 2):
            state = (clean_chords[i], clean_chords[i + 1])
            nxt = clean_chords[i + 2]
            if state not in transition_counts_2nd_order: transition_counts_2nd_order[state] = {}
            transition_counts_2nd_order[state][nxt] = transition_counts_2nd_order[state].get(nxt, 0) + 1

    print("\nConverting counts to probabilities...")
    
    # 5. CONVERT TO PROBABILITIES
    transition_matrix = {}
    for curr, nxt_dict in transition_counts.items():
        total = sum(nxt_dict.values())
        transition_matrix[curr] = {k: v / total for k, v in nxt_dict.items()}
        
    transition_matrix_2nd = {}
    for state, nxt_dict in transition_counts_2nd_order.items():
        total = sum(nxt_dict.values())
        transition_matrix_2nd[state] = {k: v / total for k, v in nxt_dict.items()}

    # 6. SERIALIZE AND SAVE
    with open(MATRIX_FILE, 'wb') as f:
        pickle.dump({
            "first_order": transition_matrix,
            "second_order": transition_matrix_2nd
        }, f)
        
    print(f"\nSuccess! Dense models saved to {MATRIX_FILE}. The AI is ready for production.")

if __name__ == "__main__":
    build_matrices()