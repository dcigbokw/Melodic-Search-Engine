from music21 import corpus, chord, interval, pitch
import pickle
import random

MATRIX_FILE = "multi_composer_matrices.pkl"
TRAIN_SEQUENCES_FILE = "training_sequences.pkl"

def process_bundles(bundles):
    """Parses a list of scores and builds 1st, 2nd, and 3rd order matrices."""
    counts_1, counts_2, counts_3 = {}, {}, {}
    raw_sequences = []
    
    for idx, score_path in enumerate(bundles):
        print(f"  Processing {idx+1}/{len(bundles)}: {score_path}")
        try:
            score = corpus.parse(score_path)
            
            # Normalize the key to C Major / A Minor
            key = score.analyze('key')
            target_pitch = pitch.Pitch('C') if key.mode == 'major' else pitch.Pitch('A')
            transpose_interval = interval.Interval(key.tonic, target_pitch)
            transposed_score = score.transpose(transpose_interval)
            
            raw_chords = transposed_score.chordify().flatten().getElementsByClass(chord.Chord)
            clean_chords = []
            
            for c in raw_chords:
                midi_array = [int(p.ps) for p in c.pitches]
                if len(midi_array) == 4:
                    midi_array.reverse()
                    clean_chords.append(tuple(midi_array))
                    
            if clean_chords:
                raw_sequences.append(clean_chords)
                    
            # 1st-Order Counts
            for i in range(len(clean_chords) - 1):
                curr, nxt = clean_chords[i], clean_chords[i + 1]
                if curr not in counts_1: counts_1[curr] = {}
                counts_1[curr][nxt] = counts_1[curr].get(nxt, 0) + 1
                
            # 2nd-Order Counts
            for i in range(len(clean_chords) - 2):
                state, nxt = (clean_chords[i], clean_chords[i + 1]), clean_chords[i + 2]
                if state not in counts_2: counts_2[state] = {}
                counts_2[state][nxt] = counts_2[state].get(nxt, 0) + 1
                
            # 3rd-Order Counts
            for i in range(len(clean_chords) - 3):
                state, nxt = (clean_chords[i], clean_chords[i + 1], clean_chords[i + 2]), clean_chords[i + 3]
                if state not in counts_3: counts_3[state] = {}
                counts_3[state][nxt] = counts_3[state].get(nxt, 0) + 1
                
        except Exception as e:
            print(f"  Skipping {score_path} due to error: {e}")

    # Convert to normalized probabilities (ensuring they sum to 1.0)
    def to_prob(counts):
        matrix = {}
        for state, nxt_dict in counts.items():
            total = sum(nxt_dict.values())
            matrix[state] = {k: v / total for k, v in nxt_dict.items()}
        return matrix

    # Return normalized probabilities AND raw counts for accurate NLL weighting
    return to_prob(counts_1), to_prob(counts_2), to_prob(counts_3), raw_sequences, counts_1, counts_2, counts_3

def build_multi_matrices():
    composers = ['bach', 'beethoven', 'chopin', 'mozart', 'tchaikovsky', 'handel']
    all_corpora = {}
    all_training_sequences = {}
    
    for comp in composers:
        print(f"\n--- Training Corpus: {comp.upper()} ---")
        bundles = list(corpus.getComposer(comp))
        
        # Shuffle deterministically to create consistent splits
        random.seed(42)
        random.shuffle(bundles)
        
        # 3-Way Split: 70% Train, 15% Evaluator, 15% Test
        split_1 = int(len(bundles) * 0.70)
        split_2 = int(len(bundles) * 0.85)
        
        train_bundles = bundles[:split_1]
        eval_bundles = bundles[split_1:split_2]
        test_bundles = bundles[split_2:]
        
        print(f"Building TRAIN matrices ({len(train_bundles)} pieces)...")
        train_matrices = process_bundles(train_bundles)
        
        print(f"Building EVALUATOR matrices ({len(eval_bundles)} pieces)...")
        eval_matrices = process_bundles(eval_bundles)
        
        print(f"Building TEST matrices ({len(test_bundles)} pieces)...")
        test_matrices = process_bundles(test_bundles)

        # Nest under train, evaluator, and test
        all_corpora[comp] = {
            "train": {
                "first_order": train_matrices[0],
                "second_order": train_matrices[1],
                "third_order": train_matrices[2]
            },
            "evaluator": {
                "first_order": eval_matrices[0],
                "second_order": eval_matrices[1],
                "third_order": eval_matrices[2]
            },
            "test": {
                # Use raw counts (indices 4, 5, 6) for accurate NLL weighting
                "first_order": test_matrices[4],
                "second_order": test_matrices[5],
                "third_order": test_matrices[6]
            }
        }
        
        all_training_sequences[comp] = train_matrices[3]

    # Serialize and Save Matrices
    with open(MATRIX_FILE, 'wb') as f:
        pickle.dump(all_corpora, f)
        
    # Serialize and Save Raw Training Sequences (for LCS / Nearest Neighbor)
    with open(TRAIN_SEQUENCES_FILE, 'wb') as f:
        pickle.dump(all_training_sequences, f)
        
    print(f"\nSuccess! Train/Eval/Test split models saved to {MATRIX_FILE}.")
    print(f"Raw training sequences saved to {TRAIN_SEQUENCES_FILE}.")

if __name__ == "__main__":
    build_multi_matrices()