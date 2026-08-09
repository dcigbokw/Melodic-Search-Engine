from music21 import corpus, chord, stream
from rules_engine import check_parallel_motion, check_crossing_and_spacing
from rules_engine import check_leading_tone_resolution
import random

print("Training matrices from scratch (This takes ~20 seconds)...")

# ---------------------------------------------------------
# 1st ORDER TRAINING
# ---------------------------------------------------------
transition_counts = {}
bach_bundles = corpus.getComposer('bach')[:300]

for score_path in bach_bundles:
    score = corpus.parse(score_path)
    raw_chords = score.chordify().flatten().getElementsByClass(chord.Chord)
    clean_chords = []
    
    for c in raw_chords:
        midi_array = [int(p.ps) for p in c.pitches]
        if len(midi_array) == 4:
            midi_array.reverse()
            clean_chords.append(tuple(midi_array))
            
    for i in range(len(clean_chords) - 1):
        current_chord = clean_chords[i]
        next_chord = clean_chords[i + 1]
        
        if current_chord not in transition_counts:
            transition_counts[current_chord] = {}
        if next_chord not in transition_counts[current_chord]:
            transition_counts[current_chord][next_chord] = 0
        transition_counts[current_chord][next_chord] += 1

transition_matrix = {}
for current_chord, next_chords_dict in transition_counts.items():
    transition_matrix[current_chord] = {}
    total_transitions = sum(next_chords_dict.values())
    for next_chord, count in next_chords_dict.items():
        transition_matrix[current_chord][next_chord] = count / total_transitions

# ---------------------------------------------------------
# 2nd ORDER TRAINING
# ---------------------------------------------------------
transition_counts_2nd_order = {}

for score_path in bach_bundles:
    score = corpus.parse(score_path)
    raw_chords = score.chordify().flatten().getElementsByClass(chord.Chord)
    clean_chords = []
    
    for c in raw_chords:
        midi_array = [int(p.ps) for p in c.pitches]
        if len(midi_array) == 4:
            midi_array.reverse()
            clean_chords.append(tuple(midi_array))
            
    for i in range(len(clean_chords) - 2):
        chord_1 = clean_chords[i]
        chord_2 = clean_chords[i + 1]
        chord_3 = clean_chords[i + 2]
        current_state = (chord_1, chord_2)
        
        if current_state not in transition_counts_2nd_order:
            transition_counts_2nd_order[current_state] = {}
        if chord_3 not in transition_counts_2nd_order[current_state]:
            transition_counts_2nd_order[current_state][chord_3] = 0
        transition_counts_2nd_order[current_state][chord_3] += 1

transition_matrix_2nd_order = {}
for current_state, next_chords_dict in transition_counts_2nd_order.items():
    transition_matrix_2nd_order[current_state] = {}
    total_transitions = sum(next_chords_dict.values())
    for next_chord, count in next_chords_dict.items():
        transition_matrix_2nd_order[current_state][next_chord] = count / total_transitions

print("Training Complete! Ready to generate.")
def is_valid_transition(chord_a, chord_b, tonic_pc=0):
    """
    Acts as the 'Evaluator'. Takes two full chords and ensures moving 
    between them doesn't break your Phase 2 counterpoint rules.
    """

    if not check_crossing_and_spacing(chord_b[0], chord_b[1], chord_b[2],chord_b[3]):
        return False
    
    # 1. Check Leading Tone Resolution for all 4 voices
    for i in range(4):
        if not check_leading_tone_resolution(chord_a[i], chord_b[i], tonic_pc=tonic_pc):
            return False
            
    # 2. Check Parallel Motion for all 6 possible pairs of voices
    # Pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    for i in range(4):
        for j in range(i + 1, 4):
            if not check_parallel_motion(chord_a[i], chord_b[i], chord_a[j], chord_b[j]):
                return False

                
    return True


# ==========================================
# THE RECURSIVE ENGINE (DFS Backtracking)
# ==========================================
def compose_recursive(song, num_chords, top_k, tonic_pc=0, state=None):
    # Initialize the shared state on the very first cold-start call
    if state is None:
        state = {"retries": 0, "max_retries": 2000}
        
    # 1. THE SAFETY VALVE: Abort if we've backtracked too many times
    if state["retries"] >= state["max_retries"]:
        return None

    # Base Case: We hit the target length!
    if len(song) == num_chords:
        # Check if the final chord resolves to our dynamic Tonic key
        if song[-1][3] % 12 == tonic_pc:
            return song  
        else:
            state["retries"] += 1  # Count this as a dead end
            return None 
            
    # Figure out our context (the last two chords)
    current_state = (song[-2], song[-1])
    
    if current_state not in transition_matrix_2nd_order:
        state["retries"] += 1
        return None
        
    raw_predictions = transition_matrix_2nd_order[current_state]
    sorted_predictions = sorted(raw_predictions.items(), key=lambda x: x[1], reverse=True)
    top_k_predictions = sorted_predictions[:top_k]
    
    valid_options = []
    for cand_chord, prob in top_k_predictions:
        if is_valid_transition(song[-1], cand_chord, tonic_pc):
            valid_options.append(cand_chord)
            
    random.shuffle(valid_options)
    
    # THE BACKTRACKING LOOP
    for cand_chord in valid_options:
        song.append(cand_chord)
        
        # Pass the state dictionary down into the future!
        result = compose_recursive(song, num_chords, top_k, tonic_pc, state)
        
        if result is not None:
            return result 
            
        song.pop() # Undo the choice
        
        # BUBBLE UP THE ABORT: If the future aborted due to max_retries, stop trying new things!
        if state["retries"] >= state["max_retries"]:
            return None
            
    # If all options fail, increment the dead end counter
    state["retries"] += 1
    return None

# ==========================================
# THE MAIN WRAPPER FUNCTION
# ==========================================
def compose_chorale_2nd_order(start_chord, num_chords=16, top_k=8, tonic_pc=0):
    """
    Main entry point for chord generation.
    Includes a Cold-Start Backtracking loop for Chord 2.
    """
    if start_chord not in transition_matrix:
        print(f"Error: Start chord {start_chord} not found in 1st-Order matrix.")
        return [start_chord]

    predictions = transition_matrix[start_chord]
    
    # 1. Gather all valid candidates for Chord 2
    valid_candidates = []
    for cand_chord, prob in predictions.items():
        if is_valid_transition(start_chord, cand_chord, tonic_pc):
            valid_candidates.append((cand_chord, prob))
            
    if not valid_candidates:
        print("Error: No valid second chords found that satisfy counterpoint rules.")
        return [start_chord]
        
    # Sort candidates by probability (highest first)
    valid_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Shuffle top candidates slightly for creativity while keeping them high-quality
    top_candidates = [c[0] for c in valid_candidates[:top_k]]
    random.shuffle(top_candidates)
    
    # 2. COLD START BACKTRACKING LOOP
    # Try each valid Chord 2 candidate until we find one that leads to a full song!
    for chord_2 in top_candidates:
        song = [start_chord, chord_2]
        
        # Reset retry counter state for each Chord 2 attempt
        state = {"retries": 0, "max_retries": 2000}
        
        final_song = compose_recursive(song, num_chords, top_k, tonic_pc, state)
        
        if final_song is not None:
            return final_song  # Full song successfully generated!
            
        # If we reach here, this specific Chord 2 was a harmonic dead end.
        # The loop will automatically try the next Chord 2 candidate!

    print("\nCritical Failure: Exhausted all Chord 2 candidates without finishing.")
    return [start_chord]


