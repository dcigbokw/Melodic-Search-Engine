import random
import pickle
import os
from rules_engine import check_parallel_motion, check_crossing_and_spacing
from rules_engine import check_leading_tone_resolution

# ==========================================
# PRODUCTION LOADING PHASE
# ==========================================
MATRIX_FILE = "multi_composer_matrices.pkl"

all_corpora = {}
transition_matrix = {}
transition_matrix_2nd_order = {}

try:
    with open(MATRIX_FILE, 'rb') as f:
        all_corpora = pickle.load(f)
        # Default fallback to Bach for your basic /generate endpoint
        transition_matrix = all_corpora.get("bach", {}).get("first_order", {})
        transition_matrix_2nd_order = all_corpora.get("bach", {}).get("second_order", {})
except FileNotFoundError:
    print(f"WARNING: {MATRIX_FILE} not found. (Safe to ignore if running mock tests!)")

def is_valid_transition(chord_a, chord_b, tonic_pc=0):
    """
    Acts as the Evaluator. Takes two full chords and ensures moving 
    between them doesn't break your Phase 2 counterpoint rules.
    """
    if not check_crossing_and_spacing(chord_b[0], chord_b[1], chord_b[2],chord_b[3]):
        return False
    
    # 1. Check Leading Tone Resolution for all 4 voices
    for i in range(4):
        if not check_leading_tone_resolution(chord_a[i], chord_b[i], tonic_pc=tonic_pc):
            return False
            
    # 2. Check Parallel Motion for all 6 possible pairs of voices
    for i in range(4):
        for j in range(i + 1, 4):
            if not check_parallel_motion(chord_a[i], chord_b[i], chord_a[j], chord_b[j]):
                return False
                
    return True

# ==========================================
# FUZZY FALLBACK LOGIC
# ==========================================
def get_fuzzy_fallback_candidates(current_chord):
    """
    If an exact 4-note chord is missing from the dataset (e.g., heavy flats/sharps),
    find candidates by matching the outer voices or just the melody.
    """
    candidates = {}
    
    # Fallback Level 1: Match Outer Voices (Soprano and Bass)
    for known_chord, transitions in transition_matrix.items():
        if known_chord[0] == current_chord[0] and known_chord[3] == current_chord[3]:
            for next_chord, prob in transitions.items():
                candidates[next_chord] = candidates.get(next_chord, 0) + prob
                
    # Fallback Level 2: Match Soprano (Melody) only if Outer Voices fail
    if not candidates:
        for known_chord, transitions in transition_matrix.items():
            if known_chord[0] == current_chord[0]:
                for next_chord, prob in transitions.items():
                    candidates[next_chord] = candidates.get(next_chord, 0) + prob
                    
    return candidates

def get_next_candidates(current_state, top_k=8):
    """Retrieves the most likely next chords, utilizing N-order logic and fuzzy fallbacks."""
    # Try 2nd-order exact match first
    if len(current_state) == 2:
        if current_state in transition_matrix_2nd_order:
            candidates = transition_matrix_2nd_order[current_state]
            return sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
        current_chord = current_state[1]
    else:
        current_chord = current_state[0]

    # Try 1st-order exact match
    if current_chord in transition_matrix:
        candidates = transition_matrix[current_chord]
        return sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Engage Fuzzy Fallback if exact match fails
    fuzzy_candidates = get_fuzzy_fallback_candidates(current_chord)
    if fuzzy_candidates:
        return sorted(fuzzy_candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Total failure
    return []

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
    
    # Retrieve candidates smoothly scaling from 2nd order down to fuzzy matches
    top_k_predictions = get_next_candidates(current_state, top_k)
    
    if not top_k_predictions:
        state["retries"] += 1
        return None
    
    valid_options = []
    for cand_chord, prob in top_k_predictions:
        if is_valid_transition(song[-1], cand_chord, tonic_pc):
            valid_options.append(cand_chord)
            
    # If strict rules reject everything, force a fuzzy connection rather than freezing
    if not valid_options and top_k_predictions:
        valid_options = [top_k_predictions[0][0]]
            
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
def compose_chorale_2nd_order(start_chord, num_chords=16, top_k=8, tonic_pc=None):
    """
    Main entry point for chord generation.
    Includes a Cold-Start Backtracking loop for Chord 2.
    """
    # Auto-detect the home key by looking at the bass note of your starting chord!
    if tonic_pc is None:
        tonic_pc = start_chord[3] % 12

    # Engage fuzzy logic on the start chord as well!
    top_k_predictions = get_next_candidates((start_chord,), top_k)
    
    if not top_k_predictions:
        print(f"Error: Start chord {start_chord} yielded no candidates even with fuzzy fallback.")
        return [start_chord]

    # 1. Gather all valid candidates for Chord 2
    valid_candidates = []
    for cand_chord, prob in top_k_predictions:
        if is_valid_transition(start_chord, cand_chord, tonic_pc):
            valid_candidates.append((cand_chord, prob))
            
    if not valid_candidates:
        valid_candidates = [top_k_predictions[0]] # Fallback safety
        
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
            
    print("\nCritical Failure: Exhausted all Chord 2 candidates without finishing.")
    return [start_chord]

start_c_major = (72, 67, 60, 48)
if __name__ == "__main__":
    generated_song = compose_chorale_2nd_order(start_c_major, num_chords=16, top_k=8)
    
    print("\n--- GENERATED CHORALE ---")
    if generated_song:
        for i, chord in enumerate(generated_song):
            print(f"Chord {i+1}: {chord}")
    else:
        print("Failed to generate a sequence.")