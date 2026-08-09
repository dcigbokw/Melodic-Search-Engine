import time
import itertools
from chord_generator import compose_chorale_2nd_order, is_valid_transition, transition_matrix

def brute_force_simulation(start_chord, num_chords, top_k):
    """
    An honest brute-force: Actually builds the arrays and checks the rules.
    """
    # Get top_k valid next chords from the 1st-order matrix to form our "pool" of choices
    if start_chord not in transition_matrix:
        return 0

    # Sort the 1st-order predictions by probability (highest first)
    raw_predictions = transition_matrix[start_chord]
    sorted_pool = sorted(raw_predictions.items(), key=lambda x: x[1], reverse=True)

    # Grab just the chord tuples for the top_k candidates
    pool = [chord for chord, prob in sorted_pool[:top_k]]
        
    
    
    # itertools.product generates EVERY possible combination of these chords
    # for the remaining length of the song (num_chords - 1)
    all_possible_songs = itertools.product(pool, repeat=(num_chords - 1))
    
    paths_explored = 0
    valid_songs_found = 0
    
    for chord_sequence in all_possible_songs:
        paths_explored += 1
        
        # Build the full song array
        song = [start_chord] + list(chord_sequence)
        
        # Check every single transition in the generated song against the rules
        is_valid = True
        for i in range(len(song) - 1):
            if not is_valid_transition(song[i], song[i+1], tonic_pc=0):
                is_valid = False
                break # Failed the rules!
                
        if is_valid:
            valid_songs_found += 1
            
    return paths_explored

def run_benchmarks():
    print("==========================================")
    print("      HONEST ALGORITHMIC BENCHMARK        ")
    print("==========================================\n")
    
    target_length = 5  # Small N so brute force can actually finish
    branching_factor = 4
    start_c_major = (72, 67, 60, 48)
    
    print(f"Target Length: {target_length} chords")
    print(f"Branching Factor (Top-K): {branching_factor}\n")
    
    # ---------------------------------------------------------
    # TEST 1: THE BRUTE-FORCE APPROACH
    # ---------------------------------------------------------
    print("1. Testing Honest Brute-Force...")
    start_time = time.perf_counter()
    
    nodes_explored = brute_force_simulation(start_c_major, target_length, branching_factor)
    
    bf_duration = time.perf_counter() - start_time
    print(f"   -> Explored {nodes_explored:,} combinations.")
    print(f"   -> Time elapsed: {bf_duration:.4f} seconds.\n")
    
    # ---------------------------------------------------------
    # TEST 2: YOUR EARLY-PRUNING ENGINE
    # ---------------------------------------------------------
    print("2. Testing Your Pruned DFS Engine...")
    start_time = time.perf_counter()
    
    result = compose_chorale_2nd_order(start_c_major, num_chords=target_length, top_k=branching_factor)
    pruned_duration = time.perf_counter() - start_time
    
    if len(result) == target_length:
        print(f"   -> Success! Valid {len(result)}-chord sequence generated.")
        print(f"   -> Time elapsed: {pruned_duration:.4f} seconds.\n")
    else:
        print(f"   -> Failure: Only generated {len(result)} chords.\n")

if __name__ == "__main__":
    run_benchmarks()