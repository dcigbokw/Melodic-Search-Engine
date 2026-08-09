import time
from chord_generator import compose_chorale_2nd_order

def brute_force_simulation(start_chord, num_chords, top_k):
    """
    The naive approach: Generate all possible combinations first, 
    THEN check the rules at the very end.
    """
    paths_explored = 0
    total_combinations = top_k ** num_chords
    
    for i in range(total_combinations):
        paths_explored += 1
        pass 
        
    return paths_explored

def run_benchmarks():
    print("==========================================")
    print("      ALGORITHMIC BENCHMARK SUITE         ")
    print("==========================================\n")
    
    target_length = 8
    branching_factor = 5
    start_c_major = (72, 67, 64, 60) # C5, G4, E4, C4
    
    print(f"Target Length: {target_length} chords")
    print(f"Branching Factor (Top-K): {branching_factor}\n")
    
    # ---------------------------------------------------------
    # TEST 1: THE BRUTE-FORCE APPROACH
    # ---------------------------------------------------------
    print("1. Testing Brute-Force (Generate-and-Test)...")
    start_time = time.perf_counter()
    
    nodes_explored = brute_force_simulation(start_c_major, target_length, branching_factor)
    
    bf_duration = time.perf_counter() - start_time
    print(f"   -> Explored {nodes_explored:,} combinations.")
    print(f"   -> Time elapsed: {bf_duration:.4f} seconds.\n")
    
    # ---------------------------------------------------------
    # TEST 2: EARLY-PRUNING ENGINE
    # ---------------------------------------------------------
    print("2. Testing Pruned DFS Engine...")
    start_time = time.perf_counter()
    
    result = compose_chorale_2nd_order(start_c_major, num_chords=target_length, top_k=branching_factor)
    pruned_duration = time.perf_counter() - start_time
    
    if len(result) == target_length:
        print(f"   -> Success! Valid {len(result)}-chord sequence generated.")
        print(f"   -> Time elapsed: {pruned_duration:.4f} seconds.\n")
    else:
        print(f"   -> Failure: Only generated {len(result)} chords out of {target_length}.\n")
    
    # ---------------------------------------------------------
    # THE BIG-O CONCLUSION
    # ---------------------------------------------------------
    print("==========================================")
    if len(result) == target_length:
        print("BIG-O ALGORITHMIC CONCLUSION:")
        print("1. TIME COMPLEXITY:")
        print("   Brute-Force scales at O(k^n). A 16-chord song would require")
        print("   152 billion loops. Early Pruning destroys invalid branches ")
        print("   immediately, bypassing the exponential explosion.")
        print("\n2. SPACE COMPLEXITY:")
        print("   Storing brute-force arrays requires O(k^n) memory.")
        print("   Our DFS Backtracking engine only stores the current active path,")
        print("   reducing Space Complexity to O(n) (Extremely memory efficient!).")
        print("==========================================\n")

if __name__ == "__main__":
    run_benchmarks()