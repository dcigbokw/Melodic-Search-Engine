import time
import itertools
import statistics
import random
import math
from chord_generator import all_corpora, is_valid_transition
from main import interpolate_matrices

def brute_force_simulation(start_chord, num_chords, top_k, matrix):
    """
    An honest brute-force: Actually builds the arrays and checks the rules.
    """
    if start_chord not in matrix:
        return 0

    # Sort the 1st-order predictions by probability (highest first)
    raw_predictions = matrix[start_chord]
    sorted_pool = sorted(raw_predictions.items(), key=lambda x: x[1], reverse=True)

    # Grab just the chord tuples for the top_k candidates
    pool = [chord for chord, prob in sorted_pool[:top_k] if isinstance(chord, tuple) and chord[0] != -1]
        
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

def instrumented_dfs_engine(start_chord, num_chords, strictness, markov_order, comp_weights, tonic_pc=0):
    """
    Runs the modern DFS engine with telemetry to track pruning and nodes explored.
    """
    matrix_1st = interpolate_matrices(all_corpora, comp_weights, "first_order")
    matrix_2nd = interpolate_matrices(all_corpora, comp_weights, "second_order")
    matrix_3rd = interpolate_matrices(all_corpora, comp_weights, "third_order")
    
    fallback_options = [c for c in matrix_1st.keys() if isinstance(c, tuple) and c[0] != -1]
    successful_drafts = []
    
    stats = {"explored": 0, "pruned": 0}
    
    def dfs_search(path, visited):
        if len(successful_drafts) >= 1: return
        
        stats["explored"] += 1
        current_depth = len(path) - 1
        
        if current_depth == num_chords - 1:
            successful_drafts.append(path.copy())
            return
            
        opts_dict = {}
        target_order = markov_order
        
        while target_order >= 1 and not opts_dict:
            if target_order == 3 and len(path) >= 3:
                opts_dict = matrix_3rd.get((path[-3], path[-2], path[-1]), {})
            elif target_order == 2 and len(path) >= 2:
                opts_dict = matrix_2nd.get((path[-2], path[-1]), {})
            elif target_order == 1:
                opts_dict = matrix_1st.get(path[-1], {})
            target_order -= 1
            
        valid_keys = [k for k in opts_dict.keys() if isinstance(k, tuple) and k[0] != -1]
        opts = sorted(valid_keys, key=lambda k: math.log(random.random() + 1e-9) / (opts_dict[k] ** 1.0), reverse=True)
        
        if len(opts) < 5 and fallback_options:
            opts.extend(random.sample(fallback_options, min(20, len(fallback_options))))
        
        for cand in opts:
            cand_state = cand[-1] if isinstance(cand, tuple) and isinstance(cand[0], tuple) else cand
            curr_state = path[-1]
            
            passes_rules = is_valid_transition(curr_state, cand_state, tonic_pc)
            if passes_rules or random.random() > strictness:
                h = hash(tuple(path + [cand_state]))
                if h in visited: continue
                visited.add(h)
                path.append(cand_state)
                dfs_search(path, visited)
                path.pop()
            else:
                stats["pruned"] += 1

    dfs_search([start_chord], set())
    return successful_drafts, stats

def run_benchmarks():
    print("==========================================")
    print("      HONEST ALGORITHMIC BENCHMARK        ")
    print("==========================================\n")
    
    start_c_major = (72, 67, 60, 48)
    comp_weights = {'bach': 100, 'beethoven': 0, 'chopin': 0, 'mozart': 0, 'tchaikovsky': 0, 'handel': 0}
    matrix_1st = interpolate_matrices(all_corpora, comp_weights, "first_order")
    
    # ---------------------------------------------------------
    # TEST 1: THE BRUTE-FORCE APPROACH (Small N)
    # ---------------------------------------------------------
    target_length_bf = 5 
    branching_factor = 4
    print(f"1. Testing Honest Brute-Force (N={target_length_bf}, Top-K={branching_factor})...")
    start_time = time.perf_counter()
    nodes_explored_bf = brute_force_simulation(start_c_major, target_length_bf, branching_factor, matrix_1st)
    bf_duration = time.perf_counter() - start_time
    print(f"   -> Explored {nodes_explored_bf:,} combinations.")
    print(f"   -> Time elapsed: {bf_duration:.4f} seconds.\n")
    
    # ---------------------------------------------------------
    # TEST 2: QUANTITATIVE EVALUATION SUITE (Large N)
    # ---------------------------------------------------------
    print("2. Running Quantitative DFS Evaluation Suite...")
    num_trials = 20
    target_length_dfs = 16
    
    test_configs = [
        {"strictness": 1.0, "order": 2, "label": "Strict Classical (100% Rules, 2nd Order)"},
        {"strictness": 0.5, "order": 2, "label": "Loose/Jazz (50% Rules, 2nd Order)"},
        {"strictness": 1.0, "order": 1, "label": "Strict Classical (100% Rules, 1st Order)"}
    ]
    
    for config in test_configs:
        print(f"\n--- Scenario: {config['label']} ---")
        
        runtimes = []
        nodes_explored = []
        pruned_branches = []
        successes = 0
        
        for _ in range(num_trials):
            start_time = time.perf_counter()
            drafts, stats = instrumented_dfs_engine(
                start_c_major, target_length_dfs, 
                config["strictness"], config["order"], comp_weights
            )
            duration = time.perf_counter() - start_time
            
            if drafts:
                successes += 1
                runtimes.append(duration * 1000) 
                nodes_explored.append(stats["explored"])
                pruned_branches.append(stats["pruned"])
                
        success_rate = (successes / num_trials) * 100
        avg_runtime = statistics.mean(runtimes) if runtimes else 0
        avg_nodes = statistics.mean(nodes_explored) if nodes_explored else 0
        avg_pruned = statistics.mean(pruned_branches) if pruned_branches else 0
        pruning_ratio = (avg_pruned / (avg_pruned + avg_nodes)) * 100 if (avg_pruned + avg_nodes) > 0 else 0
        
        print(f"   Trials:         {num_trials}")
        print(f"   Target Chords:  {target_length_dfs}")
        print(f"   Success Rate:   {success_rate:.1f}%")
        print(f"   Avg Runtime:    {avg_runtime:.2f} ms")
        print(f"   Avg Nodes/Run:  {avg_nodes:.1f}")
        print(f"   Pruning Ratio:  {pruning_ratio:.1f}%")

if __name__ == "__main__":
    run_benchmarks()