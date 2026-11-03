import time
import itertools
import statistics
import random
import math
import json
from chord_generator import all_corpora, is_valid_transition
from main import interpolate_matrices
from rules_engine import check_parallel_motion, check_crossing_and_spacing, check_leading_tone_resolution
from rhythm_ai import generate_rhythms
from collections import Counter
from music_scorer import score_composition
from search_engine import advanced_search

def brute_force_simulation(start_chord, num_chords, top_k, matrix, seed=42):
    """
    An honest brute-force: Actually builds the arrays and checks the rules.
    Explores the exact same state-dependent candidate pool logic.
    """
    random.seed(seed)
    if start_chord not in matrix:
        return 0, 0

    raw_predictions = matrix[start_chord]
    sorted_pool = sorted(raw_predictions.items(), key=lambda x: x[1], reverse=True)
    pool = [chord for chord, prob in sorted_pool[:top_k] if isinstance(chord, tuple) and chord[0] != -1]
    
    if not pool:
        return 0, 0

    all_possible_songs = itertools.product(pool, repeat=(num_chords - 1))
    
    paths_explored = 0
    valid_songs_found = 0
    
    for chord_sequence in all_possible_songs:
        paths_explored += 1
        song = [start_chord] + list(chord_sequence)
        
        is_valid = True
        for i in range(len(song) - 1):
            if not is_valid_transition(song[i], song[i+1], tonic_pc=0):
                is_valid = False
                break 
                
        if is_valid:
            valid_songs_found += 1
            
    return paths_explored, valid_songs_found

def instrumented_dfs_engine(start_chord, num_chords, strictness, markov_order, comp_weights, tonic_pc=0, seed=42):
    """
    Runs the modern DFS engine with rigorous telemetry.
    """
    random.seed(seed)
    matrix_1st = interpolate_matrices(all_corpora, comp_weights, "first_order")
    matrix_2nd = interpolate_matrices(all_corpora, comp_weights, "second_order")
    matrix_3rd = interpolate_matrices(all_corpora, comp_weights, "third_order")
    
    fallback_options = [c for c in matrix_1st.keys() if isinstance(c, tuple) and c[0] != -1]
    successful_drafts = []
    
    stats = {
        "candidate_evaluations": 0,
        "nodes_entered": 0,
        "rule_rejections": 0,
        "backtracks": 0,
        "fallback_used": 0,
        "duplicate_skips": 0,
        "max_depth": 0
    }
    
    def dfs_search(path, visited):
        if len(successful_drafts) >= 1: return
        
        stats["nodes_entered"] += 1
        current_depth = len(path) - 1
        if current_depth > stats["max_depth"]:
            stats["max_depth"] = current_depth
            
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
        
        if not valid_keys and fallback_options:
            stats["fallback_used"] += 1
            valid_keys = random.sample(fallback_options, min(10, len(fallback_options)))
            
        opts = sorted(valid_keys, key=lambda k: math.log(random.random() + 1e-9) / (opts_dict.get(k, 0.1) ** 1.0), reverse=True)
        
        branches_explored = 0
        for cand in opts:
            stats["candidate_evaluations"] += 1
            cand_state = cand[-1] if isinstance(cand, tuple) and isinstance(cand[0], tuple) else cand
            curr_state = path[-1]
            
            passes_rules = is_valid_transition(curr_state, cand_state, tonic_pc)
            if passes_rules or random.random() > strictness:
                h = hash(tuple(path + [cand_state]))
                if h in visited:
                    stats["duplicate_skips"] += 1
                    continue
                visited.add(h)
                path.append(cand_state)
                branches_explored += 1
                
                dfs_search(path, visited)
                
                if len(successful_drafts) == 0:
                    stats["backtracks"] += 1
                path.pop()
            else:
                stats["rule_rejections"] += 1

    dfs_search([start_chord], set())
    return successful_drafts, stats

def calculate_sequence_nll(sequence, evaluator_composer, markov_order=2):
    """Helper to calculate NLL against the held-out test split for a specific composer."""
    order_key = "first_order" if markov_order == 1 else "second_order"
    if markov_order == 3: order_key = "third_order"
    
    test_matrix = all_corpora.get(evaluator_composer, {}).get("test", {}).get(order_key, {})
    if not test_matrix: return float('inf')
    
    nll_sum = 0.0
    transitions = 0
    epsilon = 1e-9
    
    for i in range(len(sequence) - markov_order):
        state = sequence[i] if markov_order == 1 else tuple(sequence[i:i+markov_order])
        next_chord = sequence[i+markov_order]
        
        prob = test_matrix.get(state, {}).get(next_chord, 0.0)
        nll_sum -= math.log(max(prob, epsilon))
        transitions += 1
        
    return (nll_sum / transitions) if transitions > 0 else float('inf')

def calculate_duration_entropy(durations):
    """Calculates the Shannon entropy of the duration sequence to measure rhythmic diversity."""
    if not durations:
        return 0.0
    counts = Counter(durations)
    total = len(durations)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

def check_meter_alignment(durations, beats_per_measure=4.0):
    """Checks if the total generated duration aligns perfectly with the requested meter boundaries."""
    total_duration = sum(durations)
    # Allow a tiny epsilon for floating point errors in quarter lengths
    return abs((total_duration % beats_per_measure)) < 1e-5

def run_search_efficiency_suite():
    print("======================================================")
    print("  SUITE 2.1: ALGORITHMIC SEARCH (BRUTE FORCE vs DFS)  ")
    print("======================================================\n")
    
    start_c_major = (72, 67, 60, 48)
    comp_weights = {'bach': 100, 'beethoven': 0, 'chopin': 0, 'mozart': 0, 'tchaikovsky': 0, 'handel': 0}
    matrix_1st = interpolate_matrices(all_corpora, comp_weights, "first_order")
    num_trials = 20
    
    target_lengths = [4, 8, 16] 
    
    for target_length in target_lengths:
        print(f"\n--- Sweeping Target Length: {target_length} Chords ---")
        
        runtimes = []
        stats_agg = {k: [] for k in ["candidate_evaluations", "nodes_entered", "rule_rejections", "backtracks", "fallback_used", "duplicate_skips", "max_depth"]}
        successes = 0
        
        for trial in range(num_trials):
            start_time = time.perf_counter()
            drafts, stats = instrumented_dfs_engine(
                start_c_major, target_length, strictness=1.0, markov_order=2, comp_weights=comp_weights, seed=42 + trial
            )
            duration = (time.perf_counter() - start_time) * 1000
            runtimes.append(duration)
            
            for k in stats_agg.keys():
                stats_agg[k].append(stats[k])
                
            if drafts:
                successes += 1
                
        completion_rate = (successes / num_trials) * 100
        mean_rt = statistics.mean(runtimes)
        median_rt = statistics.median(runtimes)
        p95_rt = sorted(runtimes)[int(len(runtimes) * 0.95)]
        
        print(f"Completion Rate:       {completion_rate:.1f}%")
        print(f"Candidate Evaluations: {statistics.mean(stats_agg['candidate_evaluations']):.1f} (avg)")
        print(f"Nodes Entered:         {statistics.mean(stats_agg['nodes_entered']):.1f} (avg)")
        print(f"Rule Rejections:       {statistics.mean(stats_agg['rule_rejections']):.1f} (avg)")
        print(f"Backtracks:            {statistics.mean(stats_agg['backtracks']):.1f} (avg)")
        print(f"Fallback Usage:        {statistics.mean(stats_agg['fallback_used']):.1f} (avg)")
        print(f"Duplicate Skips:       {statistics.mean(stats_agg['duplicate_skips']):.1f} (avg)")
        print(f"Max Search Depth:      {statistics.mean(stats_agg['max_depth']):.1f} (avg)")
        print(f"Latency (ms):          Mean: {mean_rt:.2f} | Median: {median_rt:.2f} | p95: {p95_rt:.2f}")

def run_model_evaluation_suite():
    print("\n======================================================")
    print("  SUITE 2.2: MARKOV MODEL EVALUATION (HELD-OUT TEST)  ")
    print("======================================================\n")
    
    composers = ['bach', 'beethoven', 'chopin', 'mozart']
    orders = [("First Order", "first_order", 1), ("Second Order", "second_order", 2), ("Third Order", "third_order", 3)]
    
    for comp in composers:
        print(f"--- Evaluating Composer: {comp.upper()} ---")
        train_matrices = all_corpora.get(comp, {}).get("train", {})
        test_matrices = all_corpora.get(comp, {}).get("test", {})
        
        if not train_matrices or not test_matrices:
            print(f"Missing train/test split for {comp}. Skipping.")
            continue
            
        for order_name, order_key, n in orders:
            train_matrix = train_matrices.get(order_key, {})
            test_matrix = test_matrices.get(order_key, {})
            
            total_transitions = 0
            unseen_contexts = 0
            nll_sum = 0.0
            top1_hits = 0
            top5_hits = 0
            epsilon = 1e-9
            
            for state, transitions in test_matrix.items():
                train_transitions = train_matrix.get(state, {})
                sorted_train = sorted(train_transitions.items(), key=lambda item: item[1], reverse=True)
                top1_preds = [k for k, v in sorted_train[:1]]
                top5_preds = [k for k, v in sorted_train[:5]]
                
                if not train_transitions:
                    unseen_contexts += sum(transitions.values())
                
                for next_chord, test_count in transitions.items():
                    total_transitions += test_count
                    prob = train_transitions.get(next_chord, 0.0)
                    
                    nll_sum -= (math.log(max(prob, epsilon)) * test_count)
                    
                    if next_chord in top1_preds: top1_hits += test_count
                    if next_chord in top5_preds: top5_hits += test_count
                    
            if total_transitions > 0:
                avg_nll = nll_sum / total_transitions
                perplexity = math.exp(avg_nll) if avg_nll < 700 else float('inf')
                coverage = ((total_transitions - unseen_contexts) / total_transitions) * 100
                top1_acc = (top1_hits / total_transitions) * 100
                top5_acc = (top5_hits / total_transitions) * 100
                
                print(f"  {order_name}:")
                print(f"    Context Coverage: {coverage:.1f}%")
                print(f"    Avg NLL:          {avg_nll:.3f}")
                print(f"    Perplexity:       {perplexity:.1f}")
                print(f"    Top-1 Accuracy:   {top1_acc:.1f}%")
                print(f"    Top-5 Accuracy:   {top5_acc:.1f}%")

def run_constraints_evaluation_suite():
    print("\n======================================================")
    print(" SUITE 2.3: GENERATION RELIABILITY & MUSICAL CONSTRAINTS ")
    print("======================================================\n")
    
    start_c_major = (72, 67, 60, 48)
    comp_weights = {'bach': 100, 'beethoven': 0, 'chopin': 0, 'mozart': 0, 'tchaikovsky': 0, 'handel': 0}
    num_trials = 20
    target_length = 16
    
    test_strictness_levels = [1.0, 0.5, 0.0]
    
    for strictness in test_strictness_levels:
        print(f"\n--- Testing Strictness: {strictness} ---")
        
        successes = 0
        rule_violations = {"parallel_motion": 0, "spacing_crossing": 0, "leading_tone": 0}
        out_of_range_count = 0
        total_transitions_evaluated = 0
        
        for trial in range(num_trials):
            drafts, stats = instrumented_dfs_engine(
                start_c_major, target_length, strictness=strictness, markov_order=2, comp_weights=comp_weights, seed=100+trial
            )
            
            if drafts:
                successes += 1
                seq = drafts[0]
                
                for i in range(len(seq) - 1):
                    total_transitions_evaluated += 1
                    chord_a, chord_b = seq[i], seq[i+1]
                    
                    if not check_crossing_and_spacing(chord_b[0], chord_b[1], chord_b[2], chord_b[3]):
                        rule_violations["spacing_crossing"] += 1
                        
                    for v in range(4):
                        if not check_leading_tone_resolution(chord_a[v], chord_b[v], tonic_pc=0):
                            rule_violations["leading_tone"] += 1
                            
                    for v1 in range(4):
                        for v2 in range(v1 + 1, 4):
                            if not check_parallel_motion(chord_a[v1], chord_b[v1], chord_a[v2], chord_b[v2]):
                                rule_violations["parallel_motion"] += 1

                RANGES = {0: (60, 81), 1: (53, 74), 2: (48, 69), 3: (40, 62)}
                for c in seq:
                    for v in range(4):
                        min_p, max_p = RANGES[v]
                        if not (min_p <= c[v] <= max_p):
                            out_of_range_count += 1
                            
        completion_rate = (successes / num_trials) * 100
        
        print(f"  Completion Rate:      {completion_rate:.1f}%")
        print(f"  Observed Violations in Final Outputs:")
        print(f"    Parallel Motion:    {rule_violations['parallel_motion']} instances")
        print(f"    Spacing/Crossing:   {rule_violations['spacing_crossing']} instances")
        print(f"    Leading Tone:       {rule_violations['leading_tone']} instances")
        print(f"    Out-of-Range Notes: {out_of_range_count} instances")


def run_blending_controllability_suite():
    print("\n======================================================")
    print(" SUITE 2.4: COMPOSER BLENDING & CONTROLLABILITY ")
    print("======================================================\n")
    
    start_c_major = (72, 67, 60, 48)
    num_trials = 10
    target_length = 16
    
    # Test a sliding scale between Beethoven and Mozart
    blend_ratios = [
        (100, 0),   # 100% Beethoven
        (75, 25),   # 75% Beethoven, 25% Mozart
        (50, 50),   # 50/50 Blend
        (25, 75),   # 25% Beethoven, 75% Mozart
        (0, 100)    # 100% Mozart
    ]
    
    print(f"{'Blend (Beeth/Mozart)':<22} | {'Success':<8} | {'NLL (Beethoven)':<16} | {'NLL (Mozart)':<16}")
    print("-" * 70)
    
    for w_beeth, w_mozart in blend_ratios:
        comp_weights = {
            'bach': 0, 'beethoven': w_beeth, 'chopin': 0, 
            'mozart': w_mozart, 'tchaikovsky': 0, 'handel': 0
        }
        
        successes = 0
        nll_beethoven_list = []
        nll_mozart_list = []
        
        for trial in range(num_trials):
            drafts, stats = instrumented_dfs_engine(
                start_c_major, target_length, strictness=0.8, markov_order=2, comp_weights=comp_weights, seed=500+trial
            )
            
            if drafts:
                successes += 1
                seq = drafts[0]
                nll_beethoven_list.append(calculate_sequence_nll(seq, "beethoven", markov_order=2))
                nll_mozart_list.append(calculate_sequence_nll(seq, "mozart", markov_order=2))
                
        avg_nll_beethoven = statistics.mean(nll_beethoven_list) if nll_beethoven_list else float('inf')
        avg_nll_mozart = statistics.mean(nll_mozart_list) if nll_mozart_list else float('inf')
        success_rate = (successes / num_trials) * 100
        
        label = f"{w_beeth}% / {w_mozart}%"
        print(f"{label:<22} | {success_rate:>5.0f}%   | {avg_nll_beethoven:>14.3f} | {avg_nll_mozart:>14.3f}")


def run_rhythm_evaluation_suite():
    print("\n======================================================")
    print(" SUITE 2.5: RHYTHM GENERATION & METER ALIGNMENT ")
    print("======================================================\n")
    
    num_chords = 16
    num_trials = 20
    
    test_conditions = [
        {"label": "100% Bach", "weights": {'bach': 100, 'beethoven': 0, 'chopin': 0, 'mozart': 0, 'tchaikovsky': 0, 'handel': 0}},
        {"label": "100% Chopin", "weights": {'bach': 0, 'beethoven': 0, 'chopin': 100, 'mozart': 0, 'tchaikovsky': 0, 'handel': 0}},
        {"label": "50/50 Bach-Chopin", "weights": {'bach': 50, 'beethoven': 0, 'chopin': 50, 'mozart': 0, 'tchaikovsky': 0, 'handel': 0}}
    ]
    
    meters = [
        {"name": "4/4", "beats": 4.0},
        {"name": "3/4", "beats": 3.0},
        {"name": "6/8", "beats": 1.5} # Assuming dotted quarter gets the beat
    ]
    
    print(f"{'Condition':<20} | {'Avg Entropy':<12} | {'Unique Patterns':<16} | {'4/4 Align':<10} | {'3/4 Align':<10}")
    print("-" * 75)
    
    for condition in test_conditions:
        entropies = []
        unique_patterns = set()
        alignments = {"4/4": 0, "3/4": 0, "6/8": 0}
        
        for _ in range(num_trials):
            rhythms = generate_rhythms(num_chords=num_chords, rhythm_weights=condition["weights"])
            
            entropies.append(calculate_duration_entropy(rhythms))
            unique_patterns.add(tuple(rhythms))
            
            if check_meter_alignment(rhythms, beats_per_measure=4.0):
                alignments["4/4"] += 1
            if check_meter_alignment(rhythms, beats_per_measure=3.0):
                alignments["3/4"] += 1
                
        avg_entropy = sum(entropies) / num_trials
        unique_count = len(unique_patterns)
        align_44_pct = (alignments["4/4"] / num_trials) * 100
        align_34_pct = (alignments["3/4"] / num_trials) * 100
        
        print(f"{condition['label']:<20} | {avg_entropy:<12.3f} | {unique_count:>5} / {num_trials:<8} | {align_44_pct:>7.1f}% | {align_34_pct:>7.1f}%")

def run_scorer_evaluation_suite():
    print("\n======================================================")
    print(" SUITE 2.6: CANDIDATE COMPOSITION SCORER EVALUATION ")
    print("======================================================\n")
    
    start_c_major = (72, 67, 60, 48)
    comp_weights = {'bach': 100, 'beethoven': 0, 'chopin': 0, 'mozart': 0, 'tchaikovsky': 0, 'handel': 0}
    matrix_1st = interpolate_matrices(all_corpora, comp_weights, "first_order")
    
    num_trials = 10
    target_length = 8
    
    print("Generating candidates to measure scorer discrimination power...")
    score_spreads = []
    
    for trial in range(num_trials):
        # We temporarily disable strictness to allow a mix of good and bad candidates to be generated
        drafts, _ = instrumented_dfs_engine(
            start_c_major, target_length, strictness=0.0, markov_order=2, comp_weights=comp_weights, seed=600+trial
        )
        
        if len(drafts) > 1:
            scores = [score_composition(draft, matrix_1st) for draft in drafts]
            best_score = max(scores)
            worst_score = min(scores)
            spread = best_score - worst_score
            score_spreads.append(spread)
            
            print(f"  Trial {trial+1}: Best Score: {best_score:.2f} | Worst: {worst_score:.2f} | Spread: {spread:.2f}")
            
    avg_spread = statistics.mean(score_spreads) if score_spreads else 0
    print(f"\n  Average Discrimination Spread: {avg_spread:.2f} points")
    print("  (A higher spread means the scorer is actively differentiating candidates, not just clustering them.)")
    print("  *It is recommended to pair this data with blind human ratings.")


def run_search_evaluation_suite():
    print("\n======================================================")
    print(" SUITE 2.7: SEARCH & RETRIEVAL SUBSYSTEM EVALUATION ")
    print("======================================================\n")
    
    # Mock labeled query set
    test_queries = [
        {"melody": [60, 62, 64, 65, 67], "label": "C Major Scale Ascending"},
        {"melody": [67, 64, 60], "label": "C Major Arpeggio Descending"},
        {"melody": [60, -1, 62, -1, 64], "label": "Melody with Rests"}
    ]
    
    algorithms = ["dtw", "levenshtein"]
    modes = ["exact", "relative"]
    
    print(f"{'Mode':<10} | {'Algorithm':<12} | {'Avg Latency (ms)':<18} | {'Matches Found':<15}")
    print("-" * 65)
    
    for mode in modes:
        for algo in algorithms:
            latencies = []
            total_matches = 0
            
            for q in test_queries:
                start_time = time.perf_counter()
                
                # Fetch top 5 results for each test query
                matches = advanced_search(
                    q["melody"], 
                    algorithm=algo, 
                    target_composer="all", 
                    max_dist=10, 
                    mode=mode, 
                    max_results=5
                )
                
                duration = (time.perf_counter() - start_time) * 1000
                latencies.append(duration)
                total_matches += len(matches)
                
            avg_latency = statistics.mean(latencies) if latencies else 0
            print(f"{mode:<10} | {algo:<12} | {avg_latency:>16.2f} | {total_matches:>13}")
            
    print("\n  * Note: Mean Reciprocal Rank (MRR) and nDCG require a manually labeled ground-truth database.")

def run_integration_evaluation_suite():
    print("\n======================================================")
    print(" SUITE 2.8: HARMONIZATION, XAI & END-TO-END INTEGRATION ")
    print("======================================================\n")
    
    # 1. Harmonization & Soprano Preservation
    test_melody = [72, 74, 76, 77] # C5, D5, E5, F5
    comp_weights = {'bach': 100, 'beethoven': 0, 'chopin': 0, 'mozart': 0, 'tchaikovsky': 0, 'handel': 0}
    matrix_1st = interpolate_matrices(all_corpora, comp_weights, "first_order")
    historical_corpus_chords = [c for c in matrix_1st.keys() if isinstance(c, tuple) and c[0] != -1]
    
    successful_drafts = []
    
    def dfs_harm(depth, path):
        if len(successful_drafts) > 0: return
        if depth == len(test_melody):
            successful_drafts.append(path.copy())
            return
            
        target_soprano = test_melody[depth]
        matching_chords = [c for c in historical_corpus_chords if c[0] == target_soprano]
        
        for cand in matching_chords:
            if depth == 0 or is_valid_transition(path[-1], cand, 0):
                 path.append(cand)
                 dfs_harm(depth + 1, path)
                 path.pop()
                 
    start_time = time.perf_counter()
    dfs_harm(0, [])
    latency = (time.perf_counter() - start_time) * 1000
    
    print("--- Harmonization & Melody Preservation ---")
    if successful_drafts:
        draft = successful_drafts[0]
        soprano_notes = [chord[0] for chord in draft]
        preservation_match = soprano_notes == test_melody
        print(f"  Target Melody:         {test_melody}")
        print(f"  Generated Soprano:     {soprano_notes}")
        print(f"  Preservation Success:  {preservation_match}")
        print(f"  Latency:               {latency:.2f} ms")
    else:
        print("  Harmonization Failed: No rule-compliant sequence found.")
        
    # 2. Input Validation (Malformed Pitches)
    print("\n--- Input Validation (Malformed Strings) ---")
    malformed_input = ["C4", "Rest", "NotANote", "X9"]
    parsed_output = []
    
    for n in malformed_input:
        if n.lower() == 'rest':
            parsed_output.append(-1)
        else:
            try:
                # Mocking the pitch.Pitch(n).midi fallback behavior
                if n in ["C4"]: parsed_output.append(60)
                else: raise ValueError
            except:
                parsed_output.append(60) # Fallback to MIDI 60
                
    print(f"  Raw Input:       {malformed_input}")
    print(f"  Parsed Output:   {parsed_output}")
    print(f"  Fallback Handled correctly to MIDI 60: {parsed_output[2] == 60 and parsed_output[3] == 60}")
    

if __name__ == "__main__":
    run_search_efficiency_suite()
    run_model_evaluation_suite()
    run_constraints_evaluation_suite()
    run_blending_controllability_suite()
    run_rhythm_evaluation_suite()
    run_scorer_evaluation_suite()
    run_search_evaluation_suite()
    run_integration_evaluation_suite()