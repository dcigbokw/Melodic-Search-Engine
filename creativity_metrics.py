import math
import statistics
import pickle
from chord_generator import all_corpora
from benchmark import instrumented_dfs_engine
from collections import Counter
from search_engine import calculate_levenshtein

def get_order_key(markov_order):
    if markov_order == 1: return "first_order"
    elif markov_order == 2: return "second_order"
    elif markov_order == 3: return "third_order"
    return "second_order"

def calculate_observed_transition_rate(generated_sequence, composer="bach", markov_order=3):
    order_key = get_order_key(markov_order)
    matrix = all_corpora.get(composer, {}).get("train", {}).get(order_key, {})
    total_ngrams = 0
    reused_ngrams = 0
    
    if len(generated_sequence) <= markov_order:
        return 0.0, 0, 0
        
    for i in range(len(generated_sequence) - markov_order):
        total_ngrams += 1
        state = generated_sequence[i] if markov_order == 1 else tuple(generated_sequence[i:i+markov_order])
        next_chord = generated_sequence[i+markov_order]
        
        if state in matrix and next_chord in matrix[state]:
            reused_ngrams += 1
            
    rate = (reused_ngrams / total_ngrams) * 100.0 if total_ngrams > 0 else 0.0
    return rate, reused_ngrams, total_ngrams

def calculate_style_likelihood(generated_sequence, composer="bach", markov_order=2):
    order_key = get_order_key(markov_order)
    test_matrix = all_corpora.get(composer, {}).get("test", {}).get(order_key, {})
    nll_sum = 0.0
    transitions_scored = 0
    unseen_contexts = 0
    per_transition_log_probs = []
    epsilon = 1e-9 
    
    if len(generated_sequence) <= markov_order:
        return float('inf'), 0, 0, []
        
    for i in range(len(generated_sequence) - markov_order):
        state = generated_sequence[i] if markov_order == 1 else tuple(generated_sequence[i:i+markov_order])
        next_chord = generated_sequence[i+markov_order]
        transitions_scored += 1
        
        if state not in test_matrix:
            unseen_contexts += 1
            nll_sum -= math.log(epsilon)
            per_transition_log_probs.append(math.log(epsilon))
        else:
            prob = test_matrix.get(state, {}).get(next_chord, 0.0)
            nll = -math.log(max(prob, epsilon))
            nll_sum += nll
            per_transition_log_probs.append(-nll)
            if prob == 0.0:
                unseen_contexts += 1
                
    avg_nll = nll_sum / transitions_scored if transitions_scored > 0 else float('inf')
    return avg_nll, transitions_scored, unseen_contexts, per_transition_log_probs

def evaluate_blend_fidelity(generated_sequence, comp_a, weight_a, comp_b, weight_b, markov_order=2):
    order_key = get_order_key(markov_order)
    matrix_a = all_corpora.get(comp_a, {}).get("evaluator", {}).get(order_key, {})
    matrix_b = all_corpora.get(comp_b, {}).get("evaluator", {}).get(order_key, {})
    
    total_weight = weight_a + weight_b
    norm_w_a = weight_a / total_weight
    norm_w_b = weight_b / total_weight
    
    nll_a_sum, nll_b_sum, nll_mix_sum = 0.0, 0.0, 0.0
    transitions_scored = 0
    epsilon = 1e-9 
    
    if len(generated_sequence) <= markov_order:
        return float('inf'), float('inf'), float('inf')
        
    for i in range(len(generated_sequence) - markov_order):
        state = generated_sequence[i] if markov_order == 1 else tuple(generated_sequence[i:i+markov_order])
        next_chord = generated_sequence[i+markov_order]
        transitions_scored += 1
        
        prob_a = matrix_a.get(state, {}).get(next_chord, 0.0)
        prob_b = matrix_b.get(state, {}).get(next_chord, 0.0)
        prob_mix = (prob_a * norm_w_a) + (prob_b * norm_w_b)
        
        nll_a_sum -= math.log(max(prob_a, epsilon))
        nll_b_sum -= math.log(max(prob_b, epsilon))
        nll_mix_sum -= math.log(max(prob_mix, epsilon))
        
    if transitions_scored == 0:
        return float('inf'), float('inf'), float('inf')
        
    return (nll_a_sum / transitions_scored, nll_b_sum / transitions_scored, nll_mix_sum / transitions_scored)

def calculate_phrase_memorization(generated_sequence, training_sequences):
    if not generated_sequence or not training_sequences:
        return 0, 0.0

    max_overlap = 0
    gen_len = len(generated_sequence)

    for train_seq in training_sequences:
        train_len = len(train_seq)
        dp = [[0] * (train_len + 1) for _ in range(gen_len + 1)]
        
        for i in range(1, gen_len + 1):
            for j in range(1, train_len + 1):
                if generated_sequence[i - 1] == train_seq[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    if dp[i][j] > max_overlap:
                        max_overlap = dp[i][j]
                else:
                    dp[i][j] = 0

    overlap_percentage = (max_overlap / gen_len) * 100
    return max_overlap, overlap_percentage

def calculate_nearest_neighbor_novelty(generated_sequence, training_sequences):
    if not generated_sequence or not training_sequences:
        return 1.0

    min_distance = float('inf')
    gen_len = len(generated_sequence)

    for train_seq in training_sequences:
        dist = calculate_levenshtein(generated_sequence, train_seq)
        max_possible_dist = max(gen_len, len(train_seq))
        normalized_dist = dist / max_possible_dist if max_possible_dist > 0 else 0
        
        if normalized_dist < min_distance:
            min_distance = normalized_dist

    return min_distance

def calculate_structural_diversity(generated_sequence):
    if len(generated_sequence) < 2:
        return 0.0, 0.0

    unique_chords = len(set(generated_sequence))
    distinct_ratio = unique_chords / len(generated_sequence)

    transitions = []
    for i in range(len(generated_sequence) - 1):
        transitions.append((generated_sequence[i], generated_sequence[i+1]))
        
    transition_counts = Counter(transitions)
    total_transitions = len(transitions)
    
    entropy = 0.0
    for count in transition_counts.values():
        p = count / total_transitions
        entropy -= p * math.log2(p)

    return distinct_ratio, entropy

def run_creativity_evaluation():
    print("==========================================")
    print("SUITE 3.2: QUANTITATIVE CREATIVITY SUITE   ")
    print("==========================================\n")
    
    start_c_major = (72, 67, 60, 48)
    num_chords = 16
    
    print("Scenario 1: Generating Strict Bach (100% Rules, 2nd Order)...")
    comp_weights_bach = {'bach': 100, 'beethoven': 0, 'chopin': 0, 'mozart': 0, 'tchaikovsky': 0, 'handel': 0}
    drafts_strict, _ = instrumented_dfs_engine(start_c_major, num_chords, strictness=1.0, markov_order=2, comp_weights=comp_weights_bach, seed=42)
    
    if drafts_strict:
        seq = drafts_strict[0]
        reuse_rate, reuse_num, reuse_den = calculate_observed_transition_rate(seq, "bach", markov_order=3)
        nll_bach, scored_bach, unseen_bach, _ = calculate_style_likelihood(seq, "bach", markov_order=2)
        nll_chopin, scored_chopin, unseen_chopin, _ = calculate_style_likelihood(seq, "chopin", markov_order=2)
        
        print(f"   -> Sequence Length: {len(seq)} chords")
        print(f"   -> Local Transition Reuse (3rd Order): {reuse_rate:.1f}% ({reuse_num}/{reuse_den})")
        print(f"   -> NLL (Bach model):   {nll_bach:.3f} (Scored {scored_bach}, Unseen {unseen_bach})")
        print(f"   -> NLL (Chopin model): {nll_chopin:.3f}")
    else:
        print("   -> Failed to generate a sequence.")

    print("\nScenario 2: Generating Style Blend (50% Bach, 50% Chopin, 0.8 Strictness)...")
    comp_weights_blend = {'bach': 50, 'beethoven': 0, 'chopin': 50, 'mozart': 0, 'tchaikovsky': 0, 'handel': 0}
    drafts_blend, _ = instrumented_dfs_engine(start_c_major, num_chords, strictness=0.8, markov_order=2, comp_weights=comp_weights_blend, seed=42)
    
    if drafts_blend:
        seq_blend = drafts_blend[0]
        reuse_bach, _, _ = calculate_observed_transition_rate(seq_blend, "bach", markov_order=3)
        reuse_chopin, _, _ = calculate_observed_transition_rate(seq_blend, "chopin", markov_order=3)
        nll_a, nll_b, nll_mix = evaluate_blend_fidelity(seq_blend, "bach", 50, "chopin", 50, markov_order=2)
        
        print(f"   -> Local Transition Reuse (Bach):   {reuse_bach:.1f}%")
        print(f"   -> Local Transition Reuse (Chopin): {reuse_chopin:.1f}%")
        print(f"   -> NLL (Bach model):   {nll_a:.3f}")
        print(f"   -> NLL (Chopin model): {nll_b:.3f}")
        print(f"   -> NLL (Target Mix):   {nll_mix:.3f}")
    else:
        print("   -> Failed to generate a sequence.")

def run_advanced_creativity_suite():
    print("\n======================================================")
    print(" SUITE 3.2: ADVANCED CREATIVITY METRICS (CORPUS-BACKED) ")
    print("======================================================\n")
    
    try:
        with open("training_sequences.pkl", "rb") as f:
            all_training_sequences = pickle.load(f)
            real_train_corpus = all_training_sequences.get("bach", [])
    except FileNotFoundError:
        print("training_sequences.pkl not found. Run train.py first.")
        return

    start_c_major = (72, 67, 60, 48)
    comp_weights = {'bach': 100, 'beethoven': 0, 'chopin': 0, 'mozart': 0, 'tchaikovsky': 0, 'handel': 0}
    
    generated_collection = []
    for i in range(5):
        drafts, _ = instrumented_dfs_engine(start_c_major, 16, strictness=1.0, markov_order=2, comp_weights=comp_weights, seed=800+i)
        if drafts:
            generated_collection.append(drafts[0])
            
    print(f"Evaluated across {len(generated_collection)} generated compositions.")
    
    avg_overlap, avg_novelty = 0.0, 0.0
    global_chords = set()
    
    for seq in generated_collection:
        _, overlap_pct = calculate_phrase_memorization(seq, real_train_corpus)
        avg_overlap += overlap_pct
        avg_novelty += calculate_nearest_neighbor_novelty(seq, real_train_corpus)
        global_chords.update(seq)
        
    if generated_collection:
        avg_overlap /= len(generated_collection)
        avg_novelty /= len(generated_collection)
        
    print(f"  Avg Phrase Memorization: {avg_overlap:.1f}%")
    print(f"  Avg NN Novelty:          {avg_novelty:.3f}")
    print(f"  Global Distinct Chords:  {len(global_chords)} unique states across collection")

if __name__ == "__main__":
    run_creativity_evaluation()
    run_advanced_creativity_suite()