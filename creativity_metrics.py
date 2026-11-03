import math
import statistics
from chord_generator import all_corpora
from benchmark import instrumented_dfs_engine

def get_order_key(markov_order):
    if markov_order == 1: return "first_order"
    elif markov_order == 2: return "second_order"
    elif markov_order == 3: return "third_order"
    return "second_order"

def calculate_memorization_rate(generated_sequence, composer="bach", markov_order=3):
    """
    Calculates what percentage of N-grams in the generated sequence 
    are direct, exact copies from the original training corpus.
    """
    order_key = get_order_key(markov_order)
    matrix = all_corpora.get(composer, {}).get(order_key, {})
    
    total_ngrams = 0
    memorized_ngrams = 0
    
    for i in range(len(generated_sequence) - markov_order):
        total_ngrams += 1
        
        if markov_order == 1:
            state = generated_sequence[i]
        else:
            state = tuple(generated_sequence[i:i+markov_order])
            
        next_chord = generated_sequence[i+markov_order]
        
        # If this exact sequence -> next_chord path exists in the corpus
        if state in matrix and next_chord in matrix[state]:
            memorized_ngrams += 1
            
    if total_ngrams == 0: return 0.0
    return (memorized_ngrams / total_ngrams) * 100.0

def calculate_style_likelihood(generated_sequence, composer="bach", markov_order=2):
    """
    Calculates the average negative log-likelihood (NLL) of the sequence 
    under a specific composer's probability matrix. 
    Lower NLL = closer mathematical adherence to that style.
    """
    order_key = get_order_key(markov_order)
    matrix = all_corpora.get(composer, {}).get(order_key, {})
        
    nll_sum = 0.0
    transitions_scored = 0
    epsilon = 1e-9 # Prevent log(0) domain errors
    
    for i in range(len(generated_sequence) - markov_order):
        if markov_order == 1:
            state = generated_sequence[i]
        else:
            state = tuple(generated_sequence[i:i+markov_order])
            
        next_chord = generated_sequence[i+markov_order]
        
        prob = matrix.get(state, {}).get(next_chord, 0.0)
        nll_sum -= math.log(max(prob, epsilon))
        transitions_scored += 1
        
    if transitions_scored == 0: return float('inf')
    return nll_sum / transitions_scored

def run_creativity_evaluation():
    print("==========================================")
    print("    QUANTITATIVE CREATIVITY EVALUATION    ")
    print("==========================================\n")
    
    start_c_major = (72, 67, 60, 48)
    num_chords = 16
    
    # ---------------------------------------------------------
    # TEST 1: STRICT BEETHOVEN
    # ---------------------------------------------------------
    print("Scenario 1: Generating Strict Beethoven (100% Rules, 2nd Order)...")
    comp_weights_beethoven = {'bach': 0, 'beethoven': 100, 'chopin': 0, 'mozart': 0, 'tchaikovsky': 0, 'handel': 0}
    drafts_strict, _ = instrumented_dfs_engine(start_c_major, num_chords, strictness=1.0, markov_order=2, comp_weights=comp_weights_beethoven)
    
    if drafts_strict:
        seq = [start_c_major] + drafts_strict[0]
        mem_rate = calculate_memorization_rate(seq, "beethoven", markov_order=3)
        nll_beeth = calculate_style_likelihood(seq, "beethoven", markov_order=2)
        nll_mozart = calculate_style_likelihood(seq, "mozart", markov_order=2)
        
        print(f"   -> Trigram Memorization Rate: {mem_rate:.1f}%")
        print(f"   -> NLL (Beethoven model): {nll_beeth:.3f}  (Lower is better)")
        print(f"   -> NLL (Mozart model):    {nll_mozart:.3f}")
    else:
        print("   -> Failed to generate a sequence.")

    # ---------------------------------------------------------
    # TEST 2: INTERPOLATED STYLE BLEND (Beethoven + Mozart)
    # ---------------------------------------------------------
    print("\nScenario 2: Generating Style Blend (50% Beethoven, 50% Mozart, 0.8 Strictness)...")
    comp_weights_blend = {'bach': 0, 'beethoven': 50, 'chopin': 0, 'mozart': 50, 'tchaikovsky': 0, 'handel': 0}
    drafts_blend, _ = instrumented_dfs_engine(start_c_major, num_chords, strictness=0.8, markov_order=2, comp_weights=comp_weights_blend)
    
    if drafts_blend:
        seq_blend = [start_c_major] + drafts_blend[0]
        mem_rate_beeth = calculate_memorization_rate(seq_blend, "beethoven", markov_order=3)
        mem_rate_mozart = calculate_memorization_rate(seq_blend, "mozart", markov_order=3)
        nll_beeth_blend = calculate_style_likelihood(seq_blend, "beethoven", markov_order=2)
        nll_mozart_blend = calculate_style_likelihood(seq_blend, "mozart", markov_order=2)
        
        print(f"   -> Trigram Memorization (Beethoven): {mem_rate_beeth:.1f}%")
        print(f"   -> Trigram Memorization (Mozart):    {mem_rate_mozart:.1f}%")
        print(f"   -> NLL (Beethoven model): {nll_beeth_blend:.3f}")
        print(f"   -> NLL (Mozart model):    {nll_mozart_blend:.3f}")
    else:
        print("   -> Failed to generate a sequence.")
        
if __name__ == "__main__":
    run_creativity_evaluation()