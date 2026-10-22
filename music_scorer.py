import math

def score_composition(chord_sequence, dynamic_matrix=None):
    """
    Grades a sequence of SATB tuples out of 100 based on Stylistic Probability,
    Vertical Harmonic Quality, and Polyphonic Fluidity (independent rests).
    """
    if not chord_sequence or len(chord_sequence) < 2:
        return 0

    score = 100.0

    for i in range(1, len(chord_sequence)):
        prev_chord = chord_sequence[i-1]
        curr_chord = chord_sequence[i]

        # 1. THE STYLE FACTOR
        if dynamic_matrix and prev_chord in dynamic_matrix:
            prob = dynamic_matrix[prev_chord].get(curr_chord, 0.0)
            if prob > 0.05:
                score += 15  
            elif prob > 0:
                score += 5   
            else:
                score -= 10  

        # 2. POLYPHONY & STAGGERED ENTRY REWARD
        # Count how many voices are resting in the current chord
        curr_rests = sum(1 for n in curr_chord if n == -1)
        prev_rests = sum(1 for n in prev_chord if n == -1)
        
        if 0 < curr_rests < 4:
            # Huge reward for partial rests (e.g., Alto and Tenor play while Soprano and Bass rest)
            # This is what creates fluid, independent voice movement.
            score += 12 
        elif curr_rests == 0 and prev_rests > 0:
            # Reward voices re-entering the texture after a rest
            score += 5

        # Ignore full-chord rests for the voice-leading math
        if curr_rests == 4 or prev_rests == 4:
            continue

        # 3. VERTICAL HARMONIC QUALITY (Dissonance Check on active voices)
        active_pcs = set(n % 12 for n in curr_chord if n != -1)
        pcs_list = list(active_pcs)
        dissonance = 0
        for x in range(len(pcs_list)):
            for y in range(x+1, len(pcs_list)):
                interval = abs(pcs_list[x] - pcs_list[y])
                if interval == 1 or interval == 11: 
                    dissonance += 1
        
        score -= (dissonance * 5) 

        # 4. BASIC SINGABILITY (Only check voices that didn't just rest)
        for v in range(4):
            if curr_chord[v] != -1 and prev_chord[v] != -1:
                leap = abs(curr_chord[v] - prev_chord[v])
                if leap > 12: # Wild jump
                    score -= 3

    return score