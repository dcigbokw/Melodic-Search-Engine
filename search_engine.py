import pickle
import numpy as np

INDEX_FILE = "search_index.pkl"

try:
    with open(INDEX_FILE, 'rb') as f:
        data = pickle.load(f)
        database = data.get("database", {})
except FileNotFoundError:
    database = {}

def calculate_levenshtein(q_intervals, t_intervals):
    """Calculates edit distance (insertions, deletions, substitutions)."""
    n, m = len(q_intervals), len(t_intervals)
    dp = np.zeros((n + 1, m + 1))
    
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if q_intervals[i-1] == t_intervals[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return int(dp[n][m])

def calculate_dtw(q_intervals, t_intervals):
    """Dynamic Time Warping: Tolerates rhythmic warping and ornamental notes."""
    n, m = len(q_intervals), len(t_intervals)
    dtw = np.full((n + 1, m + 1), float('inf'))
    dtw[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(q_intervals[i-1] - t_intervals[j-1])
            dtw[i, j] = cost + min(
                dtw[i-1, j],    # Insertion
                dtw[i, j-1],    # Deletion
                dtw[i-1, j-1]   # Match
            )
    return int(dtw[n, m])

def calculate_horizontal_intervals(midi_list):
    """Measures the melodic jumps from note to note."""
    if len(midi_list) < 2: return []
    return [midi_list[i] - midi_list[i-1] for i in range(1, len(midi_list))]

def calculate_vertical_intervals(midi_list):
    """Measures exact chord voicing and inversion relative to the bass note."""
    if not midi_list: return []
    # Ensure the notes are sorted lowest to highest to identify the bass
    sorted_midi = sorted(midi_list)
    bass = sorted_midi[0]
    return [n - bass for n in sorted_midi]

def advanced_search(query_notes, algorithm="dtw", target_composer="all", max_dist=5, mode="exact", max_results=5):
    # 1. Transform the query based on the selected mode
    if mode == "relative":
        target_sequence = calculate_horizontal_intervals(query_notes)
        db_key = "horizontal_intervals" 
    elif mode == "relative_vertical":
        target_sequence = calculate_vertical_intervals(query_notes)
        db_key = "vertical_intervals" 
    elif mode == "exact_vertical":
        target_sequence = sorted(query_notes) # Just sorts the exact pitches bottom-to-top
        db_key = "exact_vertical_notes"
    else:
        target_sequence = query_notes
        db_key = "raw_notes"
    
    results = []

    # 2. Iterate over the values in the loaded dictionary
    for entry in database.values():
        # Filter by composer
        if target_composer != "all" and entry.get("composer") != target_composer:
            continue
            
        db_sequence = entry.get(db_key, [])
        if not db_sequence:
            continue

        # 3. Calculate Distance
        if algorithm == "dtw":
            score = calculate_dtw(target_sequence, db_sequence)
        elif algorithm == "levenshtein":
            score = calculate_levenshtein(target_sequence, db_sequence)
        else:
            # Exact Trigram search fallback
            score = 0 if target_sequence == db_sequence else float('inf')

        # 4. Filter by the user's max distance threshold
        if score <= max_dist:
            # Intercept vertical searches to only return the single matched chord (Max 4 voices)
            if mode in ["exact_vertical", "relative_vertical"]:
                display_notes = entry.get("exact_vertical_notes", [])[:4] 
            else:
                display_notes = entry.get("raw_notes", [])

            results.append({
                "title": entry.get("title", "Unknown"),
                "composer": entry.get("composer", "Unknown"),
                "score": score,
                "raw_notes": display_notes # Safely overridden
            })

    # Sort results by closest match (lowest score)
    results.sort(key=lambda x: x["score"])

    # Slice the array to respect the user's max_results preference
    return results[:max_results]