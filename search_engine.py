import pickle
import numpy as np

INDEX_FILE = "search_index.pkl"

try:
    with open(INDEX_FILE, 'rb') as f:
        data = pickle.load(f)
        database = data["database"]
        inverted_index = data["inverted_index"]
except FileNotFoundError:
    database, inverted_index = {}, {}

def encode_intervals(melody_pitches):
    return [melody_pitches[i+1] - melody_pitches[i] for i in range(len(melody_pitches)-1)]

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

def advanced_search(query_pitches, algorithm="dtw", target_composer="all", max_dist=2):
    """Filters by composer, then scores using the chosen algorithm."""
    if len(query_pitches) < 2: return []
    query_intervals = encode_intervals(query_pitches)
    
    results = []
    
    # Iterate over every phrase in the database
    for pid, phrase_data in database.items():
        # 1. Apply Composer Filter
        if target_composer != "all" and phrase_data["composer"] != target_composer:
            continue
            
        target_intervals = phrase_data["intervals"]
        
        # 2. Apply Algorithm Score
        if algorithm == "dtw":
            score = calculate_dtw(query_intervals, target_intervals)
        elif algorithm == "levenshtein":
            score = calculate_levenshtein(query_intervals, target_intervals)
        else: # Trigram (Exact Match Fallback)
            score = 0 if query_intervals == target_intervals else 999
            
        # 3. Keep matches within the threshold
        if score <= max_dist:
            results.append({
                "phrase_id": pid,
                "title": phrase_data["title"],
                "composer": phrase_data["composer"].capitalize(),
                "score": score,
                "raw_notes": phrase_data["raw_notes"]
            })
            
    # Sort by best score (lowest distance)
    results.sort(key=lambda x: x["score"])
    return results[:5] # Return top 5 matches