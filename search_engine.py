from music21 import corpus
import pickle
import os

INDEX_FILE = "search_index.pkl"

# Load the index into memory on boot
if os.path.exists(INDEX_FILE):
    with open(INDEX_FILE, 'rb') as f:
        data = pickle.load(f)
        PHRASE_DB = data["database"]
        INVERTED_INDEX = data["inverted_index"]
else:
    PHRASE_DB = {}
    INVERTED_INDEX = {}

def get_trigrams(intervals):
    return [tuple(intervals[i:i+3]) for i in range(len(intervals)-2)]
# ==========================================
# THE ENCODER
# ==========================================
def encode_intervals(melody):
    """
    Takes a list of MIDI pitches and returns a list of the intervals 
    (the difference in semitones) between each consecutive note.
    """
    intervals = []
    for i in range(len(melody) - 1):
        intervals.append(melody[i + 1] - melody[i])
    return intervals

# ==========================================
# THE LEVENSHTEIN DISTANCE MATRIX
# ==========================================
def levenshtein_distance(seq1, seq2):
    """Calculates the Edit Distance between two interval arrays."""
    rows = len(seq1) + 1
    cols = len(seq2) + 1
    dp = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for i in range(1, rows):
        dp[i][0] = i
    for j in range(1, cols):
        dp[0][j] = j
        
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if seq1[i-1] == seq2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,       # Deletion
                dp[i][j-1] + 1,       # Insertion
                dp[i-1][j-1] + cost   # Substitution
            )
    return dp[rows-1][cols-1]

# ==========================================
# THE FUZZY SEARCH
# ==========================================
def fuzzy_search_melody(query_intervals, target_intervals, max_distance=1):
    """Slides a window to find matches within the max edit distance."""
    matches = []
    query_len = len(query_intervals)
    target_len = len(target_intervals)
    
    if query_len > target_len:
        return matches

    for i in range(target_len - query_len + 1):
        current_window = target_intervals[i : i + query_len]
        dist = levenshtein_distance(query_intervals, current_window)
        if dist <= max_distance:
            matches.append((i, dist))
            
    return matches

# ==========================================
# THE CORPUS DATA PIPELINE
# ==========================================
def search_bach_corpus(query_melody, max_distance=1):
    """
    A true production search engine: Filter-then-Verify.
    """
    if not PHRASE_DB:
        return {"error": "Search index not found. Run build_index.py first."}
        
    query_intervals = encode_intervals(query_melody)
    query_trigrams = get_trigrams(query_intervals)
    
    # 1. THE FILTER PHASE (O(1) lookups)
    # We only look at phrases that share AT LEAST ONE trigram with the query
    candidate_ids = set()
    for trigram in query_trigrams:
        if trigram in INVERTED_INDEX:
            candidate_ids.update(INVERTED_INDEX[trigram])
            
    if not candidate_ids:
        return {"matches_found": 0, "results": []}
        
    # 2. THE VERIFICATION PHASE (Levenshtein DP)
    # We only run the expensive math on the narrowed-down candidate list!
    final_results = []
    for phrase_id in candidate_ids:
        target_intervals = PHRASE_DB[phrase_id]["intervals"]
        
        # Run your existing fuzzy search function!
        matches = fuzzy_search_melody(query_intervals, target_intervals, max_distance)
        
        if matches:
            final_results.append({
                "title": PHRASE_DB[phrase_id]["title"],
                "phrase_id": phrase_id,
                "edit_distances": [dist for index, dist in matches]
            })
            
    return {
        "matches_found": len(final_results),
        "candidates_filtered": len(candidate_ids),
        "results": final_results
    }
# ==========================================
# RUN THE ENGINE!
# ==========================================
if __name__ == "__main__":
    # Query: Do-Re-Mi-Fa (e.g., C, D, E, F)
    # The intervals will be: [+2, +2, +1]
    my_query = [60, 62, 64, 65] 
    
    # Let's search 100 chorales with a strict exact match 
    print(search_bach_corpus(my_query, max_distance=0,))