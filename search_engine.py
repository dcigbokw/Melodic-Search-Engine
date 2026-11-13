import pickle
from collections import defaultdict

INDEX_FILE = "search_index.pkl"

try:
    with open(INDEX_FILE, 'rb') as f:
        data = pickle.load(f)
        database = data.get("database", {})
        inverted_index = data.get("inverted_index", {})
except FileNotFoundError:
    database = {}
    inverted_index = {}

def calculate_levenshtein(q_intervals, t_intervals):
    n, m = len(q_intervals), len(t_intervals)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if q_intervals[i-1] == t_intervals[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m]

def calculate_dtw(q_intervals, t_intervals):
    n, m = len(q_intervals), len(t_intervals)
    dtw = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    dtw[0][0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(q_intervals[i-1] - t_intervals[j-1])
            dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])
    return int(dtw[n][m])

def calculate_horizontal_intervals(midi_list):
    if len(midi_list) < 2: return []
    return [midi_list[i] - midi_list[i-1] for i in range(1, len(midi_list))]

def calculate_vertical_intervals(midi_list):
    if not midi_list: return []
    sorted_midi = sorted(midi_list)
    bass = sorted_midi[0]
    return [n - bass for n in sorted_midi]

def advanced_search(query_notes, algorithm="dtw", target_composer="all", max_dist=5, mode="exact", max_results=5):
    if mode == "relative":
        target_sequence = calculate_horizontal_intervals(query_notes)
        db_key = "horizontal_intervals" 
        target_type = "horizontal"
    elif mode == "exact":
        target_sequence = query_notes
        db_key = "raw_notes"
        target_type = "horizontal"
    elif mode == "relative_vertical":
        target_sequence = calculate_vertical_intervals(query_notes)
        db_key = "vertical_intervals" 
        target_type = "vertical"
    elif mode == "exact_vertical":
        target_sequence = sorted(query_notes) 
        db_key = "exact_vertical_notes"
        target_type = "vertical"
        
    candidates = defaultdict(set)
    
    if mode == "relative" and len(target_sequence) >= 3:
        query_trigrams = [tuple(target_sequence[i:i+3]) for i in range(len(target_sequence)-2)]
        for q_idx, trigram in enumerate(query_trigrams):
            if trigram in inverted_index:
                for pid, db_offset in inverted_index[trigram]:
                    start_offset = max(0, db_offset - q_idx)
                    candidates[pid].add(start_offset)
                    
        if len(candidates) < max_results:
            for pid, entry in database.items():
                if entry.get("type") == "horizontal":
                    seq_len = len(entry.get(db_key, []))
                    q_len = len(target_sequence)
                    for i in range(max(1, seq_len - q_len + 1)):
                        candidates[pid].add(i)
    else:
        for pid, entry in database.items():
            if entry.get("type") == target_type:
                if target_type == "horizontal":
                    seq_len = len(entry.get(db_key, []))
                    q_len = len(target_sequence)
                    for i in range(max(1, seq_len - q_len + 1)):
                        candidates[pid].add(i)
                else:
                    candidates[pid].add(0)
    
    results = []
    query_len = len(target_sequence)
    query_notes_len = len(query_notes)

    for pid, offsets in candidates.items():
        entry = database[pid]
        if target_composer != "all" and entry.get("composer") != target_composer:
            continue
            
        db_sequence = entry.get(db_key, [])
        if not db_sequence:
            continue

        best_local_score = float('inf')
        best_local_offset = 0
        best_raw_window = []

        for offset in offsets:
            if target_type == "horizontal":
                window = db_sequence[offset : offset + query_len]
                raw_notes_window = entry.get("raw_notes", [])[offset : offset + query_notes_len]
            else:
                window = db_sequence
                raw_notes_window = entry.get("exact_vertical_notes", [])

            if algorithm == "dtw":
                score = calculate_dtw(target_sequence, window)
            elif algorithm == "levenshtein":
                score = calculate_levenshtein(target_sequence, window)
            else:
                score = 0 if target_sequence == window else float('inf')

            if score < best_local_score:
                best_local_score = score
                best_local_offset = offset
                best_raw_window = raw_notes_window

        if best_local_score <= max_dist:
            results.append({
                "source_id": entry.get("source_id"),
                "title": entry.get("title", "Unknown"),
                "composer": entry.get("composer", "Unknown"),
                "score": best_local_score,
                "matched_offset": best_local_offset,
                "raw_notes": best_raw_window
            })

    best_results = {}
    for r in results:
        unique_key = (r["composer"], r["source_id"])
        if unique_key not in best_results or r["score"] < best_results[unique_key]["score"]:
            best_results[unique_key] = r
            
    final_list = list(best_results.values())
    final_list.sort(key=lambda x: x["score"])
    return final_list[:max_results]