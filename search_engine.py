from music21 import corpus
# ==========================================
# 1. THE ENCODER
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
# 2. THE LEVENSHTEIN DISTANCE MATRIX
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
# 3. THE FUZZY SEARCH
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
# 4. THE CORPUS DATA PIPELINE
# ==========================================
def search_bach_corpus(query_melody, max_distance=1, max_songs=50):
    """
    Searches the Bach corpus, cleanly delimiting musical phrases by rests 
    so the sliding window never crosses an empty measure.
    """
    print(f"Encoding query and searching the first {max_songs} Bach chorales...")
    
    query_intervals = encode_intervals(query_melody)
    print(f"Query Intervals: {query_intervals}\n")
    
    bach_bundles = corpus.getComposer('bach')[:max_songs]
    total_matches = 0
    
    for score_path in bach_bundles:
        score = corpus.parse(score_path)
        soprano_part = score.parts[0]
        
        # THE DATA CLEANING PIPELINE
        phrases = []
        current_phrase = []
        
        # Extract both notes AND rests
        for element in soprano_part.flatten().notesAndRests:
            if element.isNote:
                current_phrase.append(int(element.pitch.ps))
            elif element.isRest:
                # If we hit a rest, save the phrase (if it has notes) and reset
                if len(current_phrase) > 0:
                    phrases.append(current_phrase)
                    current_phrase = []
                    
        # Catch the final phrase if the song doesn't end on a rest
        if len(current_phrase) > 0:
            phrases.append(current_phrase)
            
        # 2. THE SEARCH
        # Now we iterate through cleanly separated phrases instead of one massive track
        for phrase_index, phrase_pitches in enumerate(phrases):
            
            # We need at least 2 notes to make 1 interval!
            if len(phrase_pitches) < 2:
                continue
                
            target_intervals = encode_intervals(phrase_pitches)
            matches = fuzzy_search_melody(query_intervals, target_intervals, max_distance)
            
            if matches:
                total_matches += len(matches)
                
                # Robust title extraction
                if score.metadata:
                    title = score.metadata.title or score.metadata.movementName or str(score_path)
                else:
                    title = str(score_path)
                    
                print(f"Match found in: {title} (Phrase {phrase_index + 1})")
                for index, dist in matches:
                    print(f"  -> Starts at note index {index} within phrase (Edit Distance: {dist})")
                
    print(f"\nSearch complete! Found {total_matches} total matches.")

# ==========================================
# RUN THE ENGINE!
# ==========================================
if __name__ == "__main__":
    # Query: Do-Re-Mi-Fa (e.g., C, D, E, F)
    # The intervals will be: [+2, +2, +1]
    my_query = [60, 62, 64, 65] 
    
    # Let's search 100 chorales with a strict exact match (max_distance=0)
    search_bach_corpus(my_query, max_distance=0, max_songs=50)