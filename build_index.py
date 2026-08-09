from music21 import corpus
import pickle
import os

INDEX_FILE = "search_index.pkl"

def encode_intervals(melody_pitches):
    return [melody_pitches[i+1] - melody_pitches[i] for i in range(len(melody_pitches)-1)]

def get_trigrams(intervals):
    """Slices a list of intervals into overlapping chunks of 3."""
    return [tuple(intervals[i:i+3]) for i in range(len(intervals)-2)]

def build_search_index():
    print("Building the Melodic Search Inverted Index...")
    bach_bundles = corpus.getComposer('bach')
    
    database = {}       # Stores the actual phrase data
    inverted_index = {} # Maps trigrams -> Set of phrase IDs
    phrase_id = 0
    
    # We'll index the first 150 chorales to match our generator training size
    for idx, score_path in enumerate(bach_bundles[:150]):
        print(f"Indexing {idx+1}/150: {score_path}")
        score = corpus.parse(score_path)
        soprano_part = score.parts[0]
        
        current_phrase = []
        for element in soprano_part.flatten().notesAndRests:
            if element.isNote:
                current_phrase.append(int(element.pitch.ps))
            elif element.isRest:
                if len(current_phrase) > 3: # Need at least 4 notes for a trigram of intervals!
                    intervals = encode_intervals(current_phrase)
                    
                    # 1. Save to Database
                    database[phrase_id] = {
                        "title": score.metadata.title or os.path.basename(str(score_path)),
                        "intervals": intervals
                    }
                    
                    # 2. Build the Inverted Index
                    for trigram in get_trigrams(intervals):
                        if trigram not in inverted_index:
                            inverted_index[trigram] = set()
                        inverted_index[trigram].add(phrase_id)
                        
                    phrase_id += 1
                current_phrase = []

    print("\nSaving Index to disk...")
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump({"database": database, "inverted_index": inverted_index}, f)
        
    print(f"Success! Indexed {len(database)} phrases and {len(inverted_index)} unique trigrams.")

if __name__ == "__main__":
    build_search_index()