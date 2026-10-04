from music21 import corpus
import pickle
import os

INDEX_FILE = "search_index.pkl"

def encode_intervals(melody_pitches):
    return [melody_pitches[i+1] - melody_pitches[i] for i in range(len(melody_pitches)-1)]

def get_trigrams(intervals):
    return [tuple(intervals[i:i+3]) for i in range(len(intervals)-2)]

def build_search_index():
    print("Building the Multi-Corpus Search Index...")
    composers = ['bach', 'beethoven', 'chopin', 'mozart', 'tchaikovsky', 'handel']
    
    database = {}       
    inverted_index = {} 
    phrase_id = 0
    
    for comp in composers:
        print(f"\nIndexing corpus: {comp.upper()}")
        bundles = corpus.getComposer(comp) 
        
        for idx, score_path in enumerate(bundles):
            try:
                score = corpus.parse(score_path)
                # Grab the top part (Soprano/Melody)
                top_part = score.parts[0] 
                
                current_phrase = []
                for element in top_part.flatten().notesAndRests:
                    if element.isNote:
                        current_phrase.append(int(element.pitch.ps))
                    elif element.isRest:
                        if len(current_phrase) > 3: 
                            intervals = encode_intervals(current_phrase)
                            
                            # Save to Database with Composer Tag and Raw Notes
                            database[phrase_id] = {
                                "title": score.metadata.title or os.path.basename(str(score_path)),
                                "composer": comp,
                                "intervals": intervals,
                                "raw_notes": current_phrase # Saved so we can render the sheet music later!
                            }
                            
                            for trigram in get_trigrams(intervals):
                                if trigram not in inverted_index:
                                    inverted_index[trigram] = set()
                                inverted_index[trigram].add(phrase_id)
                                
                            phrase_id += 1
                        current_phrase = []
            except Exception as e:
                pass # Skip files that fail to parse

    print("\nSaving Multi-Corpus Index to disk...")
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump({"database": database, "inverted_index": inverted_index}, f)
        
    print(f"Success! Indexed {len(database)} phrases.")

if __name__ == "__main__":
    build_search_index()