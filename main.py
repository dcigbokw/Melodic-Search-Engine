from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from chord_generator import compose_chorale_2nd_order, transition_matrix
from search_engine import encode_intervals, search_bach_corpus
import random

app = FastAPI(
    title="Bach Generative AI & Search API",
    description="REST API for generating counterpoint and fuzzy-searching the Bach corpus.",
    version="1.0.0"
)

# ==========================================
# 1. THE GENERATOR ENDPOINT
# ==========================================
class GenerateRequest(BaseModel):
    num_chords: int = 16
    top_k: int = 5
    tonic_pc: int = 0

@app.get("/")
def read_root():
    return {"message": "Welcome to the Bach AI Engine API! Go to /docs to test the endpoints."}

@app.post("/generate")
async def generate_melody(req: GenerateRequest):
    # 1. FAIL-FAST: Ensure the matrix is actually loaded
    if not transition_matrix:
        raise HTTPException(status_code=500, detail="Matrix is empty or failed to load. Run train.py first.")
        
    # 2. DYNAMIC START SELECTION: Guarantee a valid starting chord
    # By picking from keys(), we know 100% this chord exists in the matrix.
    start_chord = random.choice(list(transition_matrix.keys()))
    
    # 3. THE RETRY LOOP: Now it only retries genuine downstream dead-ends
    for attempt in range(5):
        song = compose_chorale_2nd_order(start_chord, num_chords=req.num_chords, top_k=8)
        
        # If the engine successfully bypassed dead-ends and hit the target length
        if len(song) == req.num_chords:
            # (Insert your MIDI generation and return logic here)
            return {"status": "success", "chords": song}
            
    # 4. EXHAUSTED RETRIES
    raise HTTPException(
        status_code=500, 
        detail=f"Engine hit a harmonic dead end 5 times in a row starting from {start_chord}."
    )
# ==========================================
# 2. THE SEARCH ENDPOINT
# ==========================================
class SearchRequest(BaseModel):
    melody: List[int]
    max_distance: int = 1

@app.post("/search")
def search_corpus(req: SearchRequest):
    """
    Performs a high-speed fuzzy search across the indexed Bach corpus.
    """
    if len(req.melody) < 4: # We need 4 notes to make a 3-interval trigram
        raise HTTPException(status_code=400, detail="Melody must contain at least 4 notes for trigram indexing.")
        
    # Actually run the real search engine!
    search_results = search_bach_corpus(req.melody, req.max_distance)
    
    return {
        "status": "success",
        "query_melody": req.melody,
        "search_data": search_results
    }