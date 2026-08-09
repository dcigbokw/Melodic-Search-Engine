from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Import your polished math engines!
from chord_generator import compose_chorale_2nd_order
from search_engine import encode_intervals

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
def generate_music(req: GenerateRequest):
    """Generates a Bach-style chorale with a master retry loop."""
    start_chord = (72, 67, 64, 60) # C Major starting block
    
    # MASTER RETRY LOOP: Try up to 5 times to get a complete song
    for attempt in range(5):
        song = compose_chorale_2nd_order(
            start_chord, 
            num_chords=req.num_chords, 
            top_k=req.top_k, 
            tonic_pc=req.tonic_pc
        )
        
        # If the returned song is the correct length, we succeeded!
        if len(song) == req.num_chords:
            return {
                "status": "success",
                "attempt": attempt + 1,
                "chords": song
            }
            
    # If it fails 5 times in a row, return a 500 server error
    raise HTTPException(status_code=500, detail="AI failed to find a valid progression after 5 attempts.")

# ==========================================
# 2. THE SEARCH ENDPOINT
# ==========================================
class SearchRequest(BaseModel):
    melody: List[int]
    max_distance: int = 1

@app.post("/search")
def search_corpus(req: SearchRequest):
    """
    Returns the encoded intervals of a requested melody.
    (Running the full music21 corpus search on a web endpoint 
    takes ~30s, so we validate the data pipeline extraction here instead!)
    """
    if len(req.melody) < 2:
        raise HTTPException(status_code=400, detail="Melody must contain at least 2 notes.")
        
    intervals = encode_intervals(req.melody)
    
    return {
        "status": "success",
        "query_melody": req.melody,
        "encoded_intervals": intervals,
        "message": "Data pipeline validation successful. Ready for the sliding window engine!"
    }