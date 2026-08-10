from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from chord_generator import compose_chorale_2nd_order, transition_matrix
from search_engine import encode_intervals, search_bach_corpus
import random, os, uuid
from rhythm_ai import generate_rhythms, inject_passing_tones, export_to_midi_with_rhythm

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
def generate_melody(req: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Synchronous endpoint (def, not async def) to prevent CPU-bound 
    backtracking from blocking the FastAPI event loop.
    """
    if not transition_matrix:
        raise HTTPException(status_code=500, detail="Matrix is empty or failed to load. Run train.py first.")
        
    start_chord = random.choice(list(transition_matrix.keys()))
    tonic_pc = start_chord[3] % 12  # Auto-detect the key for passing tones
    
    for attempt in range(5):
        # Using req.top_k instead of hardcoding 8
        song = compose_chorale_2nd_order(start_chord, num_chords=req.num_chords, top_k=req.top_k)
        
        if len(song) == req.num_chords:
            # 1. Generate Rhythms
            rhythms = generate_rhythms(num_chords=len(song))
            
            # 2. Inject Passing Tones
            polished_song, polished_rhythms = inject_passing_tones(song, rhythms, tonic_pc=tonic_pc)
            
            # 3. Export to a unique temporary MIDI file
            temp_filename = f"generated_{uuid.uuid4().hex[:8]}.mid"
            export_to_midi_with_rhythm(polished_song, polished_rhythms, filename=temp_filename)
            
            # 4. Schedule the file for deletion AFTER the user downloads it
            background_tasks.add_task(os.remove, temp_filename)
            
            # 5. Return the playable MIDI file to the frontend
            return FileResponse(
                temp_filename, 
                media_type="audio/midi", 
                filename="bach_ai_chorale.mid"
            )
            
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