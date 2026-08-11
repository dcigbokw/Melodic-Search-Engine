from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from chord_generator import compose_chorale_2nd_order, transition_matrix
from search_engine import encode_intervals, search_bach_corpus
from note_parser import parse_note
import random, os, uuid
from rhythm_ai import (
    generate_rhythms, 
    inject_passing_tones, 
    export_to_midi_with_rhythm,
    train_rhythm_model,
    transition_matrix_rhythm
)

app = FastAPI(
    title="Bach Generative AI & Search API",
    description="REST API for generating counterpoint and fuzzy-searching the Bach corpus.",
    version="1.0.0"
)

# Mount the static folder to serve assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# Override the root endpoint to serve the UI instead of the JSON message
@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

# If the matrix is empty on startup, train it in memory.
if not transition_matrix_rhythm:
    print("No rhythm matrix found in memory. Training model...")
    # This populates the global dictionary you imported
    transition_matrix_rhythm.update(train_rhythm_model())

# ==========================================
# 1. THE GENERATOR ENDPOINT
# ==========================================
class GenerateRequest(BaseModel):
    num_chords: int = 16
    top_k: int = 5
    tonic_pc: int = 0


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
    melody: List[str]
    max_distance: int = 1

@app.post("/search")
def search_corpus(req: SearchRequest):
    """
    Performs a high-speed fuzzy search across the indexed Bach corpus.
    """
    try:
        parsed_melody = [parse_note(token) for token in req.melody]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if len(parsed_melody) < 4:
        raise HTTPException(status_code=400, detail="Melody must contain at least 4 notes for trigram indexing.")

    search_results = search_bach_corpus(parsed_melody, req.max_distance)
    return {"status": "success", "query_melody": parsed_melody, "search_data": search_results}