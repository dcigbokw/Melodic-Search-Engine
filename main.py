from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import json
import random
import os
import asyncio
import uuid
from chord_generator import compose_chorale_2nd_order, transition_matrix, all_corpora, is_valid_transition
from search_engine import encode_intervals, advanced_search
from note_parser import parse_note
from rules_engine import audit_human_sequence
from rhythm_ai import (
    generate_rhythms, 
    inject_passing_tones, 
    export_to_midi_with_rhythm,
    train_rhythm_model,
    transition_matrix_rhythm
)
from music21 import chord, tempo, stream, midi, pitch, note, instrument

app = FastAPI(
    title="Bach Generative AI & Search API",
    description="REST API for generating counterpoint and fuzzy-searching the Bach corpus.",
    version="1.0.0"
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

if not transition_matrix_rhythm:
    print("No rhythm matrix found in memory. Training model...")
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
    if not transition_matrix:
        raise HTTPException(status_code=500, detail="Matrix is empty or failed to load. Run train.py first.")
        
    tonic_pc = req.tonic_pc % 12
    candidate_starts = [c for c in transition_matrix.keys() if c[3] % 12 == tonic_pc]

    if not candidate_starts:
        raise HTTPException(
            status_code=400,
            detail=f"No trained chords found with a bass note matching tonic_pc={tonic_pc}."
        )

    start_chord = random.choice(candidate_starts)

    for attempt in range(5):
        song = compose_chorale_2nd_order(
            start_chord, num_chords=req.num_chords, top_k=req.top_k, tonic_pc=tonic_pc
        )
        
        if len(song) == req.num_chords:
            rhythms = generate_rhythms(num_chords=len(song))
            polished_song, polished_rhythms = inject_passing_tones(song, rhythms, tonic_pc=tonic_pc)
            
            temp_filename = f"generated_{uuid.uuid4().hex[:8]}.mid"
            export_to_midi_with_rhythm(polished_song, polished_rhythms, filename=temp_filename)
            background_tasks.add_task(os.remove, temp_filename)
            
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
# 2. SATB TRANSLATOR HELPERS (Co-Creative UI)
# ==========================================
def midi_tuple_to_satb(midi_tuple):
    if midi_tuple[0] == -1:
        return ["rest", "rest", "rest", "rest"]
        
    satb_strings = []
    for midi_val in midi_tuple:
        if midi_val == -1:
            satb_strings.append("rest")
        else:
            p = pitch.Pitch()
            p.midi = midi_val
            satb_strings.append(p.nameWithOctave) 
    return satb_strings

def satb_to_midi_tuple(satb_list):
    midi_vals = []
    for note_str in satb_list:
        note_str = note_str.strip()
        if note_str.lower() == "rest":
            midi_vals.append(-1)
        else:
            try:
                p = pitch.Pitch(note_str)
                midi_vals.append(p.midi)
            except Exception:
                midi_vals.append(60) 
    return tuple(midi_vals)

# ==========================================
# 3. THE CO-CREATIVE EXTEND ENDPOINT (SATB)
# ==========================================
class ExtendRequest(BaseModel):
    current_chords: List[List[str]]   
    num_to_add: int = 4
    composer_weights: dict = {"bach": 100, "beethoven": 0, "chopin": 0, "mozart": 0, "tchaikovsky": 0, "handel":0}
    tempo: str = "andante"
    dynamics: str = "mf"

def interpolate_matrices(matrices_dict, user_weights):
    total_weight = sum(user_weights.values())
    if total_weight <= 0:
        return matrices_dict.get("bach", {}).get("first_order", {})
        
    normalized_weights = {k: v / total_weight for k, v in user_weights.items()}
    blended_matrix = {}
    
    for composer, comp_weight in normalized_weights.items():
        if comp_weight == 0:
            continue
            
        matrix = matrices_dict.get(composer, {}).get("first_order", {})
        for state, transitions in matrix.items():
            if state not in blended_matrix:
                blended_matrix[state] = {}
            
            for next_state, prob in transitions.items():
                weighted_prob = prob * comp_weight
                if next_state not in blended_matrix[state]:
                    blended_matrix[state][next_state] = 0.0
                blended_matrix[state][next_state] += weighted_prob
                
    return blended_matrix

@app.post("/extend")
def extend_composition(req: ExtendRequest, background_tasks: BackgroundTasks):
    dynamic_matrix = interpolate_matrices(all_corpora, req.composer_weights)
    
    if not dynamic_matrix:
        raise HTTPException(status_code=500, detail="Matrix is empty.")
        
    if len(req.current_chords) < 1:
        raise HTTPException(status_code=400, detail="Provide at least 1 SATB block.")

    human_sequence = [satb_to_midi_tuple(block) for block in req.current_chords]
    
    # Audit the human edits before proceeding
    is_clean, error_msg = audit_human_sequence(human_sequence, is_valid_transition)
    if not is_clean:
        raise HTTPException(status_code=400, detail=error_msg)
    
    valid_chords = [c for c in human_sequence if c[0] != -1]
    if not valid_chords:
        current_state = random.choice(list(dynamic_matrix.keys()))
        if isinstance(current_state, tuple) and isinstance(current_state[0], tuple):
            current_state = current_state[-1]
    else:
        current_state = valid_chords[-1] 
    
    ai_extension = []
    
    for _ in range(req.num_to_add):
        if current_state in dynamic_matrix:
            next_options = dynamic_matrix[current_state]
            choices = list(next_options.keys())
            weights = list(next_options.values())
            
            # Select next chord with a fallback if rules fail
            valid_next = None
            for _ in range(10): # Try up to 10 weighted random selections to find a valid counterpoint rule match
                candidate = random.choices(choices, weights=weights, k=1)[0]
                if is_valid_transition(current_state, candidate):
                    valid_next = candidate
                    break
            
            if not valid_next:
                valid_next = random.choices(choices, weights=weights, k=1)[0] # Fallback to raw probabilities if strict rules corner it
                
            ai_extension.append(valid_next)
            current_state = valid_next
        else:
            next_chord = random.choice(list(dynamic_matrix.keys()))
            if isinstance(next_chord, tuple) and isinstance(next_chord[0], tuple):
                next_chord = next_chord[-1]
            ai_extension.append(next_chord)
            current_state = next_chord
            
    full_song_tuples = human_sequence + ai_extension
    full_song_strings = [midi_tuple_to_satb(c) for c in full_song_tuples]
    
    tempo_map = {"adagio": 60, "andante": 90, "allegro": 130}
    velocity_map = {"piano": 40, "mf": 75, "forte": 110}
    
    target_bpm = tempo_map.get(req.tempo, 90)
    target_velocity = velocity_map.get(req.dynamics, 75)
    
    s = stream.Score()
    p = stream.Part()
    p.append(tempo.MetronomeMark(number=target_bpm))
    
    for c_tuple in full_song_tuples:
        if c_tuple[0] == -1:
            r = chord.Rest()
            r.quarterLength = 1.0
            p.append(r)
        else:
            valid_pitches = [pitch.Pitch(midi=m) for m in c_tuple if m != -1]
            c_obj = chord.Chord(valid_pitches)
            c_obj.quarterLength = 1.0 
            for n in c_obj.notes:
                n.volume.velocity = target_velocity
            p.append(c_obj)
        
    s.append(p)
    
    temp_filename = f"static/collab_{uuid.uuid4().hex[:8]}.mid"
    mf = midi.translate.streamToMidiFile(s)
    mf.open(temp_filename, 'wb')
    mf.write()
    mf.close()
    
    background_tasks.add_task(delayed_cleanup, temp_filename, 15)
    
    headers = {"X-Generated-Chords": json.dumps(full_song_strings)}
    
    with open(temp_filename, "rb") as f:
        midi_data = f.read()
    return Response(content=midi_data, media_type="audio/midi", headers=headers)

# ==========================================
# 4. THE SEARCH ENDPOINT
# ==========================================
async def delayed_cleanup(filepath: str, delay: int = 15):
    await asyncio.sleep(delay)
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except OSError:
        pass

class SearchRequest(BaseModel):
    melody: List[str]
    algorithm: str = "dtw"
    composer: str = "all"
    max_distance: int = 5
    instrument_name: str = "piano" 

@app.post("/search")
def search_corpus(req: SearchRequest, background_tasks: BackgroundTasks):
    try:
        parsed_melody = [parse_note(token) for token in req.melody]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    matches = advanced_search(
        parsed_melody, 
        algorithm=req.algorithm, 
        target_composer=req.composer, 
        max_dist=req.max_distance
    )
    
    formatted_matches = []
    os.makedirs("static", exist_ok=True)
    
    inst_map = {
        "piano": instrument.Piano(),
        "harpsichord": instrument.Harpsichord(),
        "violin": instrument.Violin(),
        "cello": instrument.Violoncello(),
        "flute": instrument.Flute(),
        "organ": instrument.PipeOrgan()
    }
    
    for match in matches:
        s = stream.Score()
        p = stream.Part()
        
        chosen_inst = inst_map.get(req.instrument_name.lower(), instrument.Piano())
        p.insert(0, chosen_inst)
        
        for midi_val in match["raw_notes"]:
            n = note.Note()
            n.pitch.midi = midi_val
            n.quarterLength = 1.0
            p.append(n)
        s.append(p)
        
        temp_filename = f"static/match_{uuid.uuid4().hex[:8]}.mid"
        mf = midi.translate.streamToMidiFile(s)
        mf.open(temp_filename, 'wb')
        mf.write()
        mf.close()
        
        background_tasks.add_task(delayed_cleanup, temp_filename, 15)
        
        formatted_matches.append({
            "title": match["title"],
            "composer": match["composer"],
            "score": match["score"],
            "midi_url": f"/{temp_filename}" 
        })

    return {
        "status": "success", 
        "algorithm": req.algorithm,
        "matches": formatted_matches
    }