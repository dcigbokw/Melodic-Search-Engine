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
import math
from chord_generator import compose_chorale_2nd_order, transition_matrix, all_corpora, is_valid_transition
from search_engine import advanced_search, calculate_dtw, calculate_levenshtein
from note_parser import parse_note
from rules_engine import audit_human_sequence
from music_scorer import score_composition
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

async def delayed_cleanup(filepath: str, delay: int = 15):
    await asyncio.sleep(delay)
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except OSError:
        pass

# ==========================================
# 1. THE GENERATOR ENDPOINT
# ==========================================
# ==========================================
# 1. THE GENERATOR ENDPOINT
# ==========================================
class GenerateRequest(BaseModel):
    num_chords: int = 16
    composer: str = "bach"
    tonic_pc: int = 0
    tempo: str = "andante"
    dynamics: str = "mf"

@app.post("/generate")
def generate_melody(req: GenerateRequest, background_tasks: BackgroundTasks):
    # Route user selection to a 100% weight for the chosen composer
    comp_weights = {comp: 100 if comp == req.composer else 0 for comp in ['bach', 'beethoven', 'chopin', 'mozart', 'tchaikovsky', 'handel']}
    
    matrix_1st = interpolate_matrices(all_corpora, comp_weights, "first_order")
    matrix_2nd = interpolate_matrices(all_corpora, comp_weights, "second_order")
    matrix_3rd = interpolate_matrices(all_corpora, comp_weights, "third_order")
    
    if not matrix_1st:
        raise HTTPException(status_code=500, detail="Matrix is empty. Run train.py first.")
        
    candidate_starts = [c for c in matrix_1st.keys() if isinstance(c, tuple) and len(c) == 4 and c[3] != -1 and c[3] % 12 == req.tonic_pc]
    if not candidate_starts:
        raise HTTPException(status_code=400, detail=f"No trained chords found matching tonic_pc={req.tonic_pc}.")

    start_chord = random.choice(candidate_starts)
    fallback_options = [c for c in matrix_1st.keys() if isinstance(c, tuple) and c[0] != -1]
    successful_drafts = []
    
    # Background Parameters (Hidden from UI)
    markov_order = 2
    temperature = 1.0
    range_expansion = 0
    cadence_pref = "authentic"
    RANGES = {0: (60, 81), 1: (53, 74), 2: (48, 69), 3: (40, 62)}
    
    def dfs_generate(path, visited):
        if len(successful_drafts) >= 5: return
        
        current_depth = len(path) - 1
        if current_depth == req.num_chords - 1:
            successful_drafts.append(path.copy())
            return
            
        opts_dict = {}
        target_order = markov_order
        
        while target_order >= 1 and not opts_dict:
            if target_order == 3 and len(path) >= 3:
                opts_dict = matrix_3rd.get((path[-3], path[-2], path[-1]), {})
            elif target_order == 2 and len(path) >= 2:
                opts_dict = matrix_2nd.get((path[-2], path[-1]), {})
            elif target_order == 1:
                opts_dict = matrix_1st.get(path[-1], {})
            target_order -= 1
            
        valid_keys = [k for k in opts_dict.keys() if isinstance(k, tuple) and k[0] != -1]
        opts = sorted(valid_keys, key=lambda k: math.log(random.random() + 1e-9) / (opts_dict[k] ** (1.0 / max(0.1, temperature))), reverse=True)
        
        if len(opts) < 5 and fallback_options:
            opts.extend(random.sample(fallback_options, min(20, len(fallback_options))))
        
        for cand in opts:
            cand_state = cand[-1] if isinstance(cand, tuple) and isinstance(cand[0], tuple) else cand
            curr_state = path[-1]
            
            out_of_range = False
            if cand_state[0] != -1:
                for v in range(4):
                    min_pitch, max_pitch = RANGES[v]
                    if not (min_pitch - range_expansion <= cand_state[v] <= max_pitch + range_expansion):
                        out_of_range = True
                        break
            if out_of_range: continue

            if cand_state[0] != -1 and cadence_pref != "none":
                is_penultimate = (current_depth == req.num_chords - 2)
                is_ultimate = (current_depth == req.num_chords - 1)
                bass_pc = cand_state[3] % 12
                
                if cadence_pref == "authentic":
                    if is_penultimate and bass_pc != (req.tonic_pc + 7) % 12: continue
                    if is_ultimate and bass_pc != req.tonic_pc: continue
            
            if is_valid_transition(curr_state, cand_state, req.tonic_pc):
                h = hash(tuple(path + [cand_state]))
                if h in visited: continue
                visited.add(h)
                path.append(cand_state)
                dfs_generate(path, visited)
                path.pop()

    dfs_generate([start_chord], set())
    
    if successful_drafts:
        best_draft = None
        highest_score = -9999
        for draft in successful_drafts:
            draft_score = score_composition(draft, matrix_1st)
            if draft_score > highest_score:
                highest_score = draft_score
                best_draft = draft
        song = best_draft
    else:
        raise HTTPException(status_code=500, detail="Engine failed to find a valid progression resolving to the target key.")

    rhythms = generate_rhythms(num_chords=len(song), rhythm_weights=comp_weights)
    polished_song, polished_rhythms = inject_passing_tones(song, rhythms, tonic_pc=req.tonic_pc)
    
    tempo_map = {"adagio": 60, "andante": 90, "allegro": 130}
    velocity_map = {"piano": 40, "mf": 75, "forte": 110}
    target_bpm = tempo_map.get(req.tempo, 90)
    target_velocity = velocity_map.get(req.dynamics, 75)

    s = stream.Score()
    p = stream.Part()
    p.append(tempo.MetronomeMark(number=target_bpm))
    
    for i, c_tuple in enumerate(polished_song):
        current_duration = polished_rhythms[i] if i < len(polished_rhythms) else 1.0
        if all(n == -1 for n in c_tuple):
            r = chord.Rest()
            r.quarterLength = current_duration
            p.append(r)
        else:
            valid_pitches = [pitch.Pitch(midi=m) for m in c_tuple if m != -1]
            c_obj = chord.Chord(valid_pitches)
            c_obj.quarterLength = current_duration 
            for n in c_obj.notes: n.volume.velocity = target_velocity
            p.append(c_obj)
            
    s.append(p)
    temp_filename = f"static/generated_{uuid.uuid4().hex[:8]}.mid"
    mf = midi.translate.streamToMidiFile(s)
    mf.open(temp_filename, 'wb')
    mf.write()
    mf.close()
    
    background_tasks.add_task(delayed_cleanup, temp_filename, 7)
    with open(temp_filename, "rb") as f:
        midi_data = f.read()
    return Response(content=midi_data, media_type="audio/midi")
# ==========================================
# 2. SATB TRANSLATOR HELPERS
# ==========================================
def midi_tuple_to_satb(midi_tuple):
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
# 3. THE CO-CREATIVE EXTEND ENDPOINT
# ==========================================
class ExtendRequest(BaseModel):
    current_chords: List[List[str]]   
    num_to_add: int = 4
    composer_weights: dict = {"bach": 100, "beethoven": 0, "chopin": 0, "mozart": 0, "tchaikovsky": 0, "handel":0}
    rhythm_weights: dict = {"bach": 100, "beethoven": 0, "chopin": 0, "mozart": 0, "tchaikovsky": 0, "handel":0}
    tempo: str = "andante"
    dynamics: str = "mf"
    markov_order: int = 1
    temperature: float = 1.0       # Softmax multiplier
    range_expansion: int = 0       # Vocal tessitura expansion
    cadence_pref: str = "none"     # Force V-I, I-V, etc.
    tonic_pc: int = 0              # Key definition for cadences

def interpolate_matrices(matrices_dict, user_weights, order_key="first_order"):
    """Blends N-order Markov matrices based on UI sliders."""
    total_weight = sum(user_weights.values())
    if total_weight <= 0:
        return matrices_dict.get("bach", {}).get(order_key, {})
        
    normalized_weights = {k: v / total_weight for k, v in user_weights.items()}
    blended_matrix = {}
    
    for composer, comp_weight in normalized_weights.items():
        if comp_weight == 0: continue
        matrix = matrices_dict.get(composer, {}).get(order_key, {})
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
@app.post("/extend")
def extend_composition(req: ExtendRequest, background_tasks: BackgroundTasks):
    matrix_1st = interpolate_matrices(all_corpora, req.composer_weights, "first_order")
    matrix_2nd = interpolate_matrices(all_corpora, req.composer_weights, "second_order")
    matrix_3rd = interpolate_matrices(all_corpora, req.composer_weights, "third_order")
    
    if not matrix_1st:
        raise HTTPException(status_code=500, detail="Matrix is empty. Run train.py")
        
    if len(req.current_chords) < 1:
        raise HTTPException(status_code=400, detail="Provide at least 1 SATB block.")

    human_sequence = [satb_to_midi_tuple(block) for block in req.current_chords]
    
    is_clean, error_msg = audit_human_sequence(human_sequence, is_valid_transition)
    if not is_clean:
        raise HTTPException(status_code=400, detail=error_msg)
    
    current_state = human_sequence[-1]
    fallback_options = [c for c in matrix_1st.keys() if isinstance(c, tuple) and c[0] != -1]
    successful_drafts = [] 
    
    RANGES = {0: (60, 81), 1: (53, 74), 2: (48, 69), 3: (40, 62)}
    
    def dfs_extend(path, visited):
        if len(successful_drafts) >= 5:
            return
            
        # We now calculate depth based on AI additions, since 'path' contains the full history
        current_depth = len(path) - len(human_sequence) 
        
        if current_depth == req.num_to_add:
            successful_drafts.append(path.copy()) # Must copy the path!
            return
        
        opts_dict = {}
        target_order = req.markov_order
        
        # KATZ BACKOFF: Try highest requested order first, back off if matrix is sparse
        while target_order >= 1 and not opts_dict:
            if target_order == 3 and len(path) >= 3:
                state = (path[-3], path[-2], path[-1])
                opts_dict = matrix_3rd.get(state, {})
            elif target_order == 2 and len(path) >= 2:
                state = (path[-2], path[-1])
                opts_dict = matrix_2nd.get(state, {})
            elif target_order == 1:
                state = path[-1]
                opts_dict = matrix_1st.get(state, {})
                
            target_order -= 1
            
        valid_keys = [k for k in opts_dict.keys() if isinstance(k, tuple) and k[0] != -1]
        
        temp = max(0.1, req.temperature)
        opts = sorted(valid_keys, key=lambda k: math.log(random.random() + 1e-9) / (opts_dict[k] ** (1.0 / temp)), reverse=True)
            
        if len(opts) < 5 and fallback_options:
            opts.extend(random.sample(fallback_options, min(20, len(fallback_options))))
        
        for cand in opts:
            cand_state = cand[-1] if isinstance(cand, tuple) and isinstance(cand[0], tuple) else cand
            curr_state = path[-1] # Compare to the immediate previous chord
            
            # Note Range limit check
            out_of_range = False
            if cand_state[0] != -1:
                for v in range(4):
                    min_pitch, max_pitch = RANGES[v]
                    if not (min_pitch - req.range_expansion <= cand_state[v] <= max_pitch + req.range_expansion):
                        out_of_range = True
                        break
            if out_of_range: continue

            # Cadence Forcing
            if cand_state[0] != -1 and req.cadence_pref != "none":
                is_penultimate = (current_depth == req.num_to_add - 1)
                is_ultimate = (current_depth == req.num_to_add)
                bass_pc = cand_state[3] % 12
                
                if req.cadence_pref == "authentic":
                    if is_penultimate and bass_pc != (req.tonic_pc + 7) % 12: continue 
                    if is_ultimate and bass_pc != req.tonic_pc: continue 
                elif req.cadence_pref == "half":
                    if is_ultimate and bass_pc != (req.tonic_pc + 7) % 12: continue 
                elif req.cadence_pref == "plagal":
                    if is_penultimate and bass_pc != (req.tonic_pc + 5) % 12: continue 
                    if is_ultimate and bass_pc != req.tonic_pc: continue 

            if is_valid_transition(curr_state, cand_state, req.tonic_pc):
                h = hash(tuple(path + [cand_state]))
                if h in visited:
                    continue
                visited.add(h)
                
                # Append, Recurse, and Pop for N-Order tracking
                path.append(cand_state)
                dfs_extend(path, visited)
                path.pop()

    # Pass the full human sequence so higher orders can look backwards
    dfs_extend(human_sequence.copy(), set())
    
    if successful_drafts:
        best_draft = None
        highest_score = -9999
        for draft in successful_drafts:
            draft_score = score_composition(draft, matrix_1st) # Score against 1st order probabilities
            if draft_score > highest_score:
                highest_score = draft_score
                best_draft = draft
                
        # Strip the human sequence back out
        ai_extension = best_draft[len(human_sequence):] 
    else:
        ai_extension = [tuple(n + 4 if n != -1 else -1 for n in current_state)] * req.num_to_add
            
    full_song_tuples = human_sequence + ai_extension
    full_song_strings = [midi_tuple_to_satb(c) for c in full_song_tuples]
    
    tempo_map = {"adagio": 60, "andante": 90, "allegro": 130}
    velocity_map = {"piano": 40, "mf": 75, "forte": 110}
    target_bpm = tempo_map.get(req.tempo, 90)
    target_velocity = velocity_map.get(req.dynamics, 75)

    human_rhythms = [1.0] * len(human_sequence)
    ai_rhythms = generate_rhythms(num_chords=len(ai_extension), rhythm_weights=req.rhythm_weights)
    full_rhythms = human_rhythms + ai_rhythms

    s = stream.Score()
    p = stream.Part()
    p.append(tempo.MetronomeMark(number=target_bpm))
    
    for i, c_tuple in enumerate(full_song_tuples):
        current_duration = full_rhythms[i] 
        if all(n == -1 for n in c_tuple):
            r = chord.Rest()
            r.quarterLength = current_duration
            p.append(r)
        else:
            valid_pitches = [pitch.Pitch(midi=m) for m in c_tuple if m != -1]
            c_obj = chord.Chord(valid_pitches)
            c_obj.quarterLength = current_duration 
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
class SearchRequest(BaseModel):
    melody: List[str]
    algorithm: str = "dtw"
    composer: str = "all"
    max_distance: int = 5
    instrument_name: str = "piano" 
    search_mode: str = "exact"      
    max_results: int = 5
    playback_style: str = "default"            

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
        max_dist=req.max_distance,
        mode=req.search_mode,        
        max_results=req.max_results  
    )

    formatted_matches = []
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
        
        is_chord = False
        if req.playback_style == "chord":
            is_chord = True
        elif req.playback_style == "arpeggio":
            is_chord = False
        else: 
            if req.search_mode in ["exact_vertical", "relative_vertical"]:
                is_chord = True

        if is_chord:
            c_obj = chord.Chord([pitch.Pitch(midi=m) for m in match["raw_notes"]])
            c_obj.quarterLength = 2.0  
            p.append(c_obj)
        else:
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