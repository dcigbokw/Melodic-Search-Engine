import random
import os
import pickle
from music21 import corpus, note, chord, stream, midi

RHYTHM_INDEX_FILE = "rhythm_matrices.pkl"
all_rhythm_corpora = {}
transition_matrix_rhythm = {}

def train_rhythm_model():
    composers = ['bach', 'beethoven', 'chopin', 'mozart', 'tchaikovsky', 'handel']
    rhythm_matrices = {comp: {} for comp in composers}
    
    print("Extracting multi-composer rhythmic DNA...")
    for comp in composers:
        print(f"  -> Scanning {comp.upper()} rhythms...")
        try:
            bundles = corpus.getComposer(comp)
            for score_path in bundles[:15]:
                try:
                    score = corpus.parse(score_path)
                    top_part = score.parts[0]
                    durations = []
                    for el in top_part.flatten().notesAndRests:
                        if el.quarterLength > 0 and el.quarterLength <= 4.0: 
                            durations.append(float(el.quarterLength))
                            
                    for i in range(len(durations)-1):
                        state = durations[i]
                        next_state = durations[i+1]
                        if state not in rhythm_matrices[comp]:
                            rhythm_matrices[comp][state] = {}
                        if next_state not in rhythm_matrices[comp][state]:
                            rhythm_matrices[comp][state][next_state] = 0
                        rhythm_matrices[comp][state][next_state] += 1
                except Exception:
                    continue
        except Exception:
            pass
            
        for state, transitions in rhythm_matrices[comp].items():
            total = sum(transitions.values())
            for next_state in transitions:
                transitions[next_state] /= total
                
    with open(RHYTHM_INDEX_FILE, 'wb') as f:
        pickle.dump(rhythm_matrices, f)
    print("Rhythm DNA saved!")
    return rhythm_matrices

def load_rhythm_corpora():
    global all_rhythm_corpora, transition_matrix_rhythm
    if os.path.exists(RHYTHM_INDEX_FILE):
        with open(RHYTHM_INDEX_FILE, 'rb') as f:
            all_rhythm_corpora = pickle.load(f)
    else:
        all_rhythm_corpora = train_rhythm_model()
    
    transition_matrix_rhythm = all_rhythm_corpora.get('bach', {})

load_rhythm_corpora()

def interpolate_rhythm_matrices(weights):
    total = sum(weights.values())
    if total <= 0:
        return all_rhythm_corpora.get("bach", {})
        
    blended = {}
    for comp, w in weights.items():
        if w == 0: continue
        norm_w = w / total
        matrix = all_rhythm_corpora.get(comp, {})
        for state, trans in matrix.items():
            if state not in blended:
                blended[state] = {}
            for nxt, prob in trans.items():
                blended[state][nxt] = blended[state].get(nxt, 0.0) + (prob * norm_w)
    return blended

def generate_rhythms(num_chords, start_duration=1.0, rhythm_weights=None):
    if rhythm_weights is None:
        rhythm_weights = {"bach": 100}
        
    matrix = interpolate_rhythm_matrices(rhythm_weights)
    if not matrix:
        return [1.0] * num_chords
        
    rhythms = []
    curr = start_duration
    
    for i in range(num_chords):
        measure_pos = sum(rhythms) % 4.0
        space_left = 4.0 - measure_pos
        
        opts = list(matrix.get(curr, matrix.get(1.0, {1.0: 1})).keys())
        probs = list(matrix.get(curr, matrix.get(1.0, {1.0: 1})).values())
        
        valid_opts, valid_probs = [], []
        for opt, prob in zip(opts, probs):
            # QUANTIZER: Force block chords to whole metric beats. No syncopated fractions allowed here.
            if opt not in [1.0, 2.0, 3.0, 4.0]:
                continue
            
            # Prevent chords from awkwardly crossing the 4/4 barline
            if opt > space_left:
                continue
                
            valid_opts.append(opt)
            valid_probs.append(prob)
            
        if valid_opts:
            next_curr = random.choices(valid_opts, weights=valid_probs, k=1)[0]
        else:
            # Fallback: Just finish the measure cleanly with whatever space is left
            next_curr = max(1.0, space_left)
            
        rhythms.append(next_curr)
        curr = next_curr
        
    # Ensure the very last chord rings out to complete the final measure perfectly
    diff = sum(rhythms) % 4.0
    if diff != 0:
        rhythms[-1] += (4.0 - diff)
            
    return rhythms

def inject_passing_tones(song, rhythms, tonic_pc=0):
    # This is where the fast notes (0.5, 0.25) will go later—acting on single voices 
    # instead of full block chords.
    if len(song) >= 2 and len(rhythms) >= 2:
        if song[0][0] == 76 and rhythms[0] == 2.0:
            song.insert(1, (74, 67, 60, 48))
            rhythms[0] = 1.0
            rhythms.insert(1, 1.0)
    return song, rhythms

def export_to_midi_with_rhythm(song, rhythms, filename="output.mid"):
    s = stream.Score()
    p = stream.Part()
    for i, c_tuple in enumerate(song):
        valid_pitches = [note.Pitch(midi=m) for m in c_tuple if m != -1]
        c_obj = chord.Chord(valid_pitches)
        c_obj.quarterLength = rhythms[i] if i < len(rhythms) else 1.0
        p.append(c_obj)
    s.append(p)
    mf = midi.translate.streamToMidiFile(s)
    mf.open(filename, 'wb')
    mf.write()
    mf.close()