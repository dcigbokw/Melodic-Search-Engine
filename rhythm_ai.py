from music21 import corpus, chord, stream
import random
from chord_generator import compose_chorale_2nd_order

transition_matrix_rhythm = {}

def train_rhythm_model(num_chorales=150):
    """Parses the Bach corpus and builds the rhythm transition matrix in memory."""
    global transition_matrix_rhythm
    transition_counts_rhythm = {}
    bach_bundles = corpus.getComposer('bach')[:num_chorales]
    print(f"Training Rhythm Matrix on {len(bach_bundles)} chorales...")

    for score_path in bach_bundles:
        score = corpus.parse(score_path)
        raw_chords = score.chordify().flatten().getElementsByClass(chord.Chord)
        durations = [float(c.quarterLength) for c in raw_chords]
        for i in range(len(durations) - 1):
            current, nxt = durations[i], durations[i + 1]
            transition_counts_rhythm.setdefault(current, {})
            transition_counts_rhythm[current][nxt] = transition_counts_rhythm[current].get(nxt, 0) + 1

    matrix = {}
    for current, next_counts in transition_counts_rhythm.items():
        total = sum(next_counts.values())
        matrix[current] = {d: c / total for d, c in next_counts.items()}

    transition_matrix_rhythm = matrix
    return matrix

def generate_rhythms(num_chords, start_duration=1.0):
    rhythm_track = [start_duration]
    
    # Track where we are in the measure (0.0 to 4.0)
    current_measure_beats = start_duration
    
    # We subtract 1 because we already have the starting duration
    for _ in range(num_chords - 1):
        current_duration = rhythm_track[-1]
        
        # Grab predictions
        if current_duration in transition_matrix_rhythm:
            predictions = transition_matrix_rhythm[current_duration]
        else:
            predictions = {1.0: 1.0} 
            
        valid_options = []
        valid_weights = []
        
        # 1. FILTERING: Check if the candidate fits in the measure
        for cand_duration, prob in predictions.items():
            
            # If current beats + candidate duration is <= 4.0, it fits perfectly
            if current_measure_beats + cand_duration <= 4.0:
                valid_options.append(cand_duration)
                valid_weights.append(prob)
                
        # SAFETY FALLBACK: If Bach's suggestions overflow the measure,
        # we force a duration that perfectly fills the rest of the measure.
        if not valid_options:
            forced_duration = 4.0 - current_measure_beats
            valid_options = [forced_duration]
            valid_weights = [1.0]
            
        # Roll the weighted die
        chosen_duration = random.choices(valid_options, weights=valid_weights, k=1)[0]
        rhythm_track.append(chosen_duration)
        
        # Add the chosen duration to our measure tracker
        current_measure_beats += chosen_duration
        
        # If we perfectly hit 4.0, the measure is full! Reset for the next measure.
        if current_measure_beats == 4.0:
            current_measure_beats = 0.0
            
    return rhythm_track


def inject_passing_tones(song, rhythms, tonic_pc=0):
    """
    Scans a generated chorale and rhythm track. If the Soprano jumps by a 3rd,
    it dynamically splits the rhythm and injects a diatonic passing tone.
    """
    new_song = []
    new_rhythms = []
    
    # Define the Major Scale intervals so we only pick notes in our key
    # C Major = [0, 2, 4, 5, 7, 9, 11]
    major_scale_pcs = [(interval + tonic_pc) % 12 for interval in [0, 2, 4, 5, 7, 9, 11]]
    
    # Loop through every chord except the final resolution chord
    for i in range(len(song) - 1):
        curr_chord = song[i]
        next_chord = song[i + 1]
        curr_rhythm = rhythms[i]
        
        # Grab the Soprano notes (Index 0)
        s1 = curr_chord[0]
        s2 = next_chord[0]
        
        # Calculate the absolute distance
        distance = abs(s1 - s2)
        
        # RULE: Is it a 3rd? (3 or 4 semitones) AND is the note long enough to split?
        if distance in [3, 4] and curr_rhythm >= 1.0:
            
            # Find the pitches strictly between s1 and s2
            lower_note = min(s1, s2)
            upper_note = max(s1, s2)
            
            passing_tone = None
            for p in range(lower_note + 1, upper_note):
                if p % 12 in major_scale_pcs:
                    passing_tone = p
                    break  # Found our valid scale note!
                    
            if passing_tone:
                # 1. Append the original chord, but cut its rhythm in half
                split_duration = curr_rhythm / 2.0
                new_song.append(curr_chord)
                new_rhythms.append(split_duration)
                
                # 2. Create the "Passing Chord" (Soprano moves, others hold)
                passing_chord = (passing_tone, curr_chord[1], curr_chord[2], curr_chord[3])
                
                # 3. Inject the passing chord into the track
                new_song.append(passing_chord)
                new_rhythms.append(split_duration)
                continue # Skip the default append below
                
        # DEFAULT: If no passing tone was possible, just keep the original chord and rhythm
        new_song.append(curr_chord)
        new_rhythms.append(curr_rhythm)
        
    new_song.append(song[-1])
    new_rhythms.append(rhythms[-1])
    
    return new_song, new_rhythms

def export_to_midi_with_rhythm(song, rhythms, filename='bach_ai_final.mid'):
    score = stream.Score()
    part = stream.Part()

    # zip() pairs Chord #1 with Rhythm #1, Chord #2 with Rhythm #2, etc.
    for chord_tuple, duration in zip(song, rhythms):
        c = chord.Chord(chord_tuple)
                
        # Set the dynamic duration instead of hardcoding 1.0
        c.quarterLength = duration 
                
        # Add it to the part
        part.append(c)

    score.append(part)
    score.write('midi', fp=filename)
    print(f"\nSuccessfully exported {filename}!")



if __name__ == '__main__':
    start_c_major = (72, 67, 60, 48)
    my_song = compose_chorale_2nd_order(start_c_major, num_chords=24, top_k=8)
    my_rhythms = generate_rhythms(num_chords=len(my_song))
    print(f"Rhythms generated: {my_rhythms}")
    print(f"Total beats: {sum(my_rhythms)} (Should be exactly divisible by 4)")
    polished_song, polished_rhythms = inject_passing_tones(my_song, my_rhythms, tonic_pc=0)
    export_to_midi_with_rhythm(polished_song, polished_rhythms, filename='bach_with_passing_tones2.mid')