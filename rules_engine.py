from music21 import *
from music21 import corpus, stream, chord
import random

def check_parallel_motion(voice1_chord1, voice1_chord2, voice2_chord1, voice2_chord2):
    # 1. Calculate the vertical interval between the two voices for each chord
    interval_chord1 = abs(voice1_chord1 - voice2_chord1)
    interval_chord2 = abs(voice1_chord2 - voice2_chord2)
    
    # 2. Reduce the intervals to base semitones (0-11)
    base_interval_1 = interval_chord1 % 12
    base_interval_2 = interval_chord2 % 12
    
    # 3. Check if both chords form a perfect fifth (7) or perfect octave (0)
    is_parallel_fifth = (base_interval_1 == 7) and (base_interval_2 == 7)
    is_parallel_octave = (base_interval_1 == 0) and (base_interval_2 == 0)
    
    # Make sure the voices actually moved! 
    # Repeated notes aren't considered parallel motion
    voices_moved = (voice1_chord1 != voice1_chord2)
    
    # 4. If they moved in parallel 5ths or octaves, reject it
    if (is_parallel_fifth or is_parallel_octave) and voices_moved:
        return False
        
    return True

def check_crossing_and_spacing(soprano, alto, tenor, bass):
    cond1 = soprano >= alto >= tenor >= bass
    cond2 = ((soprano - alto) <= 12) and ((alto -tenor) <= 12)
    if not cond1 or not cond2:
        return False
    return True

def check_leading_tone_resolution(note1, note2, tonic_pc):
    # 1. Check if note1 is the leading tone for the current key
    leading_tone = (tonic_pc -1) % 12
    is_leading_tone = (note1 % 12)== leading_tone
    
    # 2. If it's not the leading tone, the rule doesn't apply
    if not is_leading_tone:
        return True
        
    # 3. If it is the leading tone, it must resolve up by exactly 1 semitone
    if note2 == note1 + 1:
        return True
    else:
        return False

# The starting state for the engine
start_chord = [72, 67, 60, 48] 
voice_ranges = [
    range(60, 85),  # 0: Soprano range
    range(53, 78),  # 1: Alto range
    range(48, 73),  # 2: Tenor range
    range(40, 65)   # 3: Bass range
]

valid_chords = []

def check_partial_rules(current_chord):
    length = len(current_chord)
    if length == 0: 
        return True # An empty backpack breaks no rules!
    
    # Get the note we just added, and its corresponding starting note
    idx = length - 1
    cand_note = current_chord[idx]
    start_note = start_chord[idx]
    
    # Check Leading Tone 
    if not check_leading_tone_resolution(start_note, cand_note, tonic_pc=0):
        return False
        
    # Check Parallel Motion against all previously placed voices
    for prev_idx in range(idx):
        prev_cand_note = current_chord[prev_idx]
        prev_start_note = start_chord[prev_idx]
        
       
        if not check_parallel_motion(prev_start_note, prev_cand_note, start_note, cand_note):
            return False
            
    # 3. Partial Crossing & Spacing 
    # (Since spacing function requires all 4 notes, we do a quick check here)
    if length > 1:
        voice_above_cand = current_chord[idx - 1]
        
        # Crossing: The voice above must be higher
        if voice_above_cand < cand_note:
            return False
            
        # Spacing: Soprano/Alto (length 2) and Alto/Tenor (length 3) max 1 octave apart
        if length in [2, 3]: 
            if voice_above_cand - cand_note > 12:
                return False

    return True

def generate_chords_backtracking(voice_index, current_chord):
    # BASE CASE 1: FAILURE (Pruning)
    if not check_partial_rules(current_chord):
        return

    # BASE CASE 2: SUCCESS
    if len(current_chord) == 4:
        # We can run the full crossing and spacing function here just as a final seal of approval!
        if check_crossing_and_spacing(current_chord[0], current_chord[1], current_chord[2], current_chord[3]):
            valid_chords.append(list(current_chord))
        return

    # Get the correct range for the current voice
    current_range = voice_ranges[voice_index]

    # CHOOSE, EXPLORE, UNDO
    for note in current_range:
        current_chord.append(note)                           
        generate_chords_backtracking(voice_index + 1, current_chord) 
        current_chord.pop()                                  

# Kick it off with Voice 0 and an empty backpack
generate_chords_backtracking(0, [])
print(f"Total valid chords found: {len(valid_chords)}")
