from music21 import *
from music21 import corpus, stream, chord
import random

def check_parallel_motion(voice1_chord1, voice1_chord2, voice2_chord1, voice2_chord2):
    # Short-circuit if any note is a rest, parallel motion cannot occur when there is a rest
    if -1 in (voice1_chord1, voice1_chord2, voice2_chord1, voice2_chord2):
        return True

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
    # 1. Evaluate Crossing
    voices = [soprano, alto, tenor, bass]
    active_voices = [v for v in voices if v != -1]
    
    # Active voices must remain in strictly descending pitch order
    for i in range(len(active_voices) - 1):
        if active_voices[i] < active_voices[i+1]:
            return False 

    # 2. Evaluate Spacing (Max 1 octave between adjacent active upper voices)
    upper_voices = [v for v in [soprano, alto, tenor] if v != -1]
    for i in range(len(upper_voices) - 1):
        if upper_voices[i] - upper_voices[i+1] > 12:
            return False

    return True

def check_leading_tone_resolution(note1, note2, tonic_pc):
    # Skip if note1 is a rest
    if note1 == -1:
        return True

    # 1. Check if note1 is the leading tone for the current key
    leading_tone = (tonic_pc -1) % 12
    is_leading_tone = (note1 % 12)== leading_tone
    
    # 2. If it's not the leading tone, the rule doesn't apply
    if not is_leading_tone:
        return True
    
    # 3. A leading tone cannot escape resolution by resting
    if note2 == -1:
        return False
    
    # 4. If it is the leading tone, it must resolve up by exactly 1 semitone
    if note2 == note1 + 1:
        return True
    else:
        return False

def audit_human_sequence(song_sequence, is_valid_transition_fn):
    """
    Scans a user-provided sequence of chords. Returns (True, None) if clean,
    or (False, error_message) if it breaks counterpoint rules.
    """
    for i in range(len(song_sequence) - 1):
        chord_a = song_sequence[i]
        chord_b = song_sequence[i+1]
        
        # We pass the dynamic tonic_pc based on the start of the sequence
        tonic = song_sequence[0][3] % 12 if song_sequence[0][3] != -1 else 0
        
        if not is_valid_transition_fn(chord_a, chord_b, tonic_pc=tonic):
            return False, f"Rule violation between Chord {i+1} and Chord {i+2}. Check your voice leading and spacing!"
            
    return True, None


# ==========================================
# INTERNAL GENERATOR CHECKS (Partial Chords)
# ==========================================
start_chord = [72, 67, 60, 48] 
voice_ranges = [
    range(60, 85),  # 0: Soprano
    range(53, 78),  # 1: Alto
    range(48, 73),  # 2: Tenor
    range(40, 65)   # 3: Bass
]

def check_partial_rules(current_chord):
    length = len(current_chord)
    if length == 0: 
        return True 
    
    idx = length - 1
    cand_note = current_chord[idx]
    start_note = start_chord[idx]
    
    if not check_leading_tone_resolution(start_note, cand_note, tonic_pc=0):
        return False
        
    for prev_idx in range(idx):
        prev_cand_note = current_chord[prev_idx]
        prev_start_note = start_chord[prev_idx]
        if not check_parallel_motion(prev_start_note, prev_cand_note, start_note, cand_note):
            return False
            
    # Pad the partial chord with rests so we can use the main crossing/spacing function safely
    padded_chord = list(current_chord) + [-1] * (4 - length)
    if not check_crossing_and_spacing(*padded_chord):
        return False

    return True

valid_chords = []
def generate_chords_backtracking(voice_index, current_chord):
    if not check_partial_rules(current_chord):
        return

    if len(current_chord) == 4:
        if check_crossing_and_spacing(*current_chord):
            valid_chords.append(list(current_chord))
        return

    current_range = voice_ranges[voice_index]
    for note in current_range:
        current_chord.append(note)                           
        generate_chords_backtracking(voice_index + 1, current_chord) 
        current_chord.pop()