import pytest
from unittest.mock import patch
from rules_engine import check_parallel_motion, check_leading_tone_resolution, check_crossing_and_spacing
from chord_generator import compose_chorale_2nd_order
from rhythm_ai import generate_rhythms, inject_passing_tones

# ==========================================
# 1. RULES ENGINE TESTS
# ==========================================
def test_parallel_fifths_rejected():
    voice1_chord_a, voice1_chord_b = 60, 62
    voice2_chord_a, voice2_chord_b = 67, 69
    assert check_parallel_motion(voice1_chord_a, voice1_chord_b, voice2_chord_a, voice2_chord_b) == False

def test_valid_motion_accepted():
    voice1_chord_a, voice1_chord_b = 60, 60 
    voice2_chord_a, voice2_chord_b = 64, 65 
    assert check_parallel_motion(voice1_chord_a, voice1_chord_b, voice2_chord_a, voice2_chord_b) == True

def test_leading_tone_fails_to_resolve():
    assert check_leading_tone_resolution(71, 69, tonic_pc=0) == False

def test_leading_tone_resolves_correctly():
    assert check_leading_tone_resolution(71, 72, tonic_pc=0) == True

def test_non_leading_tone_ignored():
    assert check_leading_tone_resolution(67, 69, tonic_pc=0) == True

def test_valid_chord_spacing_accepted():
    assert check_crossing_and_spacing(72, 67, 64, 60) == True

def test_voice_crossing_rejected():
    assert check_crossing_and_spacing(72, 74, 64, 60) == False

def test_voice_spacing_rejected():
    assert check_crossing_and_spacing(84, 67, 64, 60) == False

# ==========================================
# 2. CHORD GENERATOR TESTS (Mocked)
# ==========================================
START_CHORD = (72, 67, 60, 48)
CHORD_2 = (74, 69, 62, 50)
CHORD_3 = (76, 71, 64, 48)

mock_1st_order = {START_CHORD: {CHORD_2: 0.9}}
mock_2nd_order = {(START_CHORD, CHORD_2): {CHORD_3: 0.8}}

@patch("chord_generator.transition_matrix", mock_1st_order)
@patch("chord_generator.transition_matrix_2nd_order", mock_2nd_order)
@patch("chord_generator.is_valid_transition", return_value=True)
def test_dfs_backtracking_success(mock_rules):
    """Tests if the recursive engine successfully follows a valid Markov path to completion."""
    song = compose_chorale_2nd_order(START_CHORD, num_chords=3, top_k=2)
    assert song is not None
    assert len(song) == 3
    assert song == [START_CHORD, CHORD_2, CHORD_3]

@patch("chord_generator.transition_matrix", mock_1st_order)
@patch("chord_generator.transition_matrix_2nd_order", {}) 
@patch("chord_generator.is_valid_transition", return_value=True)
def test_dfs_dead_end_handling(mock_rules):
    """Tests if the engine gracefully aborts when the Markov chain has no data for the current state."""
    song = compose_chorale_2nd_order(START_CHORD, num_chords=3, top_k=2)
    assert len(song) == 1
    assert song[0] == START_CHORD

def test_dynamic_key_resolution():
    """
    Tests if the engine correctly auto-detects a non-C key (F Major) 
    and enforces a correct resolution to that specific key.
    """
    # Bass is 53 (F). 53 % 12 = 5.
    START_F = (77, 69, 65, 53) 
    CHORD_F_2 = (79, 70, 67, 55) 
    CHORD_F_3 = (77, 69, 65, 53) # Resolves perfectly back to F
    
    mock_1st_f = {START_F: {CHORD_F_2: 1.0}}
    mock_2nd_f = {(START_F, CHORD_F_2): {CHORD_F_3: 1.0}}
    
    with patch("chord_generator.transition_matrix", mock_1st_f):
        with patch("chord_generator.transition_matrix_2nd_order", mock_2nd_f):
            with patch("chord_generator.is_valid_transition", return_value=True):
                
                # We do NOT pass tonic_pc. The engine must figure it out itself!
                song = compose_chorale_2nd_order(START_F, num_chords=3, top_k=2)
                
                assert song is not None, "Engine failed to generate the F Major sequence."
                assert len(song) == 3, "Engine did not complete the 3-chord sequence."
                assert song[-1][3] % 12 == 5, "Engine failed to force an F resolution!"
# ==========================================
# 3. RHYTHM AI TESTS
# ==========================================
@patch("rhythm_ai.transition_matrix_rhythm", {})
def test_generate_rhythms_measure_math():
    """Ensures rhythm generator outputs the correct amount of rhythms and respects the 4/4 measure."""
    target_length = 8
    rhythms = generate_rhythms(num_chords=target_length, start_duration=1.0)
    
    assert len(rhythms) == target_length, f"Expected {target_length} durations, got {len(rhythms)}"
    
    total_beats = sum(rhythms)
    assert total_beats % 4.0 == 0, f"Measure overflow! Total beats {total_beats} is not divisible by 4."

def test_inject_passing_tones_trigger():
    """Tests if the engine correctly splits a note when the Soprano leaps by a 3rd."""
    # Soprano leaps down a major 3rd from E (76) to C (72)
    test_song = [(76, 67, 60, 48), (72, 67, 60, 48)] 
    test_rhythms = [2.0, 2.0] # Half notes (long enough to split)
    
    new_song, new_rhythms = inject_passing_tones(test_song, test_rhythms, tonic_pc=0)
    
    # It should have injected a passing tone, making the song 3 chords long
    assert len(new_song) == 3, "Engine failed to inject the passing chord."
    assert len(new_rhythms) == 3, "Engine failed to split the rhythms."
    
    # The injected chord should have D (74) in the Soprano (between E and C)
    assert new_song[1][0] == 74, "Injected passing tone was the wrong pitch."
    
    # The first rhythm (2.0) should be split into two 1.0 rhythms
    assert new_rhythms[0] == 1.0
    assert new_rhythms[1] == 1.0