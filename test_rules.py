import pytest
from rules_engine import check_parallel_motion, check_leading_tone_resolution,check_crossing_and_spacing

# ==========================================
# Parallel Fifths 
# ==========================================
def test_parallel_fifths_rejected():
    """
    Tests if the engine catches two voices moving in perfect fifths.
    Chord A: C (60) and G (67) -> Distance of 7 (Perfect Fifth)
    Chord B: D (62) and A (69) -> Distance of 7 (Perfect Fifth)
    """
    voice1_chord_a = 60
    voice1_chord_b = 62
    
    voice2_chord_a = 67
    voice2_chord_b = 69
    
    # The rule should return False because this is illegal!
    assert check_parallel_motion(voice1_chord_a, voice1_chord_b, voice2_chord_a, voice2_chord_b) == False

# ==========================================
# Valid Motion 
# ==========================================
def test_valid_motion_accepted():
    """
    Tests if the engine allows valid contrary/oblique motion.
    Chord A: C (60) and E (64) -> Major Third
    Chord B: C (60) and F (65) -> Perfect Fourth
    """
    voice1_chord_a = 60
    voice1_chord_b = 60  # Holds the same note
    
    voice2_chord_a = 64
    voice2_chord_b = 65  # Moves up a half step
    
    # The rule should return True because no parallel fifths/octaves occurred
    assert check_parallel_motion(voice1_chord_a, voice1_chord_b, voice2_chord_a, voice2_chord_b) == True

# ==========================================
# LEADING TONE TESTS
# ==========================================
def test_leading_tone_fails_to_resolve():
    """
    In C Major (tonic=0), B (71) is the leading tone. 
    If it moves down to A (69) instead of up to C (72), it should fail.
    """
    assert check_leading_tone_resolution(71, 69, tonic_pc=0) == False

def test_leading_tone_resolves_correctly():
    """
    The leading tone B (71) moves cleanly up to the tonic C (72).
    """
    assert check_leading_tone_resolution(71, 72, tonic_pc=0) == True

def test_non_leading_tone_ignored():
    """
    If the note isn't the leading tone (e.g., G moving to A), 
    the rule should ignore it and return True.
    """
    assert check_leading_tone_resolution(67, 69, tonic_pc=0) == True

# ==========================================
# SPACING AND CROSSING TESTS
# ==========================================
def test_valid_chord_spacing_accepted():
    """
    A perfectly voiced C Major chord.
    Soprano: C5 (72), Alto: G4 (67), Tenor: E4 (64), Bass: C4 (60)
    """
    assert check_crossing_and_spacing(72, 67, 64, 60) == True

def test_voice_crossing_rejected():
    """
    The Alto note (74) is mathematically higher than the Soprano note (72).
    This violates strict counterpoint voice order.
    """
    assert check_crossing_and_spacing(72, 74, 64, 60) == False

def test_voice_spacing_rejected():
    """
    The Soprano (84) and Alto (67) are 17 semitones apart.
    Standard SATB rules forbid adjacent upper voices from exceeding 12 semitones.
    """
    assert check_crossing_and_spacing(84, 67, 64, 60) == False