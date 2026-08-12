import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from rules_engine import check_parallel_motion, check_leading_tone_resolution, check_crossing_and_spacing
from chord_generator import compose_chorale_2nd_order
from rhythm_ai import generate_rhythms, inject_passing_tones
from note_parser import parse_note
from main import app
from search_engine import encode_intervals, levenshtein_distance, fuzzy_search_melody, search_bach_corpus
from build_index import get_trigrams

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
# 2. CHORD GENERATOR TESTS 
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

@patch("rhythm_ai.transition_matrix_rhythm", {0.25: {0.25: 1.0}})  # a matrix that ALWAYS wants 0.25
def test_rhythm_resolution_after_fast_run():
    """Even a pathological model that only ever wants fast notes should
    still be forced to resolve every few notes."""
    rhythms = generate_rhythms(num_chords=10, start_duration=0.25, max_consecutive_fast=3, resolution_min=1.0)
    longest_run = current_run = 0
    for r in rhythms:
        current_run = current_run + 1 if r <= 0.5 else 0
        longest_run = max(longest_run, current_run)
    assert longest_run <= 3

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


# ==========================================
# 4. NOTE_PARSER TESTS 
# ==========================================

def test_parse_raw_midi_int():
    assert parse_note("60") == 60

def test_parse_natural_note():
    assert parse_note("C4") == 60

def test_parse_sharp_note():
    assert parse_note("F#4") == 66

def test_parse_flat_note():
    assert parse_note("Bb3") == 58

def test_parse_invalid_note_raises():
    with pytest.raises(ValueError):
        parse_note("H4")  # not a real note letter


# Initialize the FastAPI TestClient
client = TestClient(app)

# ==========================================
# 5. FASTAPI ENDPOINT TESTS 
# ==========================================

def test_serve_frontend():
    """Tests if the root endpoint successfully serves the index.html file."""
    # We mock FileResponse so it doesn't fail if index.html is missing locally during a test
    with patch("main.FileResponse") as mock_file:
        mock_file.return_value = MagicMock(status_code=200)
        response = client.get("/")
        assert response.status_code == 200

@patch("main.search_bach_corpus")
@patch("main.parse_note")
def test_search_endpoint_success(mock_parse, mock_search):
    """Tests a valid payload to the /search endpoint."""
    # Mock the note parser to just return standard MIDI integers
    mock_parse.side_effect = lambda x: int(x) 
    
    # Mock the backend search engine's response
    mock_search.return_value = {
        "matches_found": 1,
        "candidates_filtered": 1,
        "results": [{"title": "bwv1.mxl", "phrase_id": 0, "edit_distances": [0]}]
    }
    
    # Send a POST request with string melodies as expected by SearchRequest
    response = client.post("/search", json={"melody": ["60", "62", "64", "65"], "max_distance": 1})
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["search_data"]["matches_found"] == 1

def test_search_endpoint_too_short():
    """Tests if the API correctly rejects a melody that is too short for a trigram."""
    # Sending only 3 notes instead of the required 4
    response = client.post("/search", json={"melody": ["60", "62", "64"], "max_distance": 1})
    
    assert response.status_code == 400
    assert "at least 4 notes" in response.json()["detail"]

@patch("main.transition_matrix", {(72, 67, 60, 48): {}})
@patch("main.compose_chorale_2nd_order")
@patch("main.export_to_midi_with_rhythm")
@patch("main.os.remove")
def test_generate_endpoint_success(mock_remove, mock_export, mock_compose):
    """Tests the /generate endpoint with a mocked chorale engine to bypass CPU-heavy backtracking."""
    # Mock the generator to instantly return a perfectly sized fake song
    mock_compose.return_value = [(72, 67, 60, 48)] * 16 
    
    # We patch FileResponse to prevent it from actually trying to read the fake MIDI on disk
    with patch("main.FileResponse") as mock_file_response:
        mock_file_response.return_value = MagicMock(status_code=200)
        
        response = client.post("/generate", json={"num_chords": 16, "top_k": 5, "tonic_pc": 0})
        assert response.status_code == 200
# ==========================================
# 6. SEARCH ENGINE MATH TESTS 
# ==========================================

def test_encode_intervals():
    """Ensures pitch arrays are correctly converted into interval differences."""
    pitches = [60, 62, 64, 65] # C, D, E, F
    intervals = encode_intervals(pitches)
    assert intervals == [2, 2, 1]

def test_get_trigrams():
    """Ensures intervals are correctly sliced into overlapping chunks of 3."""
    intervals = [2, 2, 1, 0, -1]
    trigrams = get_trigrams(intervals)
    assert trigrams == [(2, 2, 1), (2, 1, 0), (1, 0, -1)]

def test_levenshtein_distance():
    """Tests the dynamic programming edit distance calculator."""
    seq1 = [2, 2, 1]
    seq2 = [2, 2, 1] # Exact match
    seq3 = [2, 2, 2] # 1 Substitution
    
    assert levenshtein_distance(seq1, seq2) == 0
    assert levenshtein_distance(seq1, seq3) == 1

def test_fuzzy_search_melody():
    """Ensures the sliding window correctly finds target distances."""
    query = [2, 2, 1]
    # The query exists starting at index 1 of the target
    target = [0, 2, 2, 1, -2] 
    
    matches = fuzzy_search_melody(query, target, max_distance=0)
    assert matches == [(1, 0)] # Found at index 1 with 0 distance

# ==========================================
# 7. SEARCH PIPELINE E2E TEST (Mocked Corpus)
# ==========================================

# A tiny fake inverted index and phrase database to test routing
MOCK_PHRASE_DB = {
    10: {"title": "fake_chorale.mxl", "intervals": [2, 2, 1, 0]}
}
MOCK_INVERTED_INDEX = {
    (2, 2, 1): {10}
}

@patch("search_engine.PHRASE_DB", MOCK_PHRASE_DB)
@patch("search_engine.INVERTED_INDEX", MOCK_INVERTED_INDEX)
def test_search_bach_corpus_integration():
    """
    Tests the filter-then-verify logic of the main search function 
    using a controlled, mocked database.
    """
    query_melody = [60, 62, 64, 65] # Becomes intervals [2, 2, 1]
    
    results = search_bach_corpus(query_melody, max_distance=1)
    
    assert results["matches_found"] == 1
    assert results["candidates_filtered"] == 1
    assert results["results"][0]["phrase_id"] == 10
    assert results["results"][0]["title"] == "fake_chorale.mxl"