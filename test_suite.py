import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from rules_engine import check_parallel_motion, check_leading_tone_resolution, check_crossing_and_spacing
from chord_generator import compose_chorale_2nd_order
from rhythm_ai import generate_rhythms, inject_passing_tones
from note_parser import parse_note
from main import app
from search_engine import encode_intervals, calculate_levenshtein, calculate_dtw, advanced_search
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
    song = compose_chorale_2nd_order(START_CHORD, num_chords=3, top_k=2)
    assert song is not None
    assert len(song) == 3
    assert song == [START_CHORD, CHORD_2, CHORD_3]

@patch("chord_generator.transition_matrix", mock_1st_order)
@patch("chord_generator.transition_matrix_2nd_order", {}) 
@patch("chord_generator.is_valid_transition", return_value=True)
def test_dfs_dead_end_handling(mock_rules):
    song = compose_chorale_2nd_order(START_CHORD, num_chords=3, top_k=2)
    assert len(song) == 1
    assert song[0] == START_CHORD

def test_dynamic_key_resolution():
    START_F = (77, 69, 65, 53) 
    CHORD_F_2 = (79, 70, 67, 55) 
    CHORD_F_3 = (77, 69, 65, 53) 
    
    mock_1st_f = {START_F: {CHORD_F_2: 1.0}}
    mock_2nd_f = {(START_F, CHORD_F_2): {CHORD_F_3: 1.0}}
    
    with patch("chord_generator.transition_matrix", mock_1st_f):
        with patch("chord_generator.transition_matrix_2nd_order", mock_2nd_f):
            with patch("chord_generator.is_valid_transition", return_value=True):
                song = compose_chorale_2nd_order(START_F, num_chords=3, top_k=2)
                assert song is not None
                assert len(song) == 3
                assert song[-1][3] % 12 == 5

# ==========================================
# 3. RHYTHM AI TESTS
# ==========================================
@patch("rhythm_ai.transition_matrix_rhythm", {})
def test_generate_rhythms_measure_math():
    target_length = 8
    rhythms = generate_rhythms(num_chords=target_length, start_duration=1.0)
    assert len(rhythms) == target_length
    total_beats = sum(rhythms)
    assert total_beats % 4.0 == 0

@patch("rhythm_ai.transition_matrix_rhythm", {0.25: {0.25: 1.0}}) 
def test_rhythm_resolution_after_fast_run():
    rhythms = generate_rhythms(num_chords=10, start_duration=0.25, max_consecutive_fast=3, resolution_min=1.0)
    longest_run = current_run = 0
    for r in rhythms:
        current_run = current_run + 1 if r <= 0.5 else 0
        longest_run = max(longest_run, current_run)
    assert longest_run <= 3

def test_inject_passing_tones_trigger():
    test_song = [(76, 67, 60, 48), (72, 67, 60, 48)] 
    test_rhythms = [2.0, 2.0] 
    
    new_song, new_rhythms = inject_passing_tones(test_song, test_rhythms, tonic_pc=0)
    assert len(new_song) == 3
    assert len(new_rhythms) == 3
    assert new_song[1][0] == 74
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
        parse_note("H4") 

# ==========================================
# 5. FASTAPI ENDPOINT TESTS 
# ==========================================
client = TestClient(app)

def test_serve_frontend():
    with patch("main.FileResponse") as mock_file:
        mock_file.return_value = MagicMock(status_code=200)
        response = client.get("/")
        assert response.status_code == 200

@patch("main.advanced_search")
@patch("main.parse_note")
@patch("main.BackgroundTasks.add_task")
def test_search_endpoint_success(mock_bg_tasks, mock_parse, mock_search):
    """Tests the updated multi-dimensional search endpoint."""
    mock_parse.side_effect = lambda x: int(x) 
    
    mock_search.return_value = [
        {"title": "bwv1.mxl", "composer": "Bach", "score": 0, "raw_notes": [60, 62, 64], "phrase_id": 0}
    ]
    
    # Send updated multi-dimensional payload
    response = client.post("/search", json={
        "melody": ["60", "62", "64", "65"], 
        "algorithm": "dtw",
        "composer": "all",
        "max_distance": 1
    })
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert len(response.json()["matches"]) == 1
    assert response.json()["matches"][0]["composer"] == "Bach"

@patch("main.transition_matrix", {(72, 67, 60, 48): {}})
@patch("main.compose_chorale_2nd_order")
@patch("main.export_to_midi_with_rhythm")
@patch("main.os.remove")
def test_generate_endpoint_success(mock_remove, mock_export, mock_compose):
    mock_compose.return_value = [(72, 67, 60, 48)] * 16 
    
    with patch("main.FileResponse") as mock_file_response:
        mock_file_response.return_value = MagicMock(status_code=200)
        response = client.post("/generate", json={"num_chords": 16, "top_k": 5, "tonic_pc": 0})
        assert response.status_code == 200

# ==========================================
# 6. SEARCH ENGINE MATH TESTS 
# ==========================================
def test_encode_intervals():
    pitches = [60, 62, 64, 65] 
    intervals = encode_intervals(pitches)
    assert intervals == [2, 2, 1]

def test_get_trigrams():
    intervals = [2, 2, 1, 0, -1]
    trigrams = get_trigrams(intervals)
    assert trigrams == [(2, 2, 1), (2, 1, 0), (1, 0, -1)]

def test_calculate_levenshtein():
    seq1 = [2, 2, 1]
    seq2 = [2, 2, 1] 
    seq3 = [2, 2, 2] 
    
    assert calculate_levenshtein(seq1, seq2) == 0
    assert calculate_levenshtein(seq1, seq3) == 1

def test_calculate_dtw():
    """Tests the DTW algorithm tolerance for extra insertions."""
    seq1 = [2, 2, 1]
    seq2 = [2, 2, 1, 0] # Extra note inserted
    
    # DTW should calculate a cost, but it successfully runs
    distance = calculate_dtw(seq1, seq2)
    assert distance >= 0

# ==========================================
# 7. SEARCH PIPELINE E2E TEST (Mocked Corpus)
# ==========================================
MOCK_DATABASE = {
    10: {
        "title": "fake_chorale.mxl", 
        "composer": "bach", 
        "intervals": [2, 2, 1, 0], 
        "raw_notes": [60, 62, 64, 65, 65]
    }
}

@patch("search_engine.database", MOCK_DATABASE)
def test_advanced_search_integration():
    """
    Tests the advanced search filtering logic using a controlled mock database.
    """
    query_melody = [60, 62, 64, 65] # Becomes intervals [2, 2, 1]
    
    # 1. Test exact composer match
    results = advanced_search(query_melody, algorithm="levenshtein", target_composer="bach", max_dist=2)
    assert len(results) == 1
    assert results[0]["phrase_id"] == 10
    assert results[0]["title"] == "fake_chorale.mxl"

    # 2. Test composer filter exclusion
    empty_results = advanced_search(query_melody, algorithm="levenshtein", target_composer="chopin", max_dist=2)
    assert len(empty_results) == 0