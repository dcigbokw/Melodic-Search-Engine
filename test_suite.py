import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, mock_open
import math
from music21 import pitch
from rules_engine import check_parallel_motion, check_leading_tone_resolution, check_crossing_and_spacing
from chord_generator import compose_chorale_2nd_order
from rhythm_ai import generate_rhythms, inject_passing_tones
from note_parser import parse_note
from main import app, interpolate_matrices
from search_engine import (
    calculate_levenshtein, 
    calculate_dtw, 
    advanced_search,
    calculate_horizontal_intervals,
    calculate_vertical_intervals
)
from music_scorer import score_composition
from creativity_metrics import (
    calculate_phrase_memorization,
    calculate_nearest_neighbor_novelty,
    calculate_structural_diversity,
    evaluate_blend_fidelity,
    calculate_style_likelihood,
    calculate_pairwise_diversity
)
from train import process_bundles
from build_index import build_multidimensional_index
from benchmark import calculate_duration_entropy, check_meter_alignment, calculate_sequence_nll

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
# 2. CHORD GENERATOR & MATH TESTS 
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

def test_interpolate_matrices_normalization():
    mock_corpora = {
        "bach": {
            "train": {
                "first_order": {START_CHORD: {CHORD_2: 0.5, CHORD_3: 0.5}}
            }
        },
        "mozart": {
            "train": {
                "first_order": {START_CHORD: {CHORD_2: 0.2}}
            }
        }
    }
    
    weights = {"bach": 50, "mozart": 50}
    blended = interpolate_matrices(mock_corpora, weights, "first_order", "train")
    
    assert START_CHORD in blended
    row = blended[START_CHORD]
    assert round(sum(row.values()), 5) == 1.0
    assert round(row[CHORD_2], 4) == 0.5833
    assert round(row[CHORD_3], 4) == 0.4167

# ==========================================
# 3. RHYTHM AI TESTS
# ==========================================
@patch("rhythm_ai.transition_matrix_rhythm", {})
def test_generate_rhythms_measure_math():
    target_length = 8
    rhythms = generate_rhythms(num_chords=target_length)
    assert len(rhythms) == target_length

@patch("rhythm_ai.transition_matrix_rhythm", {0.25: {0.25: 1.0}}) 
def test_rhythm_resolution_after_fast_run():
    rhythms = generate_rhythms(num_chords=10)
    assert len(rhythms) == 10

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

def test_parse_invalid_note_raises():
    with pytest.raises(ValueError):
        parse_note("H4") 

# ==========================================
# 5. MUSIC SCORER TESTS 
# ==========================================
def test_score_composition_empty_or_short():
    assert score_composition([]) == 0
    assert score_composition([(60, 64, 67, 72)]) == 0

def test_score_composition_style_factor_reward():
    chord_a = (60, 64, 67, 72)
    chord_b = (62, 65, 69, 74)
    dynamic_matrix = {chord_a: {chord_b: 0.1}}
    
    score = score_composition([chord_a, chord_b], dynamic_matrix)
    assert score == 115.0

def test_score_composition_style_factor_penalty():
    chord_a = (60, 64, 67, 72)
    chord_b = (62, 65, 69, 74)
    dynamic_matrix = {chord_a: {}}
    
    score = score_composition([chord_a, chord_b], dynamic_matrix)
    assert score == 90.0

def test_score_composition_dissonance_penalty():
    chord_a = (60, 64, 67, 72)
    chord_b = (60, 61, 67, 72) 
    
    score = score_composition([chord_a, chord_b], dynamic_matrix=None)
    assert score == 95.0

def test_score_composition_polyphonic_rests():
    chord_a = (60, 64, 67, 72)
    chord_b = (-1, -1, 67, 72) 
    
    score = score_composition([chord_a, chord_b], dynamic_matrix=None)
    assert score == 112.0

# ==========================================
# 6. CREATIVITY METRICS TESTS 
# ==========================================
def test_calculate_phrase_memorization():
    gen_seq = [(1,), (2,), (3,), (4,)]
    train_corpus = [
        [(9,), (1,), (2,), (3,), (8,)],
        [(7,), (7,)]
    ]
    
    max_overlap, overlap_pct = calculate_phrase_memorization(gen_seq, train_corpus)
    assert max_overlap == 3
    assert overlap_pct == 75.0

def test_calculate_nearest_neighbor_novelty():
    gen_seq = [(1,), (2,), (3,)]
    train_corpus = [
        [(1,), (2,), (3,)],
        [(9,), (8,), (7,)]
    ]
    
    with patch("creativity_metrics.calculate_levenshtein", side_effect=lambda a, b: 0 if a==b else 3):
        min_dist = calculate_nearest_neighbor_novelty(gen_seq, train_corpus)
        assert min_dist == 0.0

def test_calculate_structural_diversity():
    gen_seq = [(1,), (2,), (1,), (2,)]
    ratio, entropy = calculate_structural_diversity(gen_seq)
    
    assert ratio == 0.5
    expected_entropy = -((2/3)*math.log2(2/3) + (1/3)*math.log2(1/3))
    assert math.isclose(entropy, expected_entropy, rel_tol=1e-5)

@patch("creativity_metrics.all_corpora")
def test_evaluate_blend_fidelity(mock_corpora):
    chord_a, chord_b, chord_c = (1,), (2,), (3,)
    seq = [chord_a, chord_b, chord_c]
    
    mock_corpora.get.side_effect = lambda comp, default: {
        "evaluator": {
            "second_order": {
                (chord_a, chord_b): {chord_c: 0.8} if comp == "bach" else {chord_c: 0.2}
            }
        }
    }
    
    nll_a, nll_b, nll_mix = evaluate_blend_fidelity(seq, "bach", 50, "chopin", 50, markov_order=2)
    assert math.isclose(nll_mix, -math.log(0.5), rel_tol=1e-5)

@patch.dict("creativity_metrics.all_corpora", {
    "bach": {
        "train": {
            "first_order": {
                60: {62: 0.9, 64: 0.1}
            }
        }
    }
}, clear=True)
def test_calculate_style_likelihood():
    avg_nll, scored, unseen, probs = calculate_style_likelihood([60, 62], composer="bach", markov_order=1)
    assert scored == 1
    assert unseen == 0
    assert math.isclose(avg_nll, -math.log(0.9))
    assert len(probs) == 1
    
    avg_nll_unseen, _, unseen_count, _ = calculate_style_likelihood([65, 67], composer="bach", markov_order=1)
    assert unseen_count == 1
    assert math.isclose(avg_nll_unseen, -math.log(1e-9))

@patch("creativity_metrics.calculate_levenshtein")
def test_calculate_pairwise_diversity(mock_levenshtein):
    mock_levenshtein.side_effect = [2, 4, 2]
    seq1 = [1, 2, 3, 4]  
    seq2 = [1, 2, 5, 5]  
    seq3 = [9, 9, 9, 9]  
    
    collection = [seq1, seq2, seq3]
    diversity = calculate_pairwise_diversity(collection)
    assert math.isclose(diversity, (0.5 + 1.0 + 0.5) / 3)
    assert calculate_pairwise_diversity([seq1]) == 0.0

# ==========================================
# 7. TRAINING & INDEXING TESTS 
# ==========================================

@patch("train.corpus.parse")
def test_process_bundles_matrix_generation(mock_parse):
    mock_score = MagicMock()
    mock_key = MagicMock()
    mock_key.mode = 'major'
    mock_key.tonic = pitch.Pitch('C')
    mock_score.analyze.return_value = mock_key
    
    mock_score.transpose.return_value = mock_score
    
    mock_chord_1 = MagicMock()
    mock_chord_1.pitches = [MagicMock(ps=60), MagicMock(ps=64), MagicMock(ps=67), MagicMock(ps=72)]
    mock_chord_2 = MagicMock()
    mock_chord_2.pitches = [MagicMock(ps=62), MagicMock(ps=65), MagicMock(ps=69), MagicMock(ps=74)]
    
    mock_chordify = MagicMock()
    mock_chordify.flatten().getElementsByClass.return_value = [mock_chord_1, mock_chord_2]
    mock_score.chordify.return_value = mock_chordify
    mock_parse.return_value = mock_score
    
    counts_1, counts_2, counts_3, raw_seqs, raw_1, raw_2, raw_3 = process_bundles(["fake_path.mxl"])
    
    expected_chord_1 = (72, 67, 64, 60)
    expected_chord_2 = (74, 69, 65, 62)
    
    assert expected_chord_1 in counts_1
    assert counts_1[expected_chord_1][expected_chord_2] == 1.0
    assert len(raw_seqs) == 1
    assert raw_seqs[0] == [expected_chord_1, expected_chord_2]

@patch("pickle.dump")
@patch("builtins.open", new_callable=mock_open)
@patch("build_index.corpus.parse")
@patch("build_index.corpus.getComposer")
def test_build_multidimensional_index(mock_get_composer, mock_parse, mock_file, mock_pickle):
    mock_get_composer.return_value = ["fake_path.mxl"]
    
    mock_score = MagicMock()
    mock_score.metadata.title = "Test Song"
    
    mock_chord_1 = MagicMock()
    mock_chord_1.pitches = [MagicMock(midi=60), MagicMock(midi=64), MagicMock(midi=67)]
    mock_chord_2 = MagicMock()
    mock_chord_2.pitches = [MagicMock(midi=62), MagicMock(midi=65), MagicMock(midi=69)]
    mock_chord_3 = MagicMock()
    mock_chord_3.pitches = [MagicMock(midi=64), MagicMock(midi=67), MagicMock(midi=71)]
    mock_chord_4 = MagicMock()
    mock_chord_4.pitches = [MagicMock(midi=65), MagicMock(midi=69), MagicMock(midi=72)]
    
    mock_chordify = MagicMock()
    mock_chordify.flatten().getElementsByClass.return_value = [
        mock_chord_1, mock_chord_2, mock_chord_3, mock_chord_4
    ]
    mock_score.chordify.return_value = mock_chordify
    mock_parse.return_value = mock_score
    
    build_multidimensional_index()
    assert mock_pickle.called
    dumped_data = mock_pickle.call_args[0][0]
    
    assert "database" in dumped_data
    assert "inverted_index" in dumped_data
    assert 0 in dumped_data["database"]
    assert len(dumped_data["inverted_index"]) > 0

# ==========================================
# 8. BENCHMARK METRIC HELPERS 
# ==========================================
def test_calculate_duration_entropy():
    assert calculate_duration_entropy([1.0, 1.0, 1.0, 1.0]) == 0.0
    entropy = calculate_duration_entropy([0.5, 1.0, 0.25, 2.0])
    assert math.isclose(entropy, 2.0, rel_tol=1e-5)

def test_check_meter_alignment():
    assert check_meter_alignment([1.0, 1.0, 1.0, 1.0], beats_per_measure=4.0) == True
    assert check_meter_alignment([1.0, 1.0, 1.0, 1.0, 1.0], beats_per_measure=4.0) == False
    assert check_meter_alignment([1.0, 1.0, 1.0], beats_per_measure=3.0) == True

@patch.dict("benchmark.all_corpora", {
    "bach": {
        "train": {
            "second_order": {
                (1, 2): {3: 0.8, 4: 0.2}
            }
        }
    }
}, clear=True)
def test_calculate_sequence_nll():
    nll_valid = calculate_sequence_nll([1, 2, 3], "bach", markov_order=2)
    assert math.isclose(nll_valid, -math.log(0.8))
    
    nll_unseen = calculate_sequence_nll([1, 2, 5], "bach", markov_order=2)
    assert math.isclose(nll_unseen, -math.log(1e-9))
    
    nll_short = calculate_sequence_nll([1, 2], "bach", markov_order=2)
    assert nll_short == float('inf')

# ==========================================
# 9. FASTAPI ENDPOINT TESTS 
# ==========================================
client = TestClient(app)

def test_serve_frontend():
    with patch("main.FileResponse") as mock_file:
        mock_file.return_value = MagicMock(status_code=200)
        response = client.get("/")
        assert response.status_code == 200

@patch("main.is_valid_transition", return_value=True)
@patch("main.interpolate_matrices")
@patch("main.export_to_midi_with_rhythm")
@patch("builtins.open", new_callable=mock_open, read_data=b"mocked_midi_data")
def test_generate_endpoint_success(mock_file_open, mock_export, mock_interp, mock_valid):
    CHORD_I = (72, 67, 60, 48)
    CHORD_V = (74, 71, 62, 55)
    
    mock_interp.return_value = {
        CHORD_I: {CHORD_I: 1.0, CHORD_V: 1.0},
        CHORD_V: {CHORD_I: 1.0, CHORD_V: 1.0}
    }
    
    response = client.post("/generate", json={
        "num_chords": 8, 
        "composer": "bach",
        "tonic_pc": 0,
        "time_signature": "4/4"
    })
    
    assert response.status_code == 200

@patch("main.interpolate_matrices")
def test_markov_tree_endpoint(mock_interp):
    CHORD_I = (72, 67, 60, 48)
    CHORD_V = (74, 71, 62, 55)
    
    mock_interp.return_value = {
        CHORD_I: {CHORD_I: 0.3, CHORD_V: 0.7}
    }
    
    response = client.post("/markov-tree", json={
        "current_chords": [["C5", "G4", "C4", "C3"]],
        "composer_weights": {"bach": 100},
        "markov_order": 1
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["edges"]) >= 2

@patch("main.is_valid_transition", return_value=False)
@patch("main.interpolate_matrices")
@patch("main.audit_human_sequence", return_value=(True, ""))
def test_extend_endpoint_explicit_failure_fallback(mock_audit, mock_interp, mock_valid):
    CHORD_I = (72, 67, 60, 48)
    mock_interp.return_value = {CHORD_I: {CHORD_I: 1.0}}
    
    response = client.post("/extend", json={
        "current_chords": [["C5", "G4", "C4", "C3"]],
        "num_to_add": 2,
        "composer_weights": {"bach": 100},
        "strictness": 1.0
    })
    
    assert response.status_code == 500
    assert "exhausted all search branches" in response.json()["detail"]

@patch("main.interpolate_matrices")
def test_harmonize_endpoint(mock_interp):
    CHORD_I = (72, 67, 60, 48)
    mock_interp.return_value = {CHORD_I: {CHORD_I: 1.0}}
    
    response = client.post("/harmonize", json={
        "melody": ["C5"],
        "tonic_pc": 0,
        "composer_weights": {"bach": 100}
    })
    
    assert response.status_code == 200
    assert "chords" in response.json()

# ==========================================
# 10. SEARCH PIPELINE E2E TEST (Mocked Corpus)
# ==========================================
MOCK_DATABASE = {
    10: {
        "title": "fake_chorale.mxl", 
        "composer": "bach", 
        "horizontal_intervals": [2, 2, 1, 0], 
        "vertical_intervals": [0, 4, 7],
        "exact_vertical_notes": [60, 64, 67],
        "raw_notes": [60, 62, 64, 65, 65]
    }
}

@patch("search_engine.database", MOCK_DATABASE)
def test_advanced_search_integration():
    query_melody = [60, 62, 64, 65] 
    
    results = advanced_search(query_melody, algorithm="levenshtein", target_composer="bach", max_dist=2)
    assert len(results) == 1
    assert results[0]["title"] == "fake_chorale.mxl"
    assert results[0]["score"] <= 2 

    empty_results = advanced_search(query_melody, algorithm="levenshtein", target_composer="chopin", max_dist=2)
    assert len(empty_results) == 0