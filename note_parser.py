import re

NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARP_CHARS = {"#", "♯", "s"}
FLAT_CHARS = {"b", "♭", "f"}

def parse_note(token: str) -> int:
    """
    Accepts either a raw MIDI integer ("60") or scientific pitch notation
    ("C4", "F#4", "Bb3") and returns a MIDI note number.
    Middle C (C4) = 60, matching the convention already used elsewhere
    in this project (music21's default).
    """
    token = token.strip()
    if not token:
        raise ValueError("Empty note token.")

    if re.fullmatch(r"-?\d+", token):
        return int(token)

    match = re.fullmatch(r"([A-Ga-g])([#♯sSbB♭fF]?)(-?\d+)", token)
    if not match:
        raise ValueError(
            f"Couldn't parse note '{token}'. Use a MIDI number (60) or "
            f"scientific pitch notation like C4, F#4, or Bb3."
        )

    letter, accidental, octave = match.groups()
    pitch_class = NOTE_TO_PC[letter.upper()]
    if accidental.lower() in SHARP_CHARS:
        pitch_class += 1
    elif accidental.lower() in FLAT_CHARS:
        pitch_class -= 1

    return (int(octave) + 1) * 12 + pitch_class