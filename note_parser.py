import re

NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARP_CHARS = {"#", "♯", "s"}
FLAT_CHARS = {"b", "♭", "f"}

def parse_note(token: str) -> int:
    token = str(token).strip()
    if not token:
        raise ValueError("Empty note token.")

    # Gracefully handle rests
    if token.lower() == "rest":
        return -1

    if re.fullmatch(r"-?\d+", token):
        return int(token)

    match = re.search(r"([A-Ga-g])\s*([#♯sSbB♭fF]?)\s*(-?\d+)?", token)
    
    if not match:
        raise ValueError(
            f"Couldn't parse note '{token}'. Use a MIDI number (60), "
            f"scientific pitch notation (C4, Bb3), or type 'rest'."
        )

    letter, accidental, octave_str = match.groups()
    pitch_class = NOTE_TO_PC[letter.upper()]
    
    if accidental.lower() in SHARP_CHARS:
        pitch_class += 1
    elif accidental.lower() in FLAT_CHARS:
        pitch_class -= 1

    octave = int(octave_str) if octave_str else 4
    return (octave + 1) * 12 + pitch_class