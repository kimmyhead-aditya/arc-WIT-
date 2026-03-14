# PER (Phoneme Error Rate) module for ARC
# This is an experimental scoring module and does not modify existing ARC scoring

from difflib import SequenceMatcher


# Very basic Hindi grapheme → phoneme mapping
# (this will be expanded later)

PHONEME_MAP = {
    "क": "k",
    "ख": "kh",
    "ग": "g",
    "घ": "gh",
    "च": "ch",
    "छ": "chh",
    "ज": "j",
    "झ": "jh",
    "ट": "t",
    "ठ": "th",
    "ड": "d",
    "ढ": "dh",
    "त": "t",
    "थ": "th",
    "द": "d",
    "ध": "dh",
    "न": "n",
    "प": "p",
    "फ": "ph",
    "ब": "b",
    "भ": "bh",
    "म": "m",
    "य": "y",
    "र": "r",
    "ल": "l",
    "व": "v",
    "स": "s",
    "ह": "h",
}


def word_to_phonemes(word):
    phonemes = []

    for char in word:
        if char in PHONEME_MAP:
            phonemes.append(PHONEME_MAP[char])
        else:
            phonemes.append(char)

    return phonemes


def compute_per(reference, hypothesis):

    ref_ph = word_to_phonemes(reference)
    hyp_ph = word_to_phonemes(hypothesis)

    if len(ref_ph) == 0:
        return 0

    matcher = SequenceMatcher(None, ref_ph, hyp_ph)

    similarity = matcher.ratio()

    per_score = similarity * 100

    return per_score