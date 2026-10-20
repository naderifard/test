"""Persian grapheme-to-phoneme, and the phoneme-to-viseme map of section 7-G1.

Decision 7 makes the dialogue Persian only, which pushes two problems into the
host that an English pipeline never meets.

**Persian script does not write its short vowels.** The word spelled p-h-l-v-a-n
is pronounced *pahlavan*: three of its eight sounds are simply absent from the
page. A grapheme-to-phoneme pass over the script therefore has to *guess* them,
and no amount of care makes that guess reliable. What saves the situation is
that this module's consumer is a mouth, not a speech recogniser -- a missing
short vowel costs one open-jaw frame out of forty, and the syllable structure
that governs where the vowels go is regular enough to reconstruct. Section 7-G
names this exact route, "a Persian G2P distributing phonemes proportionally
inside each word", as the documented fallback, and every artifact built through
it is flagged ``approximate`` so ``gates.check_align`` says so out loud.

**The viseme set has to be Persian-shaped.** The four sounds English has not got
-- ``x``, ``gh``, ``zh`` and the glottal stop -- are all articulated behind the
lips, so they group with sounds that are already here rather than needing mouth
shapes of their own. Persian's vowel inventory, on the other hand, is smaller
than English's and cleanly separated into the six of :data:`VOWELS`, which is
why a twelve-viseme set covers the language comfortably.

Everything here is stdlib and pure, so the host plans with it, the aligner stage
runs it, and the tests check it without Blender or torch.

Every row of the letter tables below carries the Unicode *name* of its letter in
a comment. That is not decoration. This file mixes right-to-left letters with
left-to-right code on one line, and a bidirectional editor reorders how such a
line is *displayed* without changing its bytes -- so the eye cannot always tell
which letter a row is about. Two of the letters here are worse than ambiguous:
the Arabic and Persian yeh, and the Arabic kaf and keheh, are identical in most
fonts and are exactly the pair :data:`FOLD` exists to reconcile. The names
disambiguate them in any editor.
"""

from __future__ import annotations

import unicodedata

# -- normalisation -----------------------------------------------------------
# Arabic-script text arrives from a dozen keyboards. Two letters differ between
# the Arabic and Persian ranges and are the classic source of a word that looks
# identical and compares unequal, so they are folded before anything else.
FOLD = {
    "ي": "ی",   # ARABIC YEH          -> FARSI YEH
    "ى": "ی",   # ALEF MAKSURA        -> FARSI YEH
    "ك": "ک",   # ARABIC KAF          -> KEHEH
    "ۀ": "ه",   # HEH WITH YEH ABOVE  -> HEH
    "ة": "ه",   # TEH MARBUTA         -> HEH
    "ـ": "",         # TATWEEL: decoration, never a sound
    "‏": "",         # RIGHT-TO-LEFT MARK
    "‎": "",         # LEFT-TO-RIGHT MARK
}

# Zero-width non-joiner. It joins *inside* a word -- "javan-mard" is one word --
# but it also marks a morpheme seam, which is where a syllable boundary falls,
# so it survives normalisation and is read as a break rather than as a space.
ZWNJ = "‌"

# Removed outright: these carry no sound of their own.
SILENT = (
    "ْ"   # SUKUN: marks the absence of a vowel
    "ٰ"   # SUPERSCRIPT ALEF
    "ٓ"   # MADDAH ABOVE
    "ٔ"   # HAMZA ABOVE
    "ٕ"   # HAMZA BELOW
)

PUNCTUATION = (
    "،"   # ARABIC COMMA
    "؛"   # ARABIC SEMICOLON
    "؟"   # ARABIC QUESTION MARK
    "«»"
    ".,;:!?()[]{}"
    "—–-"
    "\"'"
    "…·"
)


def normalise(text):
    """Fold the script to one spelling, drop decoration, keep the letters."""
    out = unicodedata.normalize("NFC", text or "")
    for source, target in FOLD.items():
        out = out.replace(source, target)
    return "".join(ch for ch in out if ch not in SILENT)


def words(text):
    """Split into words on whitespace and punctuation. ZWNJ stays inside."""
    normalised = normalise(text)
    for mark in PUNCTUATION:
        normalised = normalised.replace(mark, " ")
    return [w for w in normalised.split() if w.strip(ZWNJ)]


# -- the phoneme inventory ---------------------------------------------------
# Six vowels. Persian has no vowel length distinction that a mouth shows, so the
# traditional long/short split is not modelled.
VOWELS = frozenset(("ɒ", "e", "i", "o", "u", "a"))

AA = "ɒ"     # IPA turned script a: the long open vowel, romanised "a-bar"
SH = "ʃ"     # IPA esh
ZH = "ʒ"     # IPA ezh
GH = "ɣ"     # IPA gamma: the voiced velar fricative
GLOTTAL = "ʔ"

# One entry per Persian consonant letter. Where a letter's sound depends on its
# position in the word it lives in POSITIONAL instead.
CONSONANTS = {
    "ب": "b",        # BEH
    "پ": "p",        # PEH
    "ت": "t",        # TEH
    "ث": "s",        # THEH   - one of Persian's three /s/ spellings
    "ج": "d" + ZH,   # JEEM
    "چ": "t" + SH,   # TCHEH
    "ح": "h",        # HAH    - merged with HEH in speech
    "خ": "x",        # KHAH
    "د": "d",        # DAL
    "ذ": "z",        # THAL   - one of Persian's four /z/ spellings
    "ر": "r",        # REH
    "ز": "z",        # ZAIN
    "ژ": ZH,         # JEH
    "س": "s",        # SEEN
    "ش": SH,         # SHEEN
    "ص": "s",        # SAD
    "ض": "z",        # DAD
    "ط": "t",        # TAH
    "ظ": "z",        # ZAH
    "ع": GLOTTAL,    # AIN    - a glottal stop in Persian, not a pharyngeal
    "غ": GH,         # GHAIN
    "ق": GH,         # QAF    - merged with GHAIN in modern Tehrani
    "ف": "f",        # FEH
    "ک": "k",        # KEHEH
    "گ": "g",        # GAF
    "ل": "l",        # LAM
    "م": "m",        # MEEM
    "ن": "n",        # NOON
    "ء": GLOTTAL,    # HAMZA
    "أ": GLOTTAL,    # ALEF WITH HAMZA ABOVE
    "إ": GLOTTAL,    # ALEF WITH HAMZA BELOW
    "ؤ": GLOTTAL,    # WAW WITH HAMZA ABOVE
    "ئ": GLOTTAL,    # YEH WITH HAMZA ABOVE
}

# Letters whose sound depends on where they sit: (initial, medial, final).
POSITIONAL = {
    # ALEF: a bare initial alef is a vowel onset; elsewhere it is the long /a/.
    "ا": ("a", AA, AA),
    # ALEF WITH MADDA is the long /a/ wherever it stands.
    "آ": (AA, AA, AA),
    # WAW: the consonant /v/ at the onset, the vowel /u/ inside and at the end.
    "و": ("v", "u", "u"),
    # FARSI YEH: the glide /j/ at the onset, the vowel /i/ otherwise.
    "ی": ("j", "i", "i"),
    # HEH: /h/ at the onset and inside, but a word-final heh is the vowel /e/ --
    # "xane", not "xaneh". This is the commonest word ending in Persian, and
    # getting it wrong closes a mouth that should be open.
    "ه": ("h", "h", "e"),
}

# The vowel diacritics, when a text bothers to write them. Rare in running prose,
# common in poetry and in anything set for a reader -- and the Shahnameh is
# poetry, so they are worth honouring rather than stripping.
HARAKAT = {
    "َ": "a",      # FATHA
    "ِ": "e",      # KASRA
    "ُ": "o",      # DAMMA
    "ً": "an",     # FATHATAN - the adverbial ending
    "ٌ": "on",     # DAMMATAN
    "ٍ": "en",     # KASRATAN
}

SHADDA = "ّ"       # gemination: holds the consonant before it

# A held consonant is written with a trailing colon, the IPA length mark, so a
# geminate stays one phoneme (one viseme) and only its duration changes.
HELD = "ː"


# The letters that can spell a vowel. Two vowels never stand next to each other
# in Persian, so a WAW or YEH immediately before one of these must be the
# consonant /v/ or the glide /j/ -- which is what turns the medial WAW of
# "pahlavan" from a wrong /u/ into a right /v/.
VOWEL_LETTERS = frozenset(("ا", "آ", "و", "ی"))


def _letter(char, position):
    """The phoneme(s) one letter spells at one position in the word."""
    if char in POSITIONAL:
        return POSITIONAL[char][position]
    return CONSONANTS.get(char)


def _spell(word):
    """Letters to phonemes, before any vowel is guessed.

    Normalisation is repeated here rather than assumed. :func:`words` already
    does it, but :func:`g2p_word` is public and a caller reaching it with a raw
    word must not silently get a different answer for two spellings that
    :data:`FOLD` exists to make identical. It is idempotent, so the second pass
    over an already-folded word costs a scan and changes nothing.
    """
    letters = [c for c in normalise(word) if c != ZWNJ]
    out = []
    for index, char in enumerate(letters):
        if char == SHADDA:
            if out:
                out[-1] = out[-1] + HELD
            continue
        if char in HARAKAT:
            out.extend(HARAKAT[char])
            continue
        position = 0 if index == 0 else (2 if index == len(letters) - 1 else 1)
        following = letters[index + 1] if index + 1 < len(letters) else ""
        if position == 1 and char in ("و", "ی") and following in VOWEL_LETTERS:
            position = 0                       # read it as the consonant
        sound = _letter(char, position)
        if sound is None:
            continue
        out.append(sound)
    return out


# Persian syllables are (C)V(C)(C). Two consequences, and between them they
# recover most of the vowels the script leaves out:
#
# * **A syllable never opens with two consonants.** So a word beginning with two
#   -- which is what the spelling of "rostam" hands over, r-s-t-m -- is missing
#   a vowel between them.
# * **A syllable carries at most two consonants in its coda.** So a run of three
#   inside a word is a syllable boundary, and the vowel goes before the third,
#   which is where the next syllable has to start.
#
# Applied in that order, r-s-t-m becomes r-e-s-t-e-m and p-h-l-v-a-n becomes
# p-e-h-l-e-v-a-n: the right number of syllables in the right places, with the
# vowel *qualities* wrong (the real words are rostam and pahlavan). That trade is
# the correct one here -- syllable count is what a mouth shows, and vowel quality
# is what needs a pronunciation lexicon this fallback is allowed not to have.
MAX_CONSONANT_RUN = 2

# The vowel to guess with. /e/ is the commonest Persian short vowel and the one
# an epenthesis in a cluster almost always turns out to be.
EPENTHETIC = "e"


def _is_vowel(phone):
    return phone.rstrip(HELD) in VOWELS


def _vowelise(phones):
    """Insert the short vowels the script did not write.

    The count comes back so the stage can report how much of the line was
    reconstructed rather than read.
    """
    out, guessed = list(phones), 0
    if len(out) >= 2 and not _is_vowel(out[0]) and not _is_vowel(out[1]):
        out.insert(1, EPENTHETIC)
        guessed += 1

    spaced, run = [], 0
    for phone in out:
        if _is_vowel(phone):
            run = 0
        else:
            if run >= MAX_CONSONANT_RUN:
                spaced.append(EPENTHETIC)
                guessed += 1
                run = 0
            run += 1
        spaced.append(phone)

    if spaced and not any(_is_vowel(p) for p in spaced):
        spaced.insert(1 if len(spaced) > 1 else 0, EPENTHETIC)
        guessed += 1
    return spaced, guessed


def g2p_word(word):
    """One Persian word as a phoneme list, plus how many vowels were guessed."""
    return _vowelise(_spell(word))


def g2p(text):
    """Every word of a line, as ``[(word, [phone, ...]), ...]``, plus a count."""
    out, guessed = [], 0
    for word in words(text):
        phones, count = g2p_word(word)
        if not phones:
            continue
        out.append((word, phones))
        guessed += count
    return out, guessed


# -- visemes -----------------------------------------------------------------
# Twelve mouth shapes. The four sounds English has not got are all articulated
# behind the lips, so x, gh and the glottal stop join the velar group and zh
# joins the post-alveolar one -- which is the whole reason a Persian viseme set
# is not much bigger than an English one.
VISEME_OF = {
    AA: "AA", "a": "AA",
    "e": "E",
    "i": "I", "j": "I",
    "o": "O",
    "u": "U",
    "m": "MBP", "b": "MBP", "p": "MBP",
    "f": "FV", "v": "FV",
    "s": "SS", "z": "SS",
    SH: "SH", ZH: "SH", "t" + SH: "SH", "d" + ZH: "SH",
    "t": "TD", "d": "TD", "n": "TD", "l": "TD", "r": "TD",
    "k": "KG", "g": "KG", "x": "KG", GH: "KG", "h": "KG", GLOTTAL: "KG",
}

REST_VISEME = "sil"

# Each viseme as the four mouth channels of ``validate.VISEME_CHANNELS``.
# ``lip_press`` is what closes a mouth on m/b/p, and it is the one channel that
# has to reach nearly 1: a bilabial that does not shut is the most visible
# lip-sync failure there is.
VISEME_SHAPE = {
    "sil": {"jaw_open": 0.00, "lip_wide": 0.00, "lip_round": 0.00, "lip_press": 0.00},
    "AA":  {"jaw_open": 0.88, "lip_wide": 0.22, "lip_round": 0.00, "lip_press": 0.00},
    "E":   {"jaw_open": 0.46, "lip_wide": 0.56, "lip_round": 0.00, "lip_press": 0.00},
    "I":   {"jaw_open": 0.20, "lip_wide": 0.82, "lip_round": 0.00, "lip_press": 0.00},
    "O":   {"jaw_open": 0.46, "lip_wide": 0.00, "lip_round": 0.68, "lip_press": 0.00},
    "U":   {"jaw_open": 0.20, "lip_wide": 0.00, "lip_round": 0.88, "lip_press": 0.00},
    "MBP": {"jaw_open": 0.02, "lip_wide": 0.00, "lip_round": 0.04, "lip_press": 0.95},
    "FV":  {"jaw_open": 0.13, "lip_wide": 0.30, "lip_round": 0.00, "lip_press": 0.55},
    "SS":  {"jaw_open": 0.15, "lip_wide": 0.66, "lip_round": 0.00, "lip_press": 0.10},
    "SH":  {"jaw_open": 0.22, "lip_wide": 0.04, "lip_round": 0.56, "lip_press": 0.10},
    "TD":  {"jaw_open": 0.28, "lip_wide": 0.34, "lip_round": 0.00, "lip_press": 0.04},
    "KG":  {"jaw_open": 0.38, "lip_wide": 0.18, "lip_round": 0.00, "lip_press": 0.00},
}

VISEMES = tuple(sorted(VISEME_SHAPE))

# The mouth half of validate.VISEME_CHANNELS. The other four are the face, and
# they come from the line's emotion rather than from its sounds.
MOUTH_CHANNELS = ("jaw_open", "lip_wide", "lip_round", "lip_press")


def viseme_of(phone):
    """The mouth shape one phoneme is made with. An unknown sound rests."""
    return VISEME_OF.get((phone or "").rstrip(HELD), REST_VISEME)


def shape_of(viseme):
    """The four mouth channels for one viseme."""
    return dict(VISEME_SHAPE.get(viseme, VISEME_SHAPE[REST_VISEME]))


# -- duration weights --------------------------------------------------------
# How long a sound takes relative to its neighbours, used to share a word's
# measured duration out among its phonemes. Vowels carry the syllable and are
# longest; a stop is a closure and a release and is over quickly. These are
# ratios and never seconds -- the seconds come from the audio.
DURATION_WEIGHT = {
    AA: 1.9, "u": 1.6, "i": 1.6, "a": 1.2, "e": 1.1, "o": 1.2,
    "p": 0.7, "b": 0.7, "t": 0.7, "d": 0.7, "k": 0.8, "g": 0.8,
    GLOTTAL: 0.5,
    "f": 1.1, "v": 1.0, "s": 1.3, "z": 1.1, SH: 1.3, ZH: 1.1,
    "x": 1.1, GH: 1.0, "h": 0.9,
    "t" + SH: 1.1, "d" + ZH: 1.1,
    "m": 0.9, "n": 0.9, "l": 0.9, "r": 0.7, "j": 0.7,
}
DEFAULT_WEIGHT = 1.0

# A geminate is held; that is the only thing that distinguishes it.
GEMINATE_FACTOR = 1.7


def weight_of(phone):
    """The relative duration of one phoneme."""
    base = (phone or "").rstrip(HELD)
    weight = DURATION_WEIGHT.get(base, DEFAULT_WEIGHT)
    return weight * (GEMINATE_FACTOR if phone.endswith(HELD) else 1.0)


# -- emotion -----------------------------------------------------------------
# Decision 2 keeps facial expression out of the performance video, so the face's
# emotion has exactly one source: the ``emotion`` field section 6.1 puts on
# every dialogue line. It is expanded here, once, into the four non-mouth
# channels; the style pack scales it afterwards through ``brow_gain``.
EMOTION_SHAPE = {
    "neutral":  {"brow_raise": 0.06, "brow_furrow": 0.00, "squint": 0.04, "smile": 0.04},
    "grave":    {"brow_raise": 0.00, "brow_furrow": 0.46, "squint": 0.22, "smile": 0.00},
    "resolute": {"brow_raise": 0.14, "brow_furrow": 0.30, "squint": 0.20, "smile": 0.02},
    "angry":    {"brow_raise": 0.00, "brow_furrow": 0.72, "squint": 0.38, "smile": 0.00},
    "defiant":  {"brow_raise": 0.10, "brow_furrow": 0.52, "squint": 0.26, "smile": 0.00},
    "sorrow":   {"brow_raise": 0.38, "brow_furrow": 0.30, "squint": 0.14, "smile": 0.00},
    "grief":    {"brow_raise": 0.44, "brow_furrow": 0.36, "squint": 0.20, "smile": 0.00},
    "tender":   {"brow_raise": 0.18, "brow_furrow": 0.00, "squint": 0.08, "smile": 0.30},
    "joy":      {"brow_raise": 0.30, "brow_furrow": 0.00, "squint": 0.16, "smile": 0.62},
    "fear":     {"brow_raise": 0.62, "brow_furrow": 0.16, "squint": 0.00, "smile": 0.00},
    "awe":      {"brow_raise": 0.52, "brow_furrow": 0.00, "squint": 0.00, "smile": 0.08},
    "weary":    {"brow_raise": 0.10, "brow_furrow": 0.18, "squint": 0.34, "smile": 0.00},
    "scornful": {"brow_raise": 0.04, "brow_furrow": 0.34, "squint": 0.30, "smile": 0.12},
}
# production.schema.json's emotion enum says "grieving", "joyful" and "afraid";
# this table grew up saying "grief", "joy" and "fear". The schema is what the
# main prompt and the plan are validated against, so its words are aliases here
# rather than a second vocabulary.
EMOTION_SHAPE["grieving"] = EMOTION_SHAPE["grief"]
EMOTION_SHAPE["joyful"] = EMOTION_SHAPE["joy"]
EMOTION_SHAPE["afraid"] = EMOTION_SHAPE["fear"]

FACE_CHANNELS = ("brow_raise", "brow_furrow", "squint", "smile")


def emotion_shape(name):
    """The brow and eye channels for a named emotion, and whether it was known.

    An unknown emotion is not an error -- a chat model will invent adjectives
    forever -- but it must not pass as understood either, so the caller gets the
    neutral face and a ``False``, which ``g_align`` turns into a warning.
    """
    key = (name or "neutral").strip().lower()
    if key in EMOTION_SHAPE:
        return dict(EMOTION_SHAPE[key]), True
    return dict(EMOTION_SHAPE["neutral"]), False
