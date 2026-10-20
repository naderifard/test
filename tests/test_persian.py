"""The Persian G2P and viseme map, checked against the language.

Section 7-G's fallback route reconstructs sounds the script does not write, so
the thing worth testing is not "does it return a list" but "does it obey the
phonotactics it claims to obey". Three rules carry all the weight:

* a Persian syllable never opens with two consonants,
* a syllable never carries three consonants in its coda,
* a word-final heh is the vowel /e/, not /h/,

and each of them, if it breaks, produces a mouth that is wrong for a whole line
without anything crashing. The words used below are the ones the slice actually
says, plus the two character names, so a regression shows up as a name the film
mispronounces rather than as an abstract failure.

Romanisations in the comments are the ordinary scholarly ones.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import persian                                   # noqa: E402
from pipeline.validate import VISEME_CHANNELS                  # noqa: E402

# The two spoken lines of the slice.
SOHRAB_LINE = "نام تو چیست، ای پهلوان؟"
ROSTAM_LINE = "نام من در این میدان، تیغ من است."


def sounds(word):
    return persian.g2p_word(word)[0]


class TestNormalisation(unittest.TestCase):

    def test_arabic_and_persian_yeh_fold_to_one_letter(self):
        """The classic invisible bug: two identical-looking letters."""
        arabic = "مي"      # with ARABIC YEH
        persian_ = "می"    # with FARSI YEH
        self.assertNotEqual(arabic, persian_, "the raw strings differ")
        self.assertEqual(persian.normalise(arabic), persian.normalise(persian_))
        self.assertEqual(sounds(arabic), sounds(persian_))

    def test_arabic_kaf_folds_to_keheh(self):
        self.assertEqual(sounds("كم"), sounds("کم"))

    def test_punctuation_and_tatweel_are_not_sounds(self):
        self.assertEqual(persian.words("نام، تو؟"), ["نام", "تو"])
        self.assertEqual(sounds("نـام"), sounds("نام"))

    def test_zwnj_keeps_a_compound_as_one_word(self):
        found = persian.words("جوان‌مرد است")
        self.assertEqual(len(found), 2, "ZWNJ joins, it does not separate")


class TestGraphemeToPhoneme(unittest.TestCase):

    def test_a_word_never_opens_with_two_consonants(self):
        """Persian syllables are (C)V(C)(C); an onset cluster cannot happen."""
        for line in (SOHRAB_LINE, ROSTAM_LINE):
            for word, phones in persian.g2p(line)[0]:
                if len(phones) < 2:
                    continue
                opens_with_two = (phones[0].rstrip(persian.HELD)
                                  not in persian.VOWELS
                                  and phones[1].rstrip(persian.HELD)
                                  not in persian.VOWELS)
                self.assertFalse(opens_with_two,
                                 f"{word} spells an onset cluster: {phones}")

    def test_no_run_of_three_consonants_survives(self):
        for word in ("رستم", "پهلوان", "سهراب", "رخش", "فردوسی"):
            run = 0
            for phone in sounds(word):
                if phone.rstrip(persian.HELD) in persian.VOWELS:
                    run = 0
                else:
                    run += 1
                self.assertLessEqual(run, persian.MAX_CONSONANT_RUN,
                                     f"{word}: {sounds(word)}")

    def test_every_word_has_at_least_one_vowel(self):
        for line in (SOHRAB_LINE, ROSTAM_LINE):
            for word, phones in persian.g2p(line)[0]:
                self.assertTrue(
                    any(p.rstrip(persian.HELD) in persian.VOWELS
                        for p in phones),
                    f"{word} came out with no vowel to open a mouth on")

    def test_a_word_final_heh_is_the_vowel_e(self):
        """xane, not xaneh - the commonest word ending in the language."""
        self.assertEqual(sounds("خانه"), ["x", "ɒ", "n", "e"])

    def test_a_medial_waw_before_alef_is_the_consonant_v(self):
        """pahlavan has a /v/ in it, not a /u/: two vowels never adjoin."""
        self.assertIn("v", sounds("پهلوان"))
        self.assertNotIn("u", sounds("پهلوان"))

    def test_a_final_waw_is_still_the_vowel_u(self):
        self.assertEqual(sounds("تو"), ["t", "u"])

    def test_the_names_of_the_film_come_out_with_the_right_syllable_count(self):
        # rostam, sohrab, raxs: two syllables, two, one. Vowel *quality* is
        # allowed to be wrong - that needs a lexicon - but the count is what a
        # mouth shows and it has to be right.
        def syllables(word):
            return sum(1 for p in sounds(word)
                       if p.rstrip(persian.HELD) in persian.VOWELS)
        self.assertEqual(syllables("رستم"), 2)
        self.assertEqual(syllables("سهراب"), 2)
        self.assertEqual(syllables("رخش"), 1)
        self.assertEqual(syllables("شاهنامه"), 3)

    def test_written_vowel_diacritics_are_honoured_not_stripped(self):
        with_fatha = sounds("رَستم")
        self.assertEqual(with_fatha[:2], ["r", "a"],
                         "a written fatha beats the guessed epenthetic")

    def test_shadda_holds_one_phoneme_rather_than_repeating_it(self):
        held = sounds("ربّ")
        doubled = [p for p in held if p.endswith(persian.HELD)]
        self.assertEqual(len(doubled), 1, held)
        self.assertGreater(persian.weight_of(doubled[0]),
                           persian.weight_of(doubled[0].rstrip(persian.HELD)))

    def test_the_guessed_vowel_count_is_reported_not_hidden(self):
        _phones, guessed = persian.g2p_word("رستم")
        self.assertGreaterEqual(guessed, 1,
                                "the stage reports how much it reconstructed")


class TestVisemes(unittest.TestCase):

    def test_every_phoneme_the_g2p_can_emit_has_a_mouth_shape(self):
        emitted = set()
        for value in persian.CONSONANTS.values():
            emitted.add(value)
        for triple in persian.POSITIONAL.values():
            emitted.update(triple)
        emitted.update(persian.VOWELS)
        emitted.add(persian.EPENTHETIC)
        for phone in sorted(emitted):
            self.assertIn(phone, persian.VISEME_OF,
                          f"{phone!r} can be produced and has no viseme")

    def test_every_viseme_has_a_shape_over_the_declared_channels(self):
        for viseme in persian.VISEMES:
            shape = persian.shape_of(viseme)
            self.assertEqual(set(shape), set(persian.MOUTH_CHANNELS))
            for name, value in shape.items():
                self.assertTrue(0.0 <= value <= 1.0, f"{viseme}.{name}={value}")

    def test_the_channels_are_the_ones_the_rig_carries(self):
        """persian + validate must agree, or a curve drives nothing."""
        declared = set(persian.MOUTH_CHANNELS) | set(persian.FACE_CHANNELS)
        self.assertEqual(declared, set(VISEME_CHANNELS))

    def test_a_bilabial_closes_the_mouth_and_an_open_vowel_opens_it(self):
        # The two failures an audience notices instantly.
        self.assertGreater(persian.shape_of("MBP")["lip_press"], 0.8)
        self.assertLess(persian.shape_of("MBP")["jaw_open"], 0.1)
        self.assertGreater(persian.shape_of("AA")["jaw_open"], 0.7)

    def test_an_unknown_sound_rests_rather_than_guessing(self):
        self.assertEqual(persian.viseme_of("ñ"), persian.REST_VISEME)
        self.assertEqual(persian.shape_of(persian.REST_VISEME),
                         {name: 0.0 for name in persian.MOUTH_CHANNELS})


class TestEmotion(unittest.TestCase):

    def test_every_emotion_drives_only_face_channels(self):
        for name in persian.EMOTION_SHAPE:
            shape, known = persian.emotion_shape(name)
            self.assertTrue(known)
            self.assertEqual(set(shape), set(persian.FACE_CHANNELS))
            for channel, value in shape.items():
                self.assertTrue(0.0 <= value <= 1.0, f"{name}.{channel}")

    def test_an_unknown_emotion_is_neutral_and_says_so(self):
        shape, known = persian.emotion_shape("incandescent")
        self.assertFalse(known, "an invented adjective must not pass as known")
        self.assertEqual(shape, persian.EMOTION_SHAPE["neutral"])

    def test_the_emotions_the_slice_uses_are_all_known(self):
        for name in ("grave", "resolute"):
            self.assertTrue(persian.emotion_shape(name)[1])


class TestDurationWeights(unittest.TestCase):

    def test_vowels_are_longer_than_stops(self):
        self.assertGreater(persian.weight_of("ɒ"), persian.weight_of("t"))
        self.assertGreater(persian.weight_of("i"), persian.weight_of("p"))

    def test_an_unknown_phone_still_has_a_weight(self):
        self.assertEqual(persian.weight_of("ñ"), persian.DEFAULT_WEIGHT)


if __name__ == "__main__":
    unittest.main()
