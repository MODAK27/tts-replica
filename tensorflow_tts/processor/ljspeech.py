# -*- coding: utf-8 -*-
# This code is copy and modify from https://github.com/keithito/tacotron.
"""Perform preprocessing and raw feature extraction."""

import os
import re

import numpy as np
import soundfile as sf

from tensorflow_tts.utils import cleaners

valid_symbols = [
    "AA",
    "AA0",
    "AA1",
    "AA2",
    "AE",
    "AE0",
    "AE1",
    "AE2",
    "AH",
    "AH0",
    "AH1",
    "AH2",
    "AO",
    "AO0",
    "AO1",
    "AO2",
    "AW",
    "AW0",
    "AW1",
    "AW2",
    "AY",
    "AY0",
    "AY1",
    "AY2",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "EH0",
    "EH1",
    "EH2",
    "ER",
    "ER0",
    "ER1",
    "ER2",
    "EY",
    "EY0",
    "EY1",
    "EY2",
    "F",
    "G",
    "HH",
    "IH",
    "IH0",
    "IH1",
    "IH2",
    "IY",
    "IY0",
    "IY1",
    "IY2",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OW0",
    "OW1",
    "OW2",
    "OY",
    "OY0",
    "OY1",
    "OY2",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UH0",
    "UH1",
    "UH2",
    "UW",
    "UW0",
    "UW1",
    "UW2",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]
'''
The current phoneme set has 39 phonemes, not counting varia due to lexical stress. This phoneme (or more accurately, phone) set is based on the ARPAbet symbol set developed for speech recognition uses. You can find a description of the ARPAbet on Wikipedia, as well information on how it relates to the standard IPA symbol set.
        Phoneme Example Translation
        ------- ------- -----------
        AA	odd     AA D
        AE	at	AE T
        AH	hut	HH AH T
        AO	ought	AO T
        AW	cow	K AW
        AY	hide	HH AY D
        B 	be	B IY
        CH	cheese	CH IY Z
        D 	dee	D IY
        DH	thee	DH IY
        EH	Ed	EH D
        ER	hurt	HH ER T
        EY	ate	EY T
        F 	fee	F IY
        G 	green	G R IY N
        HH	he	HH IY
        IH	it	IH T
        IY	eat	IY T
        JH	gee	JH IY
        K 	key	K IY
        L 	lee	L IY
        M 	me	M IY
        N 	knee	N IY
        NG	ping	P IH NG
        OW	oat	OW T
        OY	toy	T OY
        P 	pee	P IY
        R 	read	R IY D
        S 	sea	S IY
        SH	she	SH IY
        T 	tea	T IY
        TH	theta	TH EY T AH
        UH	hood	HH UH D
        UW	two	T UW
        V 	vee	V IY
        W 	we	W IY
        Y 	yield	Y IY L D
        Z 	zee	Z IY
        ZH	seizure	S IY ZH ER

“ARPAbet is a phonetic transcription code developed by Advanced Research
Projects Agency (ARPA) as a part of their Speech Understanding Project (1971–1976). It
represents each phoneme of General American English with a distinct sequence of ASCII
characters. 
'''

_pad = "_"
_eos = "~"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in valid_symbols]

# Export all symbols:
symbols = (
    [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet + [_eos]
)

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


class LJSpeechProcessor(object):
    """LJSpeech processor."""

    def __init__(self, root_path, cleaner_names):
        self.root_path = root_path
        self.cleaner_names = cleaner_names

        items = []#to contain text + wav_path
        self.speaker_name = "ljspeech"
        if root_path is not None:
            with open(os.path.join(root_path, "metadata.csv"), encoding="utf-8") as ttf:
                for line in ttf:
                    parts = line.strip().split("|")
                    wav_path = os.path.join(root_path, "wavs", "%s.wav" % parts[0])
                    text = parts[2]
                    items.append([text, wav_path, self.speaker_name])

            self.items = items

    def get_one_sample(self, idx):
        text, wav_file, speaker_name = self.items[idx]

        # normalize audio signal to be [-1, 1], soundfile already norm.
        audio, rate = sf.read(wav_file)
        audio = audio.astype(np.float32)

        # convert text to ids
        text_ids = np.asarray(self.text_to_sequence(text), np.int32)

        sample = {
            "raw_text": text,
            "text_ids": text_ids,
            "audio": audio,
            "utt_id": self.items[idx][1].split("/")[-1].split(".")[0],
            "speaker_name": speaker_name,
            "rate": rate,
        }

        return sample

    def text_to_sequence(self, text):
        global _symbol_to_id

        sequence = []
        # Check for curly braces and treat their contents as ARPAbet:
        while len(text):
            m = _curly_re.match(text)
            if not m:
                sequence += _symbols_to_sequence(
                    _clean_text(text, [self.cleaner_names])
                )
                break
            sequence += _symbols_to_sequence(
                _clean_text(m.group(1), [self.cleaner_names])
            )
            sequence += _arpabet_to_sequence(m.group(2))
            text = m.group(3)
        return sequence


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols):
    return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
    return _symbols_to_sequence(["@" + s for s in text.split()])


def _should_keep_symbol(s):
    return s in _symbol_to_id and s != "_" and s != "~"
