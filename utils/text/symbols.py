""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from utils.text import cmudict

_pad = '_'
_punctuation = '!\'(),.:;? '
_special = '-'

_phonemes = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'æ', 'ç', 'ð', 'ø', 'ŋ', 'œ', 'ɐ', 'ɑ', 'ɔ', 'ə', 'ɛ', 'ɝ', 'ɡ', 'ɪ', 'ʁ', 'ʃ', 'ʊ', 'ʌ', 'ʏ', 'ʒ', 'ʔ', 'ː', '̃', '̍', '̩', 'θ']

phonemes = list(_pad + _punctuation + _special) + _phonemes
phonemes_set = set(phonemes)