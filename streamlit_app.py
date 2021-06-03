import io
import logging

import streamlit as st
from scipy.io.wavfile import write
import tensorflow as tf
import numpy as np
from spacy.lang.en import English

from data.audio import Audio
from model.factory import tts_ljspeech
from vocoding.predictors import HiFiGANPredictor, MelGANPredictor

tf.get_logger().setLevel('ERROR')
logging.root.handlers = []
logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(funcName)s: %(message)s',
                    level=logging.DEBUG)


def audio(wav, sr=22050):
    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    write(byte_io, sr, wav)
    st.audio(byte_io, format='audio/wav')


@st.cache(allow_output_mutation=True)
def get_vocoder(voc_option):
    if voc_option == 'MelGAN':
        vocoder = MelGANPredictor.pretrained()
    else:
        vocoder = HiFiGANPredictor.pretrained()
    return vocoder


@st.cache(allow_output_mutation=True)
def get_tts(step='95000'):
    model = tts_ljspeech(step)
    return model


@st.cache(allow_output_mutation=True)
def get_nlp():
    nlp = English()
    nlp.add_pipe('sentencizer')
    return nlp


st.title('Text to Speech')
st.markdown('Text to speech conversion with [TransformerTTS](https://github.com/as-ideas/TransformerTTS). Based on the open source dataset [LJSpeech](https://keithito.com/LJ-Speech-Dataset/).')
st.markdown('With [MelGAN](https://github.com/seungwonpark/melgan) or [HiFiGAN](https://github.com/jik876/hifi-gan) vocoders.')


input_text = st.text_area(label='Type in some text',
                          value='Hello there, my name is LJ, an open-source voice.\n'
                                'Not to brag, but I am a fairly popular open-source voice.\n'
                                'A voice with a character.')

nlp = get_nlp()
all_sentences = []
blocks = input_text.split('\n')
blocks = [block for block in blocks if len(block) > 0]
for block in blocks:
    doc = nlp(block)
    sentences = [str(sent) for sent in doc.sents if len(str(sent)) > 0]
    all_sentences.extend(sentences)
all_sentences = [s.strip() for s in all_sentences]
all_sentences = [s for s in all_sentences if len(s) > 0]
logging.info(all_sentences)

# source of crashes
# model_weights = [f'{x}' for x in np.arange(100_000, 60_000, -5_000)]
# tts_option = st.selectbox('Select TTS model (training steps)', model_weights, index=1)
# model = get_tts(tts_option)
model = get_tts()
audio_class = Audio.from_config(model.config)

voc_option = st.selectbox('Select Vocoder model', ['HiFiGAN', 'MelGAN', 'Griffin-Lim (no vocoder)'], index=0)

all_wavs = []
for sentence in all_sentences:
    out = model.predict(sentence)
    mel = out['mel'].numpy().T
    if voc_option == 'Griffin-Lim (no vocoder)':
        wav = audio_class.reconstruct_waveform(out['mel'].numpy().T)
    else:
        vocoder = get_vocoder(voc_option)
        wav = vocoder([mel])[0]
    all_wavs.append(wav)
wavs = []
if len(all_wavs) > 0:
    wavs = np.concatenate(all_wavs)
if len(wavs) > 0:
    audio(wavs, sr=audio_class.sampling_rate)
