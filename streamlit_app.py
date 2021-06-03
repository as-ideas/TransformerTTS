import io
import logging

import streamlit as st
from scipy.io.wavfile import write
import tensorflow as tf
import numpy as np
import torch
from spacy.lang.en import English

from data.audio import Audio
from model.factory import tts_ljspeech
from vocoding.melgan.generator import Generator

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
def get_vocoder():
    dictionary = torch.hub.load_state_dict_from_url(
        'https://github.com/seungwonpark/melgan/releases/download/v0.3-alpha/nvidia_tacotron2_LJ11_epoch6400.pt',
        map_location='cpu')
    vocoder = Generator(80, num_layers=[3, 3, 3, 3])
    vocoder.load_state_dict(dictionary['model_g'])
    vocoder.eval()
    return vocoder


@st.cache(allow_output_mutation=True)
def get_tts():
    model = tts_ljspeech('95000')
    return model


@st.cache(allow_output_mutation=True)
def get_nlp():
    nlp = English()
    nlp.add_pipe('sentencizer')
    return nlp


st.title('Text to Speech')
st.markdown(
    'Text to Speech with [TransformerTTS](https://github.com/as-ideas/TransformerTTS) and [MelGAN](https://github.com/seungwonpark/melgan)')

input_text = st.text_area(label='Type in some text',
                          value='Hello there, my name is LJ, an open-source voice.\n'
                                'Not to brag, but I am a fairly popular open-source voice.\n'
                                'A voice with a character.')

vocoder_type = 'melgan'
if st.button('GriffinLim (no vocoder)'):
    vocoder_type = 'griffinlim'

if st.button('MelGAN'):
    vocoder_type = 'melgan'

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
model = get_tts()
audio_class = Audio.from_config(model.config)

all_wavs = []
for sentence in all_sentences:
    out = model.predict(sentence)
    mel = out['mel'].numpy().T
    if vocoder_type == 'griffinlim':
        wav = audio_class.reconstruct_waveform(out['mel'].numpy().T)
    else:
        vocoder = get_vocoder()
        mel = torch.tensor([mel])
        if torch.cuda.is_available():
            vocoder = vocoder.cuda()
            mel = mel.cuda()
        
        with torch.no_grad():
            wav = vocoder.inference(mel).numpy()
    all_wavs.append(wav)
wavs = []
if len(all_wavs) > 0:
    wavs = np.concatenate(all_wavs)
if len(wavs) > 0:
    audio(wavs, sr=audio_class.sampling_rate)
