<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/as-ideas/TransformerTTS/master/docs/transformer_logo.png" width="400"/>
    <br>
</p>

<h2 align="center">
<p>A Text-to-Speech Transformer in TensorFlow 2</p>
</h2>

<p class="text">Samples are converted using the pre-trained <a href="https://github.com/jik876/hifi-gan"> HiFiGAN </a> vocoder and with the standard Griffin-Lim algorithm for comparison.
</p>

## 🎧 Model samples

<p class="text">Introductory speech ODSC Boston 2021</p>

<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/LJSpeech_TransformerTTS_hifigan/outputs/presentation_text/presentation_text_ljspeech_hifigan_bdf06b9_95000_new.wav?raw=true" controls preload></audio>


<p class="text">Peter piper picked a peck of pickled peppers.</p>

| <b>HiFiGAN</b> | <b>Griffin-Lim</b> |
|:---:|:---:|
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/LJSpeech_TransformerTTS_hifigan/outputs/test_sentences/test_sentences_ljspeech_hifigan_bdf06b9_95000_6.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/LJSpeech_TransformerTTS_glim/outputs/test_sentences/test_sentences_ljspeech_bdf06b9_95000_6.wav?raw=true" controls preload></audio>|

<p class="text">President Trump met with other leaders at the Group of twenty conference.</p>

| <b>HiFiGAN</b> | <b>Griffin-Lim</b> |
|:---:|:---:|
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/LJSpeech_TransformerTTS_hifigan/outputs/test_sentences/test_sentences_ljspeech_hifigan_bdf06b9_95000_2.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/LJSpeech_TransformerTTS_glim/outputs/test_sentences/test_sentences_ljspeech_bdf06b9_95000_2.wav?raw=true" controls preload></audio>|

<p class="text">Scientists at the CERN laboratory say they have discovered a new particle.</p>

| <b>HiFiGAN</b> | <b>Griffin-Lim</b> |
|:---:|:---:|
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/LJSpeech_TransformerTTS_hifigan/outputs/test_sentences/test_sentences_ljspeech_hifigan_bdf06b9_95000_3.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/LJSpeech_TransformerTTS_glim/outputs/test_sentences/test_sentences_ljspeech_bdf06b9_95000_3.wav?raw=true" controls preload></audio>|

<p class="text">There’s a way to measure the acute emotional intelligence that has never gone out of style.</p>

| <b>HiFiGAN</b> | <b>Griffin-Lim</b> |
|:---:|:---:|
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/LJSpeech_TransformerTTS_hifigan/outputs/test_sentences/test_sentences_ljspeech_hifigan_bdf06b9_95000_4.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/LJSpeech_TransformerTTS_glim/outputs/test_sentences/test_sentences_ljspeech_bdf06b9_95000_4.wav?raw=true" controls preload></audio>|

<p class="text">The Senate's bill to repeal and replace the Affordable Care-Act is now imperiled.</p>

| <b>HiFiGAN</b> | <b>Griffin-Lim</b> |
|:---:|:---:|
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/LJSpeech_TransformerTTS_hifigan/outputs/test_sentences/test_sentences_ljspeech_hifigan_bdf06b9_95000_5.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/LJSpeech_TransformerTTS_glim/outputs/test_sentences/test_sentences_ljspeech_bdf06b9_95000_5.wav?raw=true" controls preload></audio>|


<p class="text">If I were to talk to a human, I would definitely try to sound normal. Wouldn't I?</p>

| <b>HiFiGAN</b> | <b>Griffin-Lim</b> |
|:---:|:---:|
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/LJSpeech_TransformerTTS_hifigan/outputs/test_sentences/test_sentences_ljspeech_hifigan_bdf06b9_95000_7.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/LJSpeech_TransformerTTS_glim/outputs/test_sentences/test_sentences_ljspeech_bdf06b9_95000_7.wav?raw=true" controls preload></audio>|


### Robustness

<p class="text">To deliver interfaces that are significantly better suited to create and process RFC eight twenty one , RFC eight twenty two , RFC nine seventy seven , and MIME content.</p>

| <b>HiFiGAN</b> | <b>Griffin-Lim</b> |
|:---:|:---:|
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/LJSpeech_TransformerTTS_hifigan/outputs/mime_content/mime_content_ljspeech_hifigan_bdf06b9_95000.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/LJSpeech_TransformerTTS_glim/outputs/mime_content/mime_content_ljspeech_bdf06b9_95000.wav?raw=true" controls preload></audio>|


### Comparison with [ForwardTacotron](https://github.com/as-ideas/ForwardTacotron)
<p class="text"> In a statement announcing his resignation, Mr Ross, said: "While the intentions may have been well meaning, the reaction to this news shows that Mr Cummings interpretation of the government advice was not shared by the vast majority of people who have done as the government asked."</p>

| TransformerTTS | ForwardTacotron |
|:---:|:---:|
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/LJSpeech_TransformerTTS_hifigan/outputs/statement/statement_ljspeech_hifigan_bdf06b9_95000.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/forward_transformer_comparison.wav?raw=true" controls preload></audio>|
