<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/as-ideas/TransformerTTS/master/docs/transformer_logo.png" width="400"/>
    <br>
</p>

<h2 align="center">
<p>A Text-to-Speech Transformer in TensorFlow 2</p>
</h2>

<p class="text">Samples are converted using the pre-trained <a href="https://github.com/fatchord/WaveRNN"> WaveRNN </a> or <a href="https://github.com/seungwonpark/melgan"> MelGAN </a> vocoders.
</p>

## 🎧 Model samples

<p class="text">President Trump met with other leaders at the Group of twenty conference.</p>

| forward + wavernn | autoregressive + wavernn |
|:---:|:---:|
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward_transformer/trump.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_transformertts/Trump.wav?raw=true" controls preload></audio>|
| <b>forward + melgan</b> | <b>autoregressive + melgan</b> |
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_transformerTTS_melgan/trump_forward_melgan.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_transformerTTS_melgan/trump_autoregressive_melgan.wav?raw=true" controls preload></audio>|

<p class="text">Scientists at the CERN laboratory, say they have discovered a new particle.</p>

| forward + wavernn | autoregressive + wavernn |
|:---:|:---:|
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward_transformer/scientists.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_transformertts/cern_particle.wav?raw=true" controls preload></audio>|
| <b>forward + melgan</b> | <b>autoregressive + melgan</b> |
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_transformerTTS_melgan/scientists_forward_melgan.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_transformerTTS_melgan/scientists_autoregressive_melgan.wav?raw=true" controls preload></audio>|

<p class="text">There’s a way to measure the acute emotional intelligence that has never gone out of style.</p>

| forward + wavernn | autoregressive + wavernn |
|:---:|:---:|
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward_transformer/EQ.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_transformertts/EQ.wav?raw=true" controls preload></audio>|
| <b>forward + melgan</b> | <b>autoregressive + melgan</b> |
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_transformerTTS_melgan/EQ_forward_melgan.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_transformerTTS_melgan/EQ_autoregressive_melgan.wav?raw=true" controls preload></audio>|

<p class="text">The Senate's bill to repeal and replace the Affordable Care-Act is now imperiled.</p>

| forward + wavernn | autoregressive + wavernn |
|:---:|:---:|
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward_transformer/senate.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_transformertts/affordablecareact.wav?raw=true" controls preload></audio>|
| <b>forward + melgan</b> | <b>autoregressive + melgan</b> |
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_transformerTTS_melgan/senate_forward_melgan.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_transformerTTS_melgan/senate_autoregressive_melgan.wav?raw=true" controls preload></audio>|


### Robustness

<p class="text">To deliver interfaces that are significantly better suited to create and process RFC eight twenty one , RFC eight twenty two , RFC nine seventy seven , and MIME content.</p>

| forward | autoregressive |
|:---:|:---:|
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward_transformer/hard.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_transformertts/hard.wav?raw=true" controls preload></audio>|

### Speed control
<p class="text">For a while the preacher addresses himself to the congregation at large, who listen attentively.</p>

| 10% slower | normal speed | 25% faster |
|:---:|:---:|:---:|
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward_transformer/speed_090.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward_transformer/speed_100.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward_transformer/speed_125.wav?raw=true" controls preload></audio>|

### Comparison with [ForwardTacotron](https://github.com/as-ideas/ForwardTacotron)
<p class="text"> In a statement announcing his resignation, Mr Ross, said: "While the intentions may have been well meaning, the reaction to this news shows that Mr Cummings interpretation of the government advice was not shared by the vast majority of people who have done as the government asked."</p>

| ForwardTacotron | TransformerTTS |
|:---:|:---:|
|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward/forward_transformer_comparison.wav?raw=true" controls preload></audio>|<audio src="https://github.com/as-ideas/tts_model_outputs/blob/master/ljspeech_forward_transformer/tacotron_comparison.wav?raw=true" controls preload></audio>|
