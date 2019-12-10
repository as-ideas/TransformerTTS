import tensorflow as tf

from model.layers import Encoder, Decoder, SpeechPostnet, PointWiseFFN, ReluFeedForward, TextPostnet
from model.models import TextTransformer, MelTransformer, MelTextTransformer, TextMelTransformer


def new_text_transformer(start_token_index,
                         end_token_index,
                         input_vocab_size,
                         target_vocab_size,
                         num_layers=1,
                         d_model=64,
                         num_heads=1,
                         dff=512,
                         max_position_encoding=10000,
                         dropout_rate=0):

    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=max_position_encoding,
        rate=dropout_rate,
    )

    decoder = Decoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=max_position_encoding,
        rate=dropout_rate,
    )

    text_transformer = TextTransformer(
        encoder_prenet=tf.keras.layers.Embedding(input_vocab_size, d_model),
        decoder_prenet=tf.keras.layers.Embedding(target_vocab_size, d_model),
        decoder_postnet=TextPostnet(target_vocab_size),
        encoder=encoder,
        decoder=decoder,
        start_token_index=start_token_index,
        end_token_index=end_token_index
    )

    return text_transformer


def new_mel_transformer(start_vec,
                        stop_prob_index,
                        num_layers=1,
                        d_model=64,
                        num_heads=1,
                        dff=512,
                        dff_prenet=512,
                        max_position_encoding=10000,
                        dropout_rate=0,
                        mel_channels=80,
                        postnet_conv_filters=32,
                        postnet_conv_layers=2,
                        postnet_kernel_size=5):

    encoder = Encoder(num_layers=num_layers,
                      d_model=d_model,
                      num_heads=num_heads,
                      dff=dff,
                      maximum_position_encoding=max_position_encoding,
                      rate=dropout_rate)

    decoder = Decoder(num_layers=num_layers,
                      d_model=d_model,
                      num_heads=num_heads,
                      dff=dff,
                      maximum_position_encoding=max_position_encoding,
                      rate=dropout_rate)

    speech_out_module = SpeechPostnet(mel_channels=mel_channels,
                                        conv_filters=postnet_conv_filters,
                                        conv_layers=postnet_conv_layers,
                                        kernel_size=postnet_kernel_size)

    mel_transformer = MelTransformer(encoder_prenet=PointWiseFFN(d_model=d_model, dff=dff_prenet),
                                     decoder_prenet=ReluFeedForward(d_model=d_model, dff=dff_prenet),
                                     encoder=encoder,
                                     decoder=decoder,
                                     decoder_postnet=speech_out_module,
                                     start_vec=start_vec,
                                     stop_prob_index=stop_prob_index)

    return mel_transformer


def new_mel_text_transformer(start_token_index,
                             end_token_index,
                             target_vocab_size,
                             mel_channels=80,
                             num_layers=1,
                             d_model=64,
                             num_heads=1,
                             dff=512,
                             dff_prenet=512,
                             max_position_encoding=10000,
                             dropout_rate=0):

    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=max_position_encoding,
        rate=dropout_rate,
    )

    decoder = Decoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=max_position_encoding,
        rate=dropout_rate,
    )

    mel_text_transformer = MelTextTransformer(
        encoder_prenet=PointWiseFFN(d_model=d_model, dff=dff_prenet),
        decoder_prenet=tf.keras.layers.Embedding(target_vocab_size, d_model),
        decoder_postnet=TextPostnet(target_vocab_size),
        encoder=encoder,
        decoder=decoder,
        start_token_index=start_token_index,
        end_token_index=end_token_index,
        mel_channels=mel_channels
    )

    return mel_text_transformer


def new_text_mel_transformer(start_vec,
                             stop_prob_index,
                             input_vocab_size,
                             mel_channels=80,
                             num_layers=1,
                             d_model=64,
                             num_heads=1,
                             dff=512,
                             dff_prenet=512,
                             max_position_encoding=10000,
                             postnet_conv_filters=32,
                             postnet_conv_layers=2,
                             postnet_kernel_size=5,
                             dropout_rate=0):

    encoder = Encoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=max_position_encoding,
        rate=dropout_rate,
    )

    decoder = Decoder(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        maximum_position_encoding=max_position_encoding,
        rate=dropout_rate,
    )

    speech_out_module = SpeechPostnet(mel_channels=mel_channels,
                                        conv_filters=postnet_conv_filters,
                                        conv_layers=postnet_conv_layers,
                                        kernel_size=postnet_kernel_size)

    mel_text_transformer = TextMelTransformer(
        encoder_prenet=tf.keras.layers.Embedding(input_vocab_size, d_model),
        decoder_prenet=ReluFeedForward(d_model=d_model, dff=dff_prenet),
        decoder_postnet=speech_out_module,
        encoder=encoder,
        decoder=decoder,
        start_vec=start_vec,
        stop_prob_index=stop_prob_index
    )

    return mel_text_transformer