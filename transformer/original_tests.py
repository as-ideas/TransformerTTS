# TEST
# pos_encoding = positional_encoding(50, 512)
# print('pos_encoding.shape: ', pos_encoding.shape)
# TEST
# x = tf.random.uniform((1, 3))
# temp = create_look_ahead_mask(x.shape[1])
# TEST
# temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
# y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
# out, attn = temp_mha(y, k=y, q=y, mask=None)
# out.shape, attn.shape
# TEST
# sample_ffn = point_wise_feed_forward_network(512, 2048)
# sample_ffn(tf.random.uniform((64, 50, 512))).shape
# TEST
# sample_encoder = Encoder(num_layers=2, d_model=512, num_heads=8, dff=2048, input_vocab_size=8500, maximum_position_encoding=10000)
# temp_input = tf.random.uniform((64, 62), dtype=tf.int64, minval=0, maxval=200)
#
# sample_encoder_output = sample_encoder(temp_input, training=False, mask=None)
# print(sample_encoder_output.shape)  # (batch_size, input_seq_len, d_model)
# TEST
# sample_decoder_layer = DecoderLayer(512, 8, 2048)
#
# sample_decoder_layer_output, _, _ = sample_decoder_layer(
#     tf.random.uniform((64, 50, 512)), sample_encoder_layer_output, False, None, None
# )
#
# sample_decoder_layer_output.shape  # (batch_size, target_seq_len, d_model)
# TEST
# sample_encoder_layer = EncoderLayer(512, 8, 2048)
#
# sample_encoder_layer_output = sample_encoder_layer(tf.random.uniform((64, 43, 512)), False, None)
#
# # TEST
# sample_encoder_layer_output.shape  # (batch_size, input_seq_len, d_model)
# TEST
# fn_out.shape  # (batch_size, tar_seq_len, target_vocab_size)
# TEST
# output.shape, attn['decoder_layer2_block2'].shape
# scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
# attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
#
# sample_decoder = Decoder(num_layers=2, d_model=512, num_heads=8, dff=2048, target_vocab_size=8000, maximum_position_encoding=5000)
# temp_input = tf.random.uniform((64, 26), dtype=tf.int64, minval=0, maxval=200)
#
# output, attn = sample_decoder(
#     temp_input, enc_output=sample_encoder_output, training=False, look_ahead_mask=None, padding_mask=None
# )
# temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
# temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)
#
# fn_out, _ = sample_transformer(
#     temp_input, temp_target, training=False, enc_padding_mask=None, look_ahead_mask=None, dec_padding_mask=None
# )
# checkpoint_path = "./checkpoints/train"
#
# ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
#
# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
# if ckpt_manager.latest_checkpoint:
#     ckpt.restore(ckpt_manager.latest_checkpoint)
#     print('Latest checkpoint restored!!')
