import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from model.transformer_utils import create_mel_padding_mask, create_encoder_padding_mask


# for displaying only
def map_to_padded_length(inputs):
    duration, total_length, poss = inputs
    pos_zeros = tf.zeros(poss)
    markers = tf.ones(duration)
    try:
        padding = tf.zeros(total_length - (duration + poss))
    except Exception as e:
        print(e)
        print(total_length, duration, poss)
        return
    attention_weights = tf.concat([pos_zeros, markers, padding], axis=0)
    return attention_weights


# for displaying only
def duration_to_alignment_matrix(filled_durations):
    durs = tf.constant(filled_durations, dtype=tf.int32)[tf.newaxis, :]
    totlen = tf.math.reduce_sum(durs) * tf.ones(tf.shape(durs)[1], dtype=tf.int32)[tf.newaxis, :]
    pos = tf.math.cumsum(tf.concat([[0], durs[-1][:-1]], axis=0))[tf.newaxis, :]
    ctc = tf.transpose(tf.concat([durs, totlen, pos], axis=0))
    return tf.map_fn(map_to_padded_length, ctc, dtype=tf.float32).numpy()


def clean_attention(binary_attention, jump_threshold):
    phon_idx = 0
    clean_attn = np.zeros(binary_attention.shape)
    for i, av in enumerate(binary_attention):
        next_phon_idx = np.where(av == av.max())[0]
        if abs(next_phon_idx - phon_idx) > jump_threshold:
            next_phon_idx = phon_idx
        phon_idx = next_phon_idx
        clean_attn[i, min(phon_idx, clean_attn.shape[0] - 1)] = 1
    return clean_attn


def weight_mask(attention_weights):
    max_m, max_n = attention_weights.shape
    I = np.tile(np.arange(max_n), (max_m, 1)) / max_n
    J = np.swapaxes(np.tile(np.arange(max_m), (max_n, 1)), 0, 1) / max_m
    return np.sqrt(np.square(I - J))


def fill_zeros(duration):
    for i in range(len(duration)):
        if i < (len(duration) - 1):
            if duration[i] == 0:
                next_avail = np.where(duration[i:] > 1)[0]
                if len(next_avail) > 1:
                    next_avail = next_avail[0]
                if next_avail:
                    duration[i] = 1
                    duration[i + next_avail] -= 1
    return duration


def compute_cleanings(binary_attn, alignments_weights, binary_score):
    clean_scores = []
    clean_attns = []
    for jumpth in [2, 3, 4, 5]:
        cl_at = clean_attention(binary_attention=binary_attn, jump_threshold=jumpth)
        clean_attns.append(cl_at)
        sclean_score = np.sum(alignments_weights * cl_at)
        clean_scores.append(sclean_score)
    best_idx = np.argmin(clean_scores)
    best_score = clean_scores[best_idx]
    best_cleaned_attention = clean_attns[best_idx]
    while ((best_score - binary_score) > 2.) and (jumpth < 11):
        jumpth += 1
        best_cleaned_attention = clean_attention(binary_attention=binary_attn, jump_threshold=jumpth)
        best_score = np.sum(alignments_weights * best_cleaned_attention)
    phoneme_duration = best_cleaned_attention.sum(axis=0)
    filled_durations = fill_zeros(phoneme_duration)  # remove zero durations
    return filled_durations, best_cleaned_attention, best_score, jumpth


def binary_attention(attention_weights):
    attention_peak_per_phoneme = attention_weights.max(axis=1)
    binary_attn = (attention_weights.T == attention_peak_per_phoneme).astype(int).T
    binary_score = np.sum(attention_weights * binary_attn)
    return binary_attn, binary_score


def get_durations_from_alignment(batch_alignments, mels, phonemes, weighted=True, PLOT_OUTLIERS=False, PLOT_ALL=False):
    mel_pad_mask = create_mel_padding_mask(mels)
    phon_pad_mask = create_encoder_padding_mask(phonemes)
    durations = []
    # remove start end token or vector
    unpad_mels = []
    unpad_phonemes = []
    n_heads = batch_alignments.shape[1]
    for i, al in enumerate(batch_alignments):
        mel_len = int(mel_pad_mask[i].shape[-1] - np.sum(mel_pad_mask[i]))
        phon_len = int(phon_pad_mask[i].shape[-1] - np.sum(phon_pad_mask[i]))
        unpad_alignments = al[:, 1:mel_len - 1, 1:phon_len - 1]  # first dim is heads
        unpad_mels.append(mels[i, 1:mel_len - 1, :])
        unpad_phonemes.append(phonemes[i, 1:phon_len - 1])
        alignments_weights = weight_mask(unpad_alignments[0])
        heads_scores = []
        scored_attention = []
        for _, attention_weights in enumerate(unpad_alignments):
            score = np.sum(alignments_weights * attention_weights)
            scored_attention.append(attention_weights / score)
            heads_scores.append(score)
        
        # weighted attention
        if weighted:
            ref_attention_weights = np.sum(scored_attention, axis=0)
            binary_attn, binary_score = binary_attention(ref_attention_weights)
            filled_durations, cleaned_attention, clean_score, jumpth = compute_cleanings(
                binary_attn=binary_attn,
                alignments_weights=alignments_weights,
                binary_score=binary_score)
        
        # best attention head
        else:
            best_head = np.argmin(heads_scores)
            ref_attention_weights = unpad_alignments[best_head]
            binary_attn, binary_score = binary_attention(ref_attention_weights)
            filled_durations, cleaned_attention, clean_score, jumpth = compute_cleanings(
                binary_attn=binary_attn,
                alignments_weights=alignments_weights,
                binary_score=binary_score)
        
        if (jumpth > 5 and PLOT_OUTLIERS) or PLOT_ALL:
            fig = plt.figure(figsize=(20, 5))
            plt.subplots_adjust(hspace=.0, wspace=.0)
            plt.suptitle(f'Score difference {clean_score - binary_score}. Reference is weighted: {weighted}')
            for h, attention_weights in enumerate(unpad_alignments):
                plt.subplot(f'3{n_heads}{h + 1}')
                plt.title(f'Attention head {h}')
                plt.imshow(attention_weights.T)
            plt.subplot(3, n_heads, n_heads + 1)
            plt.title(f'Reference attention')
            plt.imshow(ref_attention_weights.T)
            plt.subplot(3, n_heads, n_heads + 2)
            plt.title(f'binary attention')
            plt.imshow(binary_attn.T)
            plt.subplot(3, n_heads, n_heads + 3)
            plt.title(f'cleaned attention')
            plt.imshow(cleaned_attention.T)
            new_alignment = duration_to_alignment_matrix(filled_durations)
            plt.subplot(3, n_heads, n_heads + 4)
            plt.title(f'FINAL attention')
            plt.imshow(new_alignment)
            plt.subplot(3, n_heads, n_heads + 5)
            plt.title(f'FINAL OVERLAPPED attention')
            plt.imshow(new_alignment + ref_attention_weights.T)
            for ax in fig.axes:
                ax.axis("off")
            plt.tight_layout()
            plt.show()
        
        assert np.sum(filled_durations) == mel_len - 2
        durations.append(filled_durations)
    return durations, unpad_mels, unpad_phonemes