import numpy as np
import tensorflow as tf

from utils.metrics import attention_score
from utils.spectrogram_ops import mel_lengths, phoneme_lengths

logger = tf.get_logger()
logger.setLevel('ERROR')


def duration_to_alignment_matrix(durations):
    starts = np.cumsum(np.append([0], durations[:-1]))
    tot_duration = np.sum(durations)
    pads = tot_duration - starts - durations
    alignments = [np.concatenate([np.zeros(starts[i]), np.ones(durations[i]), np.zeros(pads[i])]) for i in
                  range(len(durations))]
    return np.array(alignments)


def clean_attention(binary_attention, jump_threshold):
    phon_idx = 0
    clean_attn = np.zeros(binary_attention.shape)
    for i, av in enumerate(binary_attention):
        next_phon_idx = np.argmax(av)
        if abs(next_phon_idx - phon_idx) > jump_threshold:
            next_phon_idx = phon_idx
        phon_idx = next_phon_idx
        clean_attn[i, min(phon_idx, clean_attn.shape[1] - 1)] = 1
    return clean_attn


def fill_zeros(duration, take_from='next'):
    """ Fills zeros with one. Takes either from the next non-zero duration, or max."""
    for i in range(len(duration)):
        if i < (len(duration) - 1):
            if duration[i] == 0:
                if take_from == 'next':
                    next_avail = np.where(duration[i:] > 1)[0]
                    if len(next_avail) > 1:
                        next_avail = next_avail[0]
                elif take_from == 'max':
                    next_avail = np.argmax(duration[i:])
                if next_avail:
                    duration[i] = 1
                    duration[i + next_avail] -= 1
    return duration


def fix_attention_jumps(binary_attn, binary_score, mel_len, phon_len):
    """ Scans for jumps in attention and attempts to fix. If score decreases, a collapse
        is likely so it tries to relax the jump size.
        Lower jumps size is more accurate, but more prone to collapse.
    """
    clean_scores = []
    clean_attns = []
    for jumpth in [2, 3, 4, 5]:
        cl_at = clean_attention(binary_attention=binary_attn, jump_threshold=jumpth)
        clean_attns.append(cl_at)
        sclean_score = attention_score(att=tf.cast(cl_at[None, None,:,:], tf.float32),
                                       mel_len=mel_len,
                                       phon_len=phon_len,
                                       r=1)
        clean_scores.append(tf.reduce_mean(sclean_score))
    best_idx = np.argmax(clean_scores)
    best_score = clean_scores[best_idx]
    best_cleaned_attention = clean_attns[best_idx]
    while (binary_score > best_score) and (jumpth < 20):
        jumpth += 1
        best_cleaned_attention = clean_attention(binary_attention=binary_attn, jump_threshold=jumpth)
        best_score = attention_score(att=tf.cast(best_cleaned_attention[None, None,:,:], tf.float32),
                                       mel_len=mel_len,
                                       phon_len=phon_len,
                                       r=1)
        best_score = tf.reduce_mean(best_score)
    if binary_score > best_score:
        best_cleaned_attention = binary_attn
    return best_cleaned_attention


def binary_attention(attention_weights):
    attention_peak_per_phoneme = attention_weights.max(axis=1)
    binary_attn = (attention_weights.T == attention_peak_per_phoneme).astype(int).T
    check = np.sum(binary_attn, axis=1) != 1 # more than one max per row
    if sum(check) != 0: # set every entry after first to zero
        flt_row = np.where(check == True)[0]
        for row in flt_row:
            flt_col = np.where(binary_attn[row] == 1)[0]
            binary_attn[row][flt_col[1:]] = 0
    return binary_attn


def get_durations_from_alignment(batch_alignments, mels, phonemes, weighted=False, binary=False, fill_gaps=False,
                                 fix_jumps=False, fill_mode='max'):
    """
    
    :param batch_alignments: attention weights from autoregressive model.
    :param mels: mel spectrograms.
    :param phonemes: phoneme sequence.
    :param weighted: if True use weighted average of durations of heads, best head if False.
    :param binary: if True take maximum attention peak, sum if False.
    :param fill_gaps: if True fills zeros durations with ones.
    :param fix_jumps: if True, tries to scan alingments for attention jumps and interpolate.
    :param fill_mode: used only if fill_gaps is True. Is either 'max' or 'next'. Defines where to take the duration
        needed to fill the gap. Next takes it from the next non-zeros duration value, max from the sequence maximum.
    :return:
    """
    assert (binary is True) or (fix_jumps is False), 'Cannot fix jumps in non-binary attention.'
    # mel_len - 1 because we remove last timestep, which is end_vector. start vector is not predicted (or removed from GTA)
    mel_len = mel_lengths(mels, padding_value=0.) - 1 # [N]
    # phonemes contain start and end tokens (start will be removed later)
    phon_len = phoneme_lengths(phonemes) - 1
    jumpiness, peakiness, diag_measure = attention_score(att=batch_alignments, mel_len=mel_len, phon_len=phon_len, r=1)
    attn_scores = diag_measure + jumpiness + peakiness
    durations = []
    final_alignment = []
    for batch_num, al in enumerate(batch_alignments):
        unpad_mel_len = mel_len[batch_num]
        unpad_phon_len = phon_len[batch_num]
        unpad_alignments = al[:, :unpad_mel_len, 1:unpad_phon_len]  # first dim is heads
        scored_attention = unpad_alignments * attn_scores[batch_num][:, None, None]
        
        if weighted:
            ref_attention_weights = np.sum(scored_attention, axis=0)
        else:
            best_head = np.argmax(attn_scores[batch_num])
            ref_attention_weights = unpad_alignments[best_head]
        
        if binary:  # pick max attention for each mel time-step
            binary_attn = binary_attention(ref_attention_weights)
            binary_attn_score = attention_score(tf.cast(binary_attn, tf.float32)[None, None,:,:],
                                           mel_len=unpad_mel_len[None],
                                           phon_len=unpad_phon_len[None]-1,
                                           r=1)
            binary_score =  tf.reduce_mean(binary_attn_score)
            if fix_jumps:
                binary_attn = fix_attention_jumps(
                    binary_attn=binary_attn,
                    mel_len=unpad_mel_len[None],
                    phon_len=unpad_phon_len[None]-1,
                    binary_score=binary_score)
            integer_durations = binary_attn.sum(axis=0)
            # integer_durations = tf.reduce_sum(binary_attn, axis=0)
        
        else:  # takes actual attention values and normalizes to mel_len
            attention_durations = np.sum(ref_attention_weights, axis=0)
            normalized_durations = attention_durations * ((unpad_mel_len) / np.sum(attention_durations))
            integer_durations = np.round(normalized_durations)
            tot_duration = np.sum(integer_durations)
            duration_diff = tot_duration - (unpad_mel_len)
            while duration_diff != 0:
                rounding_diff = integer_durations - normalized_durations
                if duration_diff > 0:  # duration is too long -> reduce highest (positive) rounding difference
                    max_error_idx = np.argmax(rounding_diff)
                    integer_durations[max_error_idx] -= 1
                elif duration_diff < 0:  # duration is too short -> increase lowest (negative) rounding difference
                    min_error_idx = np.argmin(rounding_diff)
                    integer_durations[min_error_idx] += 1
                tot_duration = np.sum(integer_durations)
                duration_diff = tot_duration - (unpad_mel_len)
        
        if fill_gaps:  # fill zeros durations
            integer_durations = fill_zeros(integer_durations, take_from=fill_mode)
        
        assert np.sum(integer_durations) == mel_len[batch_num], f'{np.sum(integer_durations)} vs {mel_len[batch_num]}'
        new_alignment = duration_to_alignment_matrix(integer_durations.astype(int))
        best_head = np.argmax(attn_scores[batch_num])
        best_attention = unpad_alignments[best_head]
        final_alignment.append(best_attention.T + new_alignment)
        durations.append(integer_durations)
    return durations, final_alignment, jumpiness, peakiness, diag_measure
