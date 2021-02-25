import tensorflow as tf

from utils.metrics import attention_score
from utils.spectrogram_ops import mel_lengths, phoneme_lengths

logger = tf.get_logger()
logger.setLevel('ERROR')
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


def to_node_index(i, j, cols):
    return cols * i + j


def from_node_index(node_index, cols):
    return node_index // cols, node_index % cols


def to_adj_matrix(mat):
    rows = mat.shape[0]
    cols = mat.shape[1]
    
    row_ind = []
    col_ind = []
    data = []
    
    for i in range(rows):
        for j in range(cols):
            
            node = to_node_index(i, j, cols)
            
            if j < cols - 1:
                right_node = to_node_index(i, j + 1, cols)
                weight_right = mat[i, j + 1]
                row_ind.append(node)
                col_ind.append(right_node)
                data.append(weight_right)
            
            if i < rows - 1 and j < cols:
                bottom_node = to_node_index(i + 1, j, cols)
                weight_bottom = mat[i + 1, j]
                row_ind.append(node)
                col_ind.append(bottom_node)
                data.append(weight_bottom)
            
            if i < rows - 1 and j < cols - 1:
                bottom_right_node = to_node_index(i + 1, j + 1, cols)
                weight_bottom_right = mat[i + 1, j + 1]
                row_ind.append(node)
                col_ind.append(bottom_right_node)
                data.append(weight_bottom_right)
    
    adj_mat = coo_matrix((data, (row_ind, col_ind)), shape=(rows * cols, rows * cols))
    return adj_mat.tocsr()


def extract_durations_with_dijkstra(attention_map: np.array) -> np.array:
    """
    Extracts durations from the attention matrix by finding the shortest monotonic path from
    top left to bottom right.
    """
    attn_max = np.max(attention_map)
    path_probs = attn_max - attention_map
    adj_matrix = to_adj_matrix(path_probs)
    dist_matrix, predecessors = dijkstra(csgraph=adj_matrix, directed=True,
                                         indices=0, return_predecessors=True)
    path = []
    pr_index = predecessors[-1]
    while pr_index != 0:
        path.append(pr_index)
        pr_index = predecessors[pr_index]
    path.reverse()
    
    # append first and last node
    path = [0] + path + [dist_matrix.size - 1]
    cols = path_probs.shape[1]
    mel_text = {}
    durations = np.zeros(attention_map.shape[1], dtype=np.int32)
    
    # collect indices (mel, text) along the path
    for node_index in path:
        i, j = from_node_index(node_index, cols)
        mel_text[i] = j
    
    for j in mel_text.values():
        durations[j] += 1
    
    return durations


def duration_to_alignment_matrix(durations):
    starts = np.cumsum(np.append([0], durations[:-1]))
    tot_duration = np.sum(durations)
    pads = tot_duration - starts - durations
    alignments = [np.concatenate([np.zeros(starts[i]), np.ones(durations[i]), np.zeros(pads[i])]) for i in
                  range(len(durations))]
    return np.array(alignments)


def get_durations_from_alignment(batch_alignments, mels, phonemes, weighted=False):
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
    # mel_len - 1 because we remove last timestep, which is end_vector. start vector is not predicted (or removed from GTA)
    mel_len = mel_lengths(mels, padding_value=0.) - 1  # [N]
    # phonemes contain start and end tokens (start will be removed later)
    phon_len = phoneme_lengths(phonemes) - 1
    jumpiness, peakiness, diag_measure = attention_score(att=batch_alignments, mel_len=mel_len, phon_len=phon_len, r=1)
    attn_scores = diag_measure + jumpiness + peakiness
    durations = []
    final_alignment = []
    for batch_num, al in enumerate(batch_alignments):
        unpad_mel_len = mel_len[batch_num]
        unpad_phon_len = phon_len[batch_num]
        unpad_alignments = al[:, 1:unpad_mel_len, 1:unpad_phon_len]  # first dim is heads
        scored_attention = unpad_alignments * attn_scores[batch_num][:, None, None]
        
        if weighted:
            ref_attention_weights = np.sum(scored_attention, axis=0)
        else:
            best_head = np.argmax(attn_scores[batch_num])
            ref_attention_weights = unpad_alignments[best_head]
        integer_durations = extract_durations_with_dijkstra(ref_attention_weights)
        
        assert np.sum(integer_durations) == mel_len[batch_num]-1, f'{np.sum(integer_durations)} vs {mel_len[batch_num]-1}'
        new_alignment = duration_to_alignment_matrix(integer_durations.astype(int))
        best_head = np.argmax(attn_scores[batch_num])
        best_attention = unpad_alignments[best_head]
        final_alignment.append(best_attention.T + new_alignment)
        durations.append(integer_durations)
    return durations, final_alignment, jumpiness, peakiness, diag_measure
