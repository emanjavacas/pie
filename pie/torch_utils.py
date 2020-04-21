
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


def init_hidden_for(inp, num_dirs, num_layers, hid_dim, cell,
                    h_0=None, add_init_jitter=False):
    """
    General function for initializing RNN hidden states

    Parameters:
    - inp: torch.Tensor(seq_len, batch_size, dim)
    """
    size = (num_dirs * num_layers, inp.size(1), hid_dim)

    # create h_0
    if h_0 is not None:
        h_0 = h_0.repeat(1, inp.size(1), 1)
    else:
        h_0 = torch.zeros(*size, device=inp.device)

    # eventualy add jitter
    if add_init_jitter:
        h_0 = h_0 + torch.normal(torch.zeros_like(h_0), 0.3)

    if cell.startswith('LSTM'):
        # compute memory cell
        return h_0, torch.zeros_like(h_0)
    else:
        return h_0


def pack_sort(inp, lengths, batch_first=False):
    """
    Transform input into PaddedSequence sorting batch by length (as required).
    Also return an index variable that unsorts the output back to the original
    order.

    Parameters:
    -----------
    inp: torch.Tensor(seq_len x batch x dim)
    lengths: LongTensor of length ``batch``

    >>> from torch.nn.utils.rnn import pad_packed_sequence as unpack
    >>> inp = torch.tensor([[1, 3], [2, 4], [0, 5]], dtype=torch.float)
    >>> lengths = torch.tensor([2, 3]) # unsorted order
    >>> sorted_inp, unsort = pack_sort(inp, lengths)
    >>> sorted_inp, _ = unpack(sorted_inp)
    >>> sorted_inp[:, unsort].tolist()  # original order
    [[1.0, 3.0], [2.0, 4.0], [0.0, 5.0]]
    >>> sorted_inp.tolist()  # sorted by length
    [[3.0, 1.0], [4.0, 2.0], [5.0, 0.0]]
    """
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths)  # no need to use gpu

    lengths, sort = torch.sort(lengths, descending=True)
    _, unsort = sort.sort()

    if batch_first:
        inp = pack_padded_sequence(inp[sort], lengths.tolist())
    else:
        inp = pack_padded_sequence(inp[:, sort], lengths.tolist())

    return inp, unsort


def get_last_token(t, lengths):
    """
    Grab last hidden activation of each batch element according to `lengths`

    #                               ^ (1)      ^ (2)      ^ (3)
    >>> t = torch.tensor([[[1],[2],[3]], [[2],[3],[4]], [[3],[4],[5]]])
    >>> lengths = torch.tensor([3, 2, 1])
    >>> get_last_token(t, lengths).tolist()
    [[3], [3], [3]]
    """
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths)

    seq_len, batch, _ = t.size()
    index = torch.arange(0, batch, dtype=torch.int64, device=t.device) * seq_len
    index = index + (lengths - 1)
    t = t.transpose(0, 1).contiguous()  # make it batch first
    t = t.view(seq_len * batch, -1)
    t = t.index_select(0, index)
    t = t.view(batch, -1)
    return t


def pad_flat_batch(emb, nwords, maxlen=None):
    """
    Transform a 2D flat batch (batch of words in multiple sentences) into a 3D
    padded batch where words have been allocated to their respective sentence
    according to user passed sentence lengths `nwords`

    Parameters
    ===========
    emb : torch.Tensor(total_words x emb_dim), flattened tensor of word embeddings
    nwords : torch.Tensor(batch), number of words per sentence

    Returns
    =======
    torch.Tensor(max_seq_len x batch x emb_dim) where:
        - max_seq_len = max(nwords)
        - batch = len(nwords)

    >>> emb = [[0], [1], [2], [3], [4], [5]]
    >>> nwords = [3, 1, 2]
    >>> pad_flat_batch(torch.tensor(emb), torch.tensor(nwords)).tolist()
    [[[0], [3], [4]], [[1], [0], [5]], [[2], [0], [0]]]
    """
    if len(emb) != sum(nwords):
        raise ValueError("Got {} items but was asked to pad {}"
                         .format(len(emb), sum(nwords).item()))

    output, last = [], 0
    maxlen = maxlen or max(nwords).item()

    for sentlen in nwords.tolist():
        padding = (0, 0, 0, maxlen - sentlen)
        output.append(F.pad(emb[last:last+sentlen], padding))
        last = last + sentlen

    # (seq_len x batch x emb_dim)
    output = torch.stack(output, dim=1)

    return output


def flatten_padded_batch(batch, nwords):
    """
    Inverse of pad_flat_batch

    Parameters
    ===========
    batch : tensor(seq_len, batch, encoding_size), output of the encoder
    nwords : tensor(batch), lengths of the sequence (without padding)

    Returns
    ========
    tensor(nwords, encoding_size)

    >>> batch = [[[0], [3], [4]], [[1], [0], [5]], [[2], [0], [0]]]
    >>> nwords = [3, 1, 2]
    >>> flatten_padded_batch(torch.tensor(batch), torch.tensor(nwords)).tolist()
    [[0], [1], [2], [3], [4], [5]]
    """
    with torch.no_grad():
        output = []
        for sent, sentlen in zip(batch.transpose(0, 1), nwords):
            output.extend(list(sent[:sentlen].chunk(sentlen)))  # remove <eos>

        return torch.cat(output, dim=0)


def pad_batch(batch, padding_id, device='cpu', return_lengths=True):
    """
    Pad batch into tensor
    """
    lengths = [len(example) for example in batch]
    maxlen, batch_size = max(lengths), len(batch)
    output = torch.zeros(
        maxlen, batch_size, device=device, dtype=torch.int64
    ) + padding_id

    for i, example in enumerate(batch):
        output[0:lengths[i], i].copy_(
            torch.tensor(example, dtype=torch.int64, device=device))

    if return_lengths:
        lengths = torch.tensor(lengths, dtype=torch.int64, device=device)
        return output, lengths

    return output


def pad(batch, pad=0, pos='pre'):
    """
    >>> batch = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]])
    >>> pad(batch, pad=-1, pos='pre').tolist()
    [[-1, -1], [1, 1], [2, 2], [3, 3], [4, 4]]
    >>> pad(batch, pad=5, pos='post').tolist()
    [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
    """
    if pos.lower() == 'pre':
        padding = (0, 0) * (batch.dim() - 1) + (1, 0)
    elif pos.lower() == 'post':
        padding = (0, 0) * (batch.dim() - 1) + (0, 1)
    else:
        raise ValueError("Unknown value for pos: {}".format(pos))

    return F.pad(batch, padding, value=pad)


def make_length_mask(lengths):
    """
    Compute binary length mask.

    lengths: torch.Tensor(batch, dtype=int) should be on the desired
        output device.

    Returns
    =======

    mask: torch.ByteTensor(batch x seq_len)
    """
    maxlen, batch = lengths.detach().max(), len(lengths)
    return torch.arange(0, maxlen, dtype=torch.int64, device=lengths.device) \
                .repeat(batch, 1) \
                .lt(lengths.unsqueeze(1))


def word_dropout(inp, p, training, encoder):
    """
    Drop input words based on frequency
    """
    # don't do anything during training
    if p == 0.0 or not training:
        return inp

    # (seq_len x batch)
    seq_len, batch = inp.size()
    # get batch of word frequencies
    mask = torch.tensor(
        [[encoder.freqs[encoder.inverse_table[inp[i, j].item()]]
          for j in range(batch)] for i in range(seq_len)]
    ).float().to(inp.device)
    # compute bernoulli mask
    # PyTorch 1.2 and 1.3 :
    #   - Deprecated masked_fill_ used with int.
    #   - Mask should be booleans in general
    mask = (1 - torch.bernoulli(mask / (p + mask))).bool()
    # don't drop padding
    mask.masked_fill_(inp.eq(encoder.get_pad()).bool(), 0)
    # set words to unknowns
    return inp.masked_fill(mask, encoder.get_unk())


def sequential_dropout(inp, p, training):
    if not training or not p:
        return inp

    mask = inp.new(1, inp.size(1), inp.size(2)).bernoulli_(1 - p)
    mask = mask / (1 - p)

    return inp * mask.expand_as(inp)


def log_sum_exp(x, dim=-1):
    """
    Numerically stable log_sum_exp

    Parameters
    ==========
    x : torch.tensor

    >>> import torch
    >>> x = torch.randn(10, 5)
    """
    max_score, _ = torch.max(x, dim)
    max_score_broadcast = max_score.unsqueeze(dim).expand_as(x)
    return max_score + (x - max_score_broadcast).exp().sum(dim).log()


def viterbi_decode(tag_sequence, transition):
    """
    Perform Viterbi decoding in log space over a sequence given a transition matrix
    specifying pairwise (transition) potentials between tags and a matrix of shape
    (sequence_length, num_tags) specifying unary potentials for possible tags per
    timestep.

    Parameters
    ==========
    tag_sequence: torch.Tensor, required.
        A tensor of shape (sequence_length, num_tags) representing scores for
        a set of tags over a given sequence.
    trans: torch.Tensor, required.
        A tensor of shape (num_tags, num_tags) representing the binary potentials
        for transitioning between a given pair of tags.

    Returns
    =======
    viterbi_path: The tag indices of the maximum likelihood tag sequence
    viterbi_score: float, The score of the viterbi path
    """
    seq_len, vocab = tag_sequence.size()

    path_scores = []
    path_indices = []

    path_scores.append(tag_sequence[0, :])

    # Evaluate the scores for all possible paths.
    for t in range(1, seq_len):

        # Add pairwise potentials to current scores.
        summed_potentials = path_scores[t - 1].unsqueeze(-1) + transition
        scores, paths = torch.max(summed_potentials, 0)
        path_scores.append(tag_sequence[t, :] + scores.squeeze())
        path_indices.append(paths.squeeze())

    # Construct the most likely sequence backwards.
    viterbi_score, best_path = torch.max(path_scores[-1].cpu(), 0)
    viterbi_path = [int(best_path.numpy())]
    for backward_t in reversed(path_indices):
        viterbi_path.append(int(backward_t[viterbi_path[-1]]))

    # Reverse the backward path.
    viterbi_path.reverse()
    
    return viterbi_path, viterbi_score
