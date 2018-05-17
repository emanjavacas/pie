
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


def pad_batch(emb, nwords):
    """
    Parameters
    ===========
    emb : torch.Tensor(batch * nwords x emb_dim)
    nwords : list(int), number of words per sentence

    Returns
    =======
    torch.Tensor(seq_len x batch x emb_dim) where:
        - seq_len = max(nwords)
        - batch = len(nwords)
    """
    # (emb_dim x batch * nwords)
    emb = emb.t()

    if isinstance(nwords, torch.Tensor):
        nwords = nwords.tolist()

    output = []
    last = 0
    maxlen = max(nwords)

    for sentlen in nwords:
        sentlen = sentlen - 1   # remove <eos>
        padding = (0, maxlen - sentlen)
        output.append(F.pad(emb[:, last:last+sentlen], padding))
        last = last + sentlen

    # (batch x emb_dim x max_nwords)
    output = torch.stack(output)
    # (emb_dim x batch x max_nwords) -> (max_nwords x batch x emb_dim)
    output = output.transpose(0, 1).transpose(0, 2)

    return output


def pad_flatten_batch(batch, nwords):
    """
    Parameters
    ===========
    batch : tensor(seq_len, batch, encoding_size), output of the encoder
    nwords : tensor(batch), lengths of the sequence (without padding) including
        <eos> symbols

    Returns
    ========
    tensor(nwords, encoding_size)
    """
    output = []
    for sent, sentlen in zip(batch.t(), nwords):
        output.extend(list(sent[:sentlen-1].chunk(sentlen-1)))  # remove <eos>

    return torch.cat(output, dim=0)
