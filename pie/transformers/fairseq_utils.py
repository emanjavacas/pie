import torch
from typing import List, Tuple
from collections import Counter, namedtuple


BepConfig = namedtuple("BepConfig", ("start_token_id", "start_token_str", "end_token_id", "end_token_str"))

Roberta = BepConfig(0, "<s>", 2, "</s>")
Camembert = BepConfig(5, "<s>", 6, "</s>")


def extract_features_aligned_to_words(bert, sentence: List[str], return_all_hiddens: bool = False,
                                      bep_config: BepConfig = Roberta) -> torch.Tensor:
    """Extract RoBERTa features, aligned to spaCy's word-level tokenizer."""
    tokens = sentence
    # tokenize both with GPT-2 BPE and spaCy
    bpe_toks = bert.encode(" ".join(tokens))

    tokens_with_space = \
        [tok + " " for tok in tokens[:-1]] + \
        [tokens[-1]] + \
        [bep_config.end_token_str]

    alignment = align_bpe_to_words(bert, bpe_toks, tokens_with_space, bep_config=bep_config)

    # extract features and align them
    features = bert.extract_features(bpe_toks, return_all_hiddens=return_all_hiddens)
    features = features.squeeze(0)
    aligned_feats = align_features_to_words(bert, features, alignment, bep_config=bep_config)

    return aligned_feats[1:-1]


def clean(text):
    return text.strip()


def align_bpe_to_words(roberta, bpe_tokens: torch.LongTensor, other_tokens: List[str],
                       bep_config: BepConfig):
    """
    Helper to align GPT-2 BPE to other tokenization formats (e.g., spaCy).

    Args:
        roberta (RobertaHubInterface): RoBERTa instance
        bpe_tokens (torch.LongTensor): GPT-2 BPE tokens of shape `(T_bpe)`
        other_tokens (List[str]): other tokens of shape `(T_words)`
        start_token (Tuple[str,int]): Token that is used by the model as a start token
        end_token (Tuple[str,int]): Token that is used by the model as a end token

    Returns:
        List[str]: mapping from *other_tokens* to corresponding *bpe_tokens*.
    """
    assert bpe_tokens.dim() == 1
    assert bpe_tokens[0] == bep_config.start_token_id

    # remove whitespaces to simplify alignment
    bpe_tokens = [roberta.task.source_dictionary.string([x]) for x in bpe_tokens]
    bpe_tokens = [clean(roberta.bpe.decode(x) if x not in {'<s>', ''} else x) for x in bpe_tokens]
    other_tokens = [clean(str(o)) for o in other_tokens]

    # strip leading <s>
    bpe_tokens = bpe_tokens[1:]
    assert ''.join(bpe_tokens) == ''.join(other_tokens)

    # create alignment from every word to a list of BPE tokens
    alignment = []
    bpe_toks = filter(lambda item: item[1] != '', enumerate(bpe_tokens, start=1))
    j, bpe_tok = next(bpe_toks)
    for other_tok in other_tokens:
        bpe_indices = []
        while True:
            if other_tok.startswith(bpe_tok):
                bpe_indices.append(j)
                other_tok = other_tok[len(bpe_tok):]
                try:
                    j, bpe_tok = next(bpe_toks)
                except StopIteration:
                    j, bpe_tok = None, None
            elif bpe_tok.startswith(other_tok):
                # other_tok spans multiple BPE tokens
                bpe_indices.append(j)
                bpe_tok = bpe_tok[len(other_tok):]
                other_tok = ''
            else:
                raise Exception('Cannot align "{}" and "{}"'.format(other_tok, bpe_tok))
            if other_tok == '':
                break
        assert len(bpe_indices) > 0
        alignment.append(bpe_indices)
    assert len(alignment) == len(other_tokens)

    return alignment


def align_features_to_words(roberta, features, alignment, bep_config: BepConfig):
    """
    Align given features to words.

    Args:
        roberta (RobertaHubInterface): RoBERTa instance
        features (torch.Tensor): features to align of shape `(T_bpe x C)`
        alignment: alignment between BPE tokens and words returned by
            func:`align_bpe_to_words`.
    """
    assert features.dim() == 2

    bpe_counts = Counter(j for bpe_indices in alignment for j in bpe_indices)
    print(alignment, bpe_counts, bep_config)
    assert bpe_counts[0] == 0  # <s> shouldn't be aligned
    denom = features.new([bpe_counts.get(j, 1) for j in range(len(features))])
    weighted_features = features / denom.unsqueeze(-1)

    output = [weighted_features[0]]
    largest_j = -1
    for bpe_indices in alignment:
        output.append(weighted_features[bpe_indices].sum(dim=0))
        largest_j = max(largest_j, *bpe_indices)
    for j in range(largest_j + 1, len(features)):
        output.append(weighted_features[j])
    output = torch.stack(output)
    assert torch.all(torch.abs(output.sum(dim=0) - features.sum(dim=0)) < 1e-4)
    return output
