
import torch


class Beam(object):
    """
    Beam class for performing beam search

    Parameters
    -----------
    width: int, beam buffer size. The higher the better the chances are of
        actually decoding the best sequence but also the bigger the memory
        footprint
    prev: int, integer token to use as first decoding step
    eos: int or None, integer corresponding to the <eos> symbol in the
        vocabulary. It will be used as terminating criterion for the decoding
    """
    def __init__(self, width, eos, bos=None, device='cpu'):
        # attributes
        self.width = width
        self.eos = eos
        self.bos = bos or eos

        # data
        self.active = True
        self.scores = torch.zeros(width)
        init_state = torch.zeros(width, dtype=torch.int64, device=device) + self.bos
        # output values at each beam
        self.beam_values = [init_state]
        # backpointer to previous beam
        self.source_beams = []

    def __len__(self):
        """
        number of steps already decoded
        """
        return len(self.source_beams)

    def _new_beam(self, outs):
        """
        Computes a new beam based on the current model output and the history.

        outs: (width x vocab)
        """
        if len(self) > 0:
            beam_outs = outs + self.scores.unsqueeze(1).expand_as(outs)
            # EOS nihilation (adapted from OpenNMT)
            for i in range(len(beam_outs)):
                if self.beam_values[-1][i] == self.eos:
                    beam_outs[i] = -1e20
        else:
            # all beams have same start values, just pick the 1st
            beam_outs = outs[0]

        width, vocab = outs.size()
        # compute best outputs over a flatten vector of size (width x vocab)
        # i.e. regardless their source beam
        scores, flatten_ids = beam_outs.view(-1).topk(self.width, dim=0)
        # compute source beam, best candidates (can't do // on tensors)
        source_beams, beam = flatten_ids / vocab, flatten_ids % vocab

        return scores, source_beams, beam

    def get_source_beam(self, step=-1):
        """
        Get entries in previous beam leading to a given step.
        """
        if step >= len(self.source_beams):
            raise ValueError("Only {} decoded steps".format(len(self.source_beams)))

        return self.source_beams[step]

    def get_current_state(self):
        """
        Get beam at current step
        """
        return self.beam_values[-1]

    def finished(self, beam):
        """
        Finished criterion based on whether the last best hypothesis is EOS
        """
        return beam[0] == self.eos

    def advance(self, outs):
        """
        Runs a decoder step accumulating the path and the ids.
        """
        scores, source_beams, beam = self._new_beam(outs)

        if self.finished(beam):
            self.active = False

        self.scores = scores
        self.source_beams.append(source_beams)
        self.beam_values.append(beam)

    def get_hypothesis(self, idx):
        """
        Get hypothesis for `idx` entry in the current beam step.
        Note that the beam isn't mantained in sorted order.
        """
        if idx > self.width:
            raise ValueError("Beam has only capacity {}".format(self.width))

        hypothesis = []
        for step in range(len(self) - 1, -1, -1):
            hypothesis.append(self.beam_values[step+1][idx].item())
            idx = self.get_source_beam(step=step)[idx]
        return hypothesis[::-1]

    def decode(self, n=1):
        """
        Get n best hypothesis at current step.
        """
        if n > self.width:
            raise ValueError("Beam has only capacity {}".format(self.width))

        scores, beam_ids = torch.sort(self.scores, dim=0, descending=True)
        best_scores, best_beam_ids = scores[:n].tolist(), beam_ids[:n].tolist()
        best_hyps = [self.get_hypothesis(b) for b in best_beam_ids]
        return best_scores, best_hyps
