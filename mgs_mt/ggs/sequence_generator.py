import math
import torch

class SequenceGenerator(object):
    def __init__(self, tgt_dict, bos_token, max_decoding_length=1024, unk_penalty=0.0, return_bos=True):
        self.bos = bos_token
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.max_decoding_length = max_decoding_length - 1 # exclude the EOS marker
        self.unk_penalty = unk_penalty
        self.return_bos = return_bos

    @torch.no_grad()
    def greedy(self, model, sample):
        model.eval()

        # Encode
        encoder_out = model.encoder(
            src_tokens=sample['net_input']["src_tokens"],
            src_lengths=sample['net_input']["src_lengths"],
        )
        input_size = sample['net_input']['src_tokens'].size()
        bsz, src_len = input_size[0], input_size[1]

        # Decode
        tokens = (
            torch.zeros(bsz, self.max_decoding_length + 2, dtype=torch.long)
            .to(sample['net_input']['src_tokens'].device)
            .fill_(self.pad)
        )
        tokens[:, 0] = self.bos
        finished = torch.zeros(bsz).to(tokens.device).bool()
        states = {}
        for step in range(self.max_decoding_length + 1):  # +1 for EOS
            lprobs = self._forward_one(
                model,
                encoder_out,
                tokens[:, :step+1],
                incremental_states=states,
                return_logits=False
            )
            self._hacks(lprobs, step)
            pred_tok = lprobs.argmax(dim=1)

            tokens[:, step+1][~finished] = pred_tok[~finished]
            finished = finished | pred_tok.eq(self.eos)
            if (~finished).sum() == 0:
                tokens = tokens[:, :step+1]
                break

        if not self.return_bos:
            tokens = tokens[:, 1:]

        return tokens

    def _forward_one(
           self, model, encoder_out, tokens, incremental_states=None, return_logits=False, **decoder_kwargs
    ):
        if incremental_states is not None:
            decoder_out = model.decoder(
                tokens,
                encoder_out=encoder_out,
                incremental_state=incremental_states,
                **decoder_kwargs
            )
        else:
            decoder_out = model.decoder(tokens, encoder_out=encoder_out, **decoder_kwargs)
        decoder_out = list(decoder_out)
        if return_logits:
            logits_t = decoder_out[0][:, -1, :]
            return logits_t
        log_probs = model.get_normalized_probs(decoder_out, log_probs=True)
        log_probs = log_probs[:, -1, :]
        return log_probs

    def _hacks(self, lprobs, step):
        lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)  # nan

