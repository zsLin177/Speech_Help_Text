import imp
import torch
import torch.nn.functional as F

from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.ctc import CTC
from wenet.transformer.encoder import TransformerEncoder

from wenet.utils.cmvn import load_cmvn
from wenet.utils.common import remove_duplicates_and_blank
from wenet.utils.mask import make_pad_mask
from torch.nn.utils.rnn import pad_sequence

class CTCModel(torch.nn.Module):
    '''
    Only use ctc
    '''
    def __init__(
        self,
        alphabet,
        configs
    ):
        super().__init__()
        self.alph = alphabet
        self.blank_id = alphabet.blank_id
        self.unk_id = alphabet.unk_id
        self.pad_id = alphabet.pad_id
        self.vocab_size = alphabet.size()
        
        self.configs = configs

        input_dim = configs['input_dim']
        if configs['cmvn_file'] is not None:
            mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
            global_cmvn = GlobalCMVN(
                torch.from_numpy(mean).float(),
                torch.from_numpy(istd).float())
        else:
            global_cmvn = None
        encoder = TransformerEncoder(input_dim,
                                     global_cmvn=global_cmvn,
                                     **configs['encoder_conf'])

        ctc = CTC(self.vocab_size, encoder.output_size(), blank_id=self.blank_id)
        self.encoder = encoder
        self.ctc = ctc

    def forward(self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor):
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        loss_ctc = self.ctc(encoder_out, encoder_out_lens, text,
                                text_lengths)
        loss = loss_ctc
        return loss, None, loss_ctc
    
    def loss(self, encoder_out, encoder_out_lens, text, text_lengths):
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, text,
                                text_lengths)
        return loss_ctc

    def _forward_encoder(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ):
        # Let's assume B = batch_size
        # 1. Encoder
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(
                speech,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        else:
            encoder_out, encoder_mask = self.encoder(
                speech,
                speech_lengths,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        return encoder_out, encoder_mask

    def n_ctc_greedy_search(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ):
        assert decoding_chunk_size != 0
        batch_size = encoder_out.shape[0]
        # Let's assume B = batch_size
        maxlen = encoder_out.size(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (B, maxlen, vocab_size)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
        mask = make_pad_mask(encoder_out_lens, maxlen)  # (B, maxlen)
        topk_index = topk_index.masked_fill_(mask, self.ctc.ctc_lo.out_features-1)  # (B, maxlen)
        hyps = [hyp.tolist() for hyp in topk_index]
        scores = topk_prob.max(1)
        hyps = [remove_duplicates_and_blank(hyp, self.blank_id) for hyp in hyps]
        return hyps, scores

    def ctc_greedy_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ):
        """ Apply CTC greedy search

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
        Returns:
            List[List[int]]: best path result
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]
        # Let's assume B = batch_size
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (B, maxlen, vocab_size)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
        mask = make_pad_mask(encoder_out_lens, maxlen)  # (B, maxlen)
        topk_index = topk_index.masked_fill_(mask, self.pad_id)  # (B, maxlen)
        hyps = [hyp.tolist() for hyp in topk_index]
        scores = topk_prob.max(1)
        hyps = [remove_duplicates_and_blank(hyp, self.blank_id) for hyp in hyps]
        return hyps, scores

    def batchify(self, batch):
        guids = [ele.guid for ele in batch]

        ## Audio Feature
        if batch[0].audio_feature is not None:
            audio_features = [ele.audio_feature for ele in batch]
            audio_feature_lengths = torch.tensor([ele.audio_feature_length for ele in batch], dtype=torch.int)
            audio_features = pad_sequence(audio_features, batch_first=True, padding_value=0)
            max_len = audio_feature_lengths.max().item()
            audio_feature_masks = ~(make_pad_mask(audio_feature_lengths, max_len))
            if self.configs['use_gpu']:
                audio_features = audio_features.cuda()
                audio_feature_masks = audio_feature_masks.cuda()
                audio_feature_lengths = audio_feature_lengths.cuda()

            # padded_audio_features = []
            # padded_audio_feature_masks = []
            # audio_feature_lengths = [ele.audio_feature_length for ele in batch]
            # max_feature_length = self.configs['max_audio_length']

            # for ele in batch:
            #     padding_feature_len = max_feature_length - ele.audio_feature_length
            #     padded_audio_features.append(
            #         F.pad(ele.audio_feature, pad=(0, 0, 0, padding_feature_len), value=0.0).unsqueeze(0))
            #     padded_audio_feature_masks.append([1] * ele.audio_feature_length + [0] * padding_feature_len)
            # audio_features = torch.cat(padded_audio_features, dim=0)
            # audio_feature_masks = torch.IntTensor(padded_audio_feature_masks) > 0
            # audio_feature_lengths = torch.IntTensor(audio_feature_lengths)
            # if self.configs['use_gpu']:
            #     audio_features = audio_features.cuda()
            #     audio_feature_masks = audio_feature_masks.cuda()
            #     audio_feature_lengths = audio_feature_lengths.cuda()
        else:
            audio_features, audio_feature_masks, audio_feature_lengths = None, None, None
        
        char_emb_ids = torch.LongTensor([ele.char_emb_ids for ele in batch])
        char_lens = torch.LongTensor([ele.char_len for ele in batch])
        char_emb_mask = torch.FloatTensor([ele.char_emb_mask for ele in batch])

        if self.configs['use_gpu']:
            char_emb_ids = char_emb_ids.cuda()
            char_lens = char_lens.cuda()
            char_emb_mask = char_emb_mask.cuda()

        return {"guids": guids, "audio_features": audio_features,
                        "audio_feature_masks": audio_feature_masks, "audio_feature_lengths": audio_feature_lengths,
                        "char_emb_ids": char_emb_ids, "char_lens": char_lens, "char_emb_mask": char_emb_mask}

        