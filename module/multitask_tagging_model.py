from transformers import BertModel
from .audio_encoder import AudioTransformerEncoder
import torch.nn.functional as F
from .crf import CRF
import torch.nn as nn
import torch
from ZEN import ZenModel
from module.attention_fusion_layer import ShortCutAttentionFusionLayer, GateAttentionFusionLayer
from .base_encoder_v2 import BaseEncoder
from .umt import CoupledCMT
from transformers import BertConfig
from wenet.transformer.asr_model import init_ctc_model
from wenet.transformer.cma import CTCAlignFusion
import pdb


class MultitaskTaggingModel(BaseEncoder):
    def __init__(self, args, data):
        super(MultitaskTaggingModel, self).__init__(args)
        if self.args.text_encoder == "ZEN":
            self.text_encoder = ZenModel.from_pretrained(args.zen_directory)
            print("building ZEN Encoder")
            self.name = "audio_ZEN"
        else:
            if self.args.text_encoder == "BERT":
                plm_directory = args.bert_directory
                self.name = "audio_BERT"
                print("building BERT Encoder")
            elif self.args.text_encoder == "WoBERT":
                plm_directory = args.wobert_directory
                self.name = "audio_WoBERT"
                print("building WoBERT Encoder")
            elif self.args.text_encoder == "MacBERT":
                plm_directory = args.macbert_directory
                self.name = "audio_MacBERT"
                print("building MacBERT Encoder")
            else:
                raise Exception("Unsupport encoder: %s" % (self.args.encoder))
            self.text_encoder = BertModel.from_pretrained(plm_directory)
        self.audio_encoder = AudioTransformerEncoder(self.args.num_mel_bins, d_model=self.args.audio_hidden_dim, n_blocks=self.args.n_blocks)
        # bert_config = BertConfig()
        # self.audio_encoder = AudioTransformerEncoder(self.args.num_mel_bins, d_model=self.args.audio_hidden_dim, n_heads=, d_ff=bert_config.intermediate_size, n_blocks=self.args.n_blocks)
        # self.dropout = nn.Dropout(args.dropout)

        # self.fusion_layer = GateAttentionFusionLayer(self.args)
        if self.args.fusion_layer == "UMT":
            self.fusion_layer = CoupledCMT(self.args)
            output_dim = self.text_encoder.config.hidden_size * 3  # + self.args.audio_hidden_dim
        else:
            self.fusion_layer = GateAttentionFusionLayer(self.args)
            output_dim = self.text_encoder.config.hidden_size * 2
        if args.ner_type == "Flat_NER":
            label_dim = data.flat_label_alphabet.size()
        else:
            label_dim = data.nested_label_alphabet.size()
        # output_dim = output_dim + 366
        self.hidden2tag = nn.Linear(output_dim, label_dim + 2)
        self.crf = CRF(label_dim, args.use_gpu, args.crf_reduction)
        if args.fix_embeddings:
            self.text_encoder.embeddings.word_embeddings.weight.requires_grad = False
            self.text_encoder.embeddings.position_embeddings.weight.requires_grad = False
            self.text_encoder.embeddings.token_type_embeddings.weight.requires_grad = False

        ## CTC loss
        self.audio_embedding = nn.Embedding(data.char_alphabet.size(), self.args.audio_hidden_dim)
        self.ctc = nn.CTCLoss(blank=data.char_alphabet.blank_id, reduction="mean" if args.ctc_reduction == "mean" else "sum")
        # if args.fix_audioEncoder:



    def forward(self, input):
        crf_input, _, _ = self.get_crf_input(input)
        _, best_path = self.crf.viterbi_decode(crf_input, input["label_mask"])
        return best_path

    def neg_log_likelihood(self, input, char_alphabet_pad_id=0):
        crf_input, audio_logit, audio_mask = self.get_crf_input(input)
        crf_loss = self.crf.neg_log_likelihood_loss(crf_input, input["label_mask"], input["label_ids"])
        # return crf_loss
        input_lengths = torch.sum(audio_mask, dim=-1)
        target_lengths = input["char_lens"]
        log_prob = F.log_softmax(audio_logit, dim=2).permute(1, 0, 2)
        # bsz = input["char_emb_ids"].size()[0]
        ctc_loss = self.ctc(log_prob, input["char_emb_ids"], input_lengths, target_lengths)
        return self.args.crf_coef * crf_loss + self.args.ctc_coef * ctc_loss
        # if self.args.batch_ctc_loss:
        #     assert self.args.ctc_reduction == "mean"
        #     return self.args.crf_coef * crf_loss + self.args.ctc_coef * ctc_loss * bsz
        # else:
        #     return self.args.crf_coef * crf_loss + self.args.ctc_coef * ctc_loss


    def get_crf_input(self, input):
        if self.args.text_encoder == "ZEN":
            textual_repr = self.text_encoder(input_ids=input["input_ids"], input_ngram_ids=input["ngram_ids"],
                                                   ngram_position_matrix=input["ngram_positions"],
                                                   attention_mask=input["input_masks"], output_all_encoded_layers=False)[0]
        else:
            textual_repr = self.text_encoder(input["input_ids"], attention_mask=input["input_masks"])[0]
        audio_repr, audio_mask = self.audio_encoder(input["audio_features"], input["audio_feature_masks"])
        # slzhou: logit for masked ctc
        audio_logit = torch.matmul(audio_repr, self.audio_embedding.weight.data.permute(1, 0))
        audio_prob = masked_softmax(audio_logit, input["char_emb_mask"], 2)
        char_emb_ids = input["char_emb_ids"].unsqueeze(1).expand(-1, audio_repr.size(1), -1)

        # time_series_feat = torch.gather(audio_prob, 2, char_emb_ids).permute(0, 2, 1)
        # # slzhou: [batch_size, seq_len, 366] the prob of each char align to each frame
        # time_series_feat = time_series_feat.masked_fill(~audio_mask.unsqueeze(1), 0)

        fusional_textual_repr = self.fusion_layer(textual_repr, input["input_masks"], audio_repr, audio_mask.float())
        # audio_emb_repr = self.audio_embedding(input["char_emb_ids"])
        # multimodal_repr = torch.cat((textual_repr, fusional_textual_repr, time_series_feat), dim=-1)
        multimodal_repr = torch.cat((textual_repr, fusional_textual_repr), dim=-1)
        hidden_repr = self.hidden2tag(multimodal_repr)
        return hidden_repr, audio_logit, audio_mask


def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)

def masked_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        vector = vector + (1.0 - mask) * -10000.0
    return F.softmax(vector, dim=dim)


class SpeechSeqtagModel(BaseEncoder):
    def __init__(self, args, data, config):
        super(SpeechSeqtagModel, self).__init__(args)
        if self.args.text_encoder == "BERT":
            plm_directory = args.bert_directory
            self.name = "audio_BERT"
            print("building BERT Encoder")
        else:
            raise Exception("Unexpected encoder: %s" % (self.args.encoder))
        self.text_encoder = BertModel.from_pretrained(plm_directory)
        self.audio_encoder = init_ctc_model(config)
        if self.args.audio_checkpoint != 'None':
            print('loading audio model from checkpoint %s' % self.args.audio_checkpoint)
            self.audio_encoder.load_state_dict(torch.load(self.args.audio_checkpoint))
        
        if self.args.fusion_layer == "UMT":
            self.fusion_layer = CoupledCMT(self.args)
            output_dim = self.text_encoder.config.hidden_size * 3  # + self.args.audio_hidden_dim
        else:
            self.fusion_layer = GateAttentionFusionLayer(self.args)
            output_dim = self.text_encoder.config.hidden_size * 2
        if args.ner_type == "Flat_NER":
            label_dim = data.flat_label_alphabet.size()
        else:
            label_dim = data.nested_label_alphabet.size()
        # output_dim = output_dim + 366
        self.hidden2tag = nn.Linear(output_dim, label_dim + 2)
        self.crf = CRF(label_dim, args.use_gpu, args.crf_reduction)
        if args.fix_embeddings:
            self.text_encoder.embeddings.word_embeddings.weight.requires_grad = False
            self.text_encoder.embeddings.position_embeddings.weight.requires_grad = False
            self.text_encoder.embeddings.token_type_embeddings.weight.requires_grad = False
    
    def get_crf_input(self, input):
        if self.args.text_encoder == "ZEN":
            textual_repr = self.text_encoder(input_ids=input["input_ids"], input_ngram_ids=input["ngram_ids"],
                                                   ngram_position_matrix=input["ngram_positions"],
                                                   attention_mask=input["input_masks"], output_all_encoded_layers=False)[0]
        else:
            textual_repr = self.text_encoder(input["input_ids"], attention_mask=input["input_masks"])[0]
        audio_repr, audio_mask = self.audio_encoder._forward_encoder(input["audio_features"], input["audio_feautre_lengths"])
        audio_mask = audio_mask.squeeze(1)
        encoder_out_lens = audio_mask.sum(1)

        fusional_textual_repr = self.fusion_layer(textual_repr, input["input_masks"], audio_repr, audio_mask.float())
        multimodal_repr = torch.cat((textual_repr, fusional_textual_repr), dim=-1)
        hidden_repr = self.hidden2tag(multimodal_repr)
        return hidden_repr, audio_repr, encoder_out_lens

    def forward(self, input):
        crf_input, _, _ = self.get_crf_input(input)
        _, best_path = self.crf.viterbi_decode(crf_input, input["label_mask"])
        return best_path
    
    def neg_log_likelihood(self, input, char_alphabet_pad_id):
        crf_input, audio_repr, audio_output_lengths = self.get_crf_input(input)
        crf_loss = self.crf.neg_log_likelihood_loss(crf_input, input["label_mask"], input["label_ids"])
        # return crf_loss
        target, target_lengths = input['char_emb_ids'], input['char_lens']
        target[range(target.shape[0]), target_lengths-1]=char_alphabet_pad_id
        target = target[:, 1:]
        target_lengths = target_lengths-2
        ctc_loss = self.audio_encoder.loss(audio_repr, audio_output_lengths, target, target_lengths)
        return self.args.crf_coef * crf_loss + self.args.ctc_coef * ctc_loss

    
class TokenizedAudioReprSeqtagModel(BaseEncoder):
    def __init__(self, args, data, config):
        super(TokenizedAudioReprSeqtagModel, self).__init__(args)
        self.audio_encoder = init_ctc_model(config)
        if self.args.ctc_checkpoint != 'None':
            print('loading ctc model from checkpoint %s' % self.args.ctc_checkpoint)
            self.audio_encoder.load_state_dict(torch.load(self.args.ctc_checkpoint))
        
        if args.ner_type == "Flat_NER":
            label_dim = data.flat_label_alphabet.size()
        else:
            label_dim = data.nested_label_alphabet.size()

        # audio crf
        # self.bos_eos_embed = nn.Embedding(2, self.audio_encoder.encoder._output_size)
        self.tokenized_sp_embed = CTCAlignFusion(self.audio_encoder.encoder._output_size, postlayer_type=args.postlayer_type, n_layers=args.postlayer)
        self.hidden2tag = nn.Linear(self.audio_encoder.encoder._output_size, label_dim+2)
        self.crf = CRF(label_dim, args.use_gpu, args.crf_reduction)

        # if args.fix_embeddings:
        #     self.text_encoder.embeddings.word_embeddings.weight.requires_grad = False
        #     self.text_encoder.embeddings.position_embeddings.weight.requires_grad = False
        #     self.text_encoder.embeddings.token_type_embeddings.weight.requires_grad = False
    
    def get_crf_input(self, input):
        audio_repr, audio_mask = self.audio_encoder._forward_encoder(input["audio_features"], input["audio_feautre_lengths"])
        audio_mask = audio_mask.squeeze(1)
        encoder_out_lens = audio_mask.sum(1)
        probs = self.audio_encoder.ctc.probs(audio_repr, self.args.tem)
        tokenized_sp_repr = self.tokenized_sp_embed.get_tokenized_sp_repr(audio_repr, input['char_emb_ids'], probs, ~(input["input_masks"].bool()))
        hidden_repr = self.hidden2tag(tokenized_sp_repr)
        return hidden_repr, audio_repr, encoder_out_lens, tokenized_sp_repr, audio_mask


    def forward(self, input):
        crf_input, _, _, tokenized_sp_repr, audio_mask = self.get_crf_input(input)
        _, best_path = self.crf.viterbi_decode(crf_input, input["label_mask"])
        return best_path
    
    def neg_log_likelihood(self, input, char_alphabet_pad_id):
        crf_input, audio_repr, audio_output_lengths, tokenized_sp_repr, audio_mask = self.get_crf_input(input)
        crf_loss = self.crf.neg_log_likelihood_loss(crf_input, input["label_mask"], input["label_ids"])
        # return crf_loss
        target, target_lengths = input['char_emb_ids'].clone(), input['char_lens'].clone()
        target[range(target.shape[0]), target_lengths-1]=char_alphabet_pad_id
        target = target[:, 1:]
        target_lengths = target_lengths-2
        ctc_loss = self.audio_encoder.loss(audio_repr, audio_output_lengths, target, target_lengths)
        return self.args.crf_coef * crf_loss + self.args.ctc_coef * ctc_loss
    
    def get_crfloss_and_repr(self, input):
        crf_input, audio_repr, audio_output_lengths, tokenized_sp_repr, audio_mask = self.get_crf_input(input)
        crf_loss = self.crf.neg_log_likelihood_loss(crf_input, input["label_mask"], input["label_ids"])
        return crf_loss, audio_repr, audio_output_lengths, tokenized_sp_repr, audio_mask


class JointTokenSeqtagModel(BaseEncoder):
    def __init__(self, args, data, config):
        super(JointTokenSeqtagModel, self).__init__(args)
        self.audio_model = TokenizedAudioReprSeqtagModel(args, data, config)
        if self.args.token_audio_checkpoint != 'None':
            print('loading tokenized audio model from checkpoint %s for TokenizedAudioReprSeqtagModel' % self.args.token_audio_checkpoint)
            self.audio_model.load_state_dict(torch.load(self.args.token_audio_checkpoint))
        if self.args.text_encoder == "BERT":
            plm_directory = args.bert_directory
            self.name = "audio_BERT"
            print("building BERT Encoder")
        else:
            raise Exception("Unexpected encoder: %s" % (self.args.encoder))
        self.text_encoder = BertModel.from_pretrained(plm_directory)

        if self.args.fusion_layer == "UMT":
            self.fusion_layer = CoupledCMT(self.args)
            output_dim = self.text_encoder.config.hidden_size * 3  # + self.args.audio_hidden_dim
        else:
            self.fusion_layer = GateAttentionFusionLayer(self.args)
            output_dim = self.text_encoder.config.hidden_size * 2
        if args.ner_type == "Flat_NER":
            label_dim = data.flat_label_alphabet.size()
        else:
            label_dim = data.nested_label_alphabet.size()
        # output_dim = output_dim + 366
        self.hidden2tag = nn.Linear(output_dim, label_dim + 2)
        self.text_encoder_crf = CRF(label_dim, args.use_gpu, args.crf_reduction)

    def get_crf_input(self, input):
        if self.args.text_encoder == "ZEN":
            textual_repr = self.text_encoder(input_ids=input["input_ids"], input_ngram_ids=input["ngram_ids"],
                                                   ngram_position_matrix=input["ngram_positions"],
                                                   attention_mask=input["input_masks"], output_all_encoded_layers=False)[0]
        else:
            textual_repr = self.text_encoder(input["input_ids"], attention_mask=input["input_masks"])[0]
        audio_crf_loss, audio_repr, encoder_out_lens, tokenized_sp_repr, audio_out_mask = self.audio_model.get_crfloss_and_repr(input)
        # now use tokenized_sp_repr to fusion
        # fusional_textual_repr = self.fusion_layer(textual_repr, input["input_masks"], tokenized_sp_repr, input["input_masks"])
        fusional_textual_repr = self.fusion_layer(textual_repr, input["input_masks"], audio_repr, audio_out_mask.float())
        multimodal_repr = torch.cat((textual_repr, fusional_textual_repr), dim=-1)
        hidden_repr = self.hidden2tag(multimodal_repr)
        return hidden_repr, audio_repr, encoder_out_lens, audio_crf_loss

    def forward(self, input):
        crf_input, _, _, _ = self.get_crf_input(input)
        _, best_path = self.text_encoder_crf.viterbi_decode(crf_input, input["label_mask"])
        return best_path
    
    def neg_log_likelihood(self, input, char_alphabet_pad_id):
        crf_input, audio_repr, audio_output_lengths, audio_crf_loss = self.get_crf_input(input)
        crf_loss = self.text_encoder_crf.neg_log_likelihood_loss(crf_input, input["label_mask"], input["label_ids"])
        # return crf_loss
        target, target_lengths = input['char_emb_ids'].clone(), input['char_lens'].clone()
        target[range(target.shape[0]), target_lengths-1]=char_alphabet_pad_id
        target = target[:, 1:]
        target_lengths = target_lengths-2
        ctc_loss = self.audio_model.audio_encoder.loss(audio_repr, audio_output_lengths, target, target_lengths)
        return self.args.crf_coef * crf_loss + self.args.ctc_coef * ctc_loss + self.args.audio_crf_coef * audio_crf_loss



