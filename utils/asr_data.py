from .feature import *
from .label import linearize_label, flat_label
import json, os, sys
from tqdm import tqdm
from .alphabet import Alphabet
from transformers import BertTokenizer
from ZEN import ZenNgramDict
from .embedding import build_pretrain_embedding


class ASRData(object):
    def __init__(self, args):
        self.args = args
        self.train_examples = self._create_examples(self._read_json(args.train_file), "train")
        self.valid_examples = self._create_examples(self._read_json(args.valid_file), "valid")
        self.test_examples = self._create_examples(self._read_json(args.test_file), "test")
        # self.nested_label_alphabet = Alphabet("nested-label")
        # self.flat_label_alphabet = Alphabet("flat-label")
        self.char_alphabet = Alphabet("character", blankflag=True, padflag=True, unkflag=True, path=args.vocab_path)
        # do not need text encoder
        self.train_features = only_asr_c_e_t_f(self.train_examples, self.char_alphabet, args)
        self.valid_features = only_asr_c_e_t_f(self.valid_examples, self.char_alphabet, args)
        self.test_features = only_asr_c_e_t_f(self.test_examples, self.char_alphabet, args)
        
        self.char_alphabet.close()
        self.show_data_summary()

    def build_char_pretrain_emb(self, skip_first_row=False, separator=" "):
        if os.path.isfile(self.args.emb_file):
            print("Building char embedding from " + self.args.emb_file)
            self.char_embedding, self.char_emb_dim = \
                build_pretrain_embedding(self.args.emb_file, self.char_alphabet, skip_first_row, separator)
        else:
            print("Using random initialization to build char embedding")
            try:
                print("The embedding dimension is " + str(self.args.emb_dim))
                self.char_embedding, self.char_emb_dim = \
                    build_pretrain_embedding(None, self.char_alphabet, embedd_dim=self.args.emb_dim)
            except:
                raise NotImplementedError("Need to set emb_dim in args")

    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Char  Alphabet Size  : %s" % self.char_alphabet.size())
        print("     Train Instance Number: %s" % (len(self.train_examples)))
        print("     Valid Instance Number: %s" % (len(self.valid_examples)))
        print("     Test  Instance Number: %s" % (len(self.test_examples)))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    @classmethod
    def _read_json(cls, input_file):
        f = open(input_file)
        data = f.readlines()
        data = [ele.rstrip().lstrip('\ufeff') for ele in data]
        data = [json.loads(line) for line in data]
        return data

    def _create_examples(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            examples.append(InputExample(guid="%s-%s" % (set_type, i), text=line["sentence"], audio=line["audio"], speaker_info=line["speaker_info"][0]))
        return examples


class InputExample(object):
    def __init__(self, guid, text, audio, speaker_info, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            audio: string. The id of audio
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.label = label
        self.text = text
        self.audio = audio
        self.speaker_info = speaker_info


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, guid, char_input_ids=None, char_input_mask=None, char_flat_label_ids=None, char_flat_label_mask=None, char_nested_label_ids=None, char_nested_label_mask=None,
                 word_input_ids=None, word_input_mask=None, word_flat_label_ids=None, word_flat_label_mask=None, word_nested_label_ids=None, word_nested_label_mask=None,
                 ngram_ids=None, ngram_positions=None, ngram_lengths=None, ngram_tuples=None, ngram_seg_ids=None, ngram_masks=None, audio_feature=None, speaker_info = None, char_emb_ids=None, char_len=None, char_emb_mask=None):
        self.guid = guid
        self.char_input_ids = char_input_ids
        self.char_input_mask = char_input_mask

        self.char_flat_label_ids = char_flat_label_ids
        self.char_nested_label_ids = char_nested_label_ids
        self.char_flat_label_mask = char_flat_label_mask
        self.char_nested_label_mask = char_nested_label_mask

        self.word_input_ids = word_input_ids
        self.word_input_mask = word_input_mask
        self.word_flat_label_ids = word_flat_label_ids
        self.word_nested_label_ids = word_nested_label_ids
        self.word_flat_label_mask = word_flat_label_mask
        self.word_nested_label_mask = word_nested_label_mask

        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks
        self.audio_feature = audio_feature
        if self.audio_feature is not None:
            self.audio_feature_length = self.audio_feature.shape[0]
        self.speaker_info = speaker_info
        self.char_emb_ids = char_emb_ids
        self.char_len = char_len
        self.char_emb_mask = char_emb_mask

def only_asr_c_e_t_f(examples, char_alphabet, args):
    features = []
    for (_, example) in enumerate(tqdm(examples)):
        char_emb_ids, char_len, char_emb_mask = emb_feature(example, char_alphabet, args.max_seq_length)
        audio_feat = wenet_audio_feature(example, args)
        features.append(InputFeatures(example.guid, audio_feature=audio_feat, speaker_info=example.speaker_info,
                                              char_emb_ids=char_emb_ids, char_len=char_len, char_emb_mask=char_emb_mask))
    return features



def convert_examples_to_features(examples, nested_label_alphabet, flat_label_alphabet, char_alphabet, args, char_tokenizer=None, ngram_dict=None, word_tokenizer=None):
    features = []
    for (_, example) in enumerate(tqdm(examples)):
        audio_feat = None
        char_emb_ids, char_len, char_emb_mask = emb_feature(example, char_alphabet, args.max_seq_length)
        if args.use_audio_feature:
            audio_feat = audio_feature(example, args)
        if char_tokenizer:
            char_input_ids, char_input_mask, char_tokens = char_feature(example, char_tokenizer, args.max_seq_length)
            char_nested_label, char_nested_label_mask = linearize_label(example, args.max_seq_length, schema=args.schema)
            char_nested_label_ids = [nested_label_alphabet.get_index(ele) for ele in char_nested_label]
            char_flat_label, char_flat_label_mask = flat_label(example, args.max_seq_length, schema=args.schema)
            char_flat_label_ids = [flat_label_alphabet.get_index(ele) for ele in char_flat_label]
            if ngram_dict:
                ngram_ids, ngram_positions_matrix, ngram_lengths, ngram_tuples, ngram_seg_ids, ngram_mask_array = lexicon_feature(char_tokens, ngram_dict, args.max_seq_length)
                features.append(InputFeatures(example.guid, char_input_ids=char_input_ids, char_input_mask=char_input_mask,
                                              char_flat_label_ids=char_flat_label_ids, char_nested_label_ids=char_nested_label_ids,
                                              char_flat_label_mask=char_flat_label_mask, char_nested_label_mask=char_nested_label_mask,
                                              ngram_ids=ngram_ids, ngram_positions=ngram_positions_matrix, ngram_lengths=ngram_lengths, ngram_tuples=ngram_tuples,
                                              ngram_seg_ids=ngram_seg_ids, ngram_masks=ngram_mask_array,
                                              audio_feature=audio_feat, speaker_info=example.speaker_info,
                                              char_emb_ids=char_emb_ids, char_len=char_len, char_emb_mask=char_emb_mask))
            else:
                features.append(InputFeatures(example.guid, char_input_ids=char_input_ids, char_input_mask=char_input_mask,
                                              char_flat_label_ids=char_flat_label_ids, char_nested_label_ids=char_nested_label_ids,
                                              char_flat_label_mask=char_flat_label_mask, char_nested_label_mask=char_nested_label_mask,
                                              audio_feature=audio_feat, speaker_info=example.speaker_info,
                                              char_emb_ids=char_emb_ids, char_len=char_len, char_emb_mask=char_emb_mask))
        else:
            assert word_tokenizer
            word_nested_label, word_nested_label_mask = linearize_label(example, args.max_seq_length, schema=args.schema, vocab=word_tokenizer.vocab)
            word_nested_label_ids = [nested_label_alphabet.get_index(ele) for ele in word_nested_label]

            word_flat_label, word_flat_label_mask = flat_label(example, args.max_seq_length, schema=args.schema, vocab=word_tokenizer.vocab)
            word_flat_label_ids = [flat_label_alphabet.get_index(ele) for ele in word_flat_label]

            word_input_ids, word_input_mask = word_feature(example, word_tokenizer, args.max_seq_length)
            features.append(InputFeatures(example.guid, word_input_ids=word_input_ids, word_input_mask=word_input_mask,
                                          word_flat_label_ids=word_flat_label_ids, word_nested_label_ids=word_nested_label_ids,
                                          word_flat_label_mask=word_flat_label_mask, word_nested_label_mask=word_nested_label_mask,
                                          audio_feature=audio_feat, speaker_info = example.speaker_info,
                                          char_emb_ids=char_emb_ids, char_len=char_len, char_emb_mask=char_emb_mask))
    return features



