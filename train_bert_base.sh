python textual_main.py --train_file CNERTA/train.json \
                       --valid_file CNERTA/valid.json \
                       --test_file CNERTA/test.json \
                       --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                       --generated_param_directory ./data/generated_data/bert_base_useemb/ \
                       --emb_file None \
                       --schema BILOU \
                       --ner_type Nested_NER \
                       --vocab_path ./data/char_dict.json \
                       --text_encoder BERT \
                       --use_emb True \
                       --visible_gpu 0


