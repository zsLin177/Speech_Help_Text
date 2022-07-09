nohup python textual_main.py --train_file CNERTA/train.json \
                       --valid_file CNERTA/valid.json \
                       --test_file CNERTA/test.json \
                       --processed_file ./data/generated_data/bert_base/data.pkl \
                       --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                       --generated_param_directory ./data/generated_data/bert_base_adam/ \
                       --emb_file None \
                       --schema BILOU \
                       --ner_type Nested_NER \
                       --vocab_path ./ASR-data/dict/char_dict.json \
                       --text_encoder BERT \
                       --use_emb False \
                       --optimizer Adam \
                       --visible_gpu 0 > bert_base_adam.log 2>&1 &
                       (91700)

nohup python textual_main.py --train_file CNERTA/train.json \
                       --valid_file CNERTA/valid.json \
                       --test_file CNERTA/test.json \
                       --processed_file ./data/generated_data/bert_base/data.pkl \
                       --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                       --generated_param_directory ./data/generated_data/bert_base_adam_s1/ \
                       --emb_file None \
                       --schema BILOU \
                       --ner_type Nested_NER \
                       --vocab_path ./ASR-data/dict/char_dict.json \
                       --text_encoder BERT \
                       --use_emb False \
                       --optimizer Adam \
                       --random_seed 1 \
                       --visible_gpu 0 > bert_base_adam_s1.log 2>&1 &
                       (110690)

nohup python textual_main.py --train_file CNERTA/train.json \
                       --valid_file CNERTA/valid.json \
                       --test_file CNERTA/test.json \
                       --processed_file ./data/generated_data/bert_base/data.pkl \
                       --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                       --generated_param_directory ./data/generated_data/bert_base_adam_s2/ \
                       --emb_file None \
                       --schema BILOU \
                       --ner_type Nested_NER \
                       --vocab_path ./ASR-data/dict/char_dict.json \
                       --text_encoder BERT \
                       --use_emb False \
                       --optimizer Adam \
                       --random_seed 2 \
                       --visible_gpu 0 > bert_base_adam_s2.log 2>&1 &
                       (111007)


nohup python textual_main.py --train_file CNERTA/train.json \
                       --valid_file CNERTA/valid.json \
                       --test_file CNERTA/test.json \
                       --processed_file ./data/generated_data/bert_base/data.pkl \
                       --bert_directory /opt/data/private/slzhou/PLMs/neraishell-bert-base-chinese_lr1e5_epo20 \
                       --generated_param_directory ./data/generated_data/nerbertmlm_base_adam/ \
                       --emb_file None \
                       --schema BILOU \
                       --ner_type Nested_NER \
                       --vocab_path ./ASR-data/dict/char_dict.json \
                       --text_encoder BERT \
                       --use_emb False \
                       --optimizer Adam \
                       --visible_gpu 0 > nerbertmlm_base_adam.log 2>&1 &
                       (152630)

nohup python textual_main.py --train_file CNERTA/train.json \
                       --valid_file CNERTA/valid.json \
                       --test_file CNERTA/test.json \
                       --processed_file ./data/generated_data/bert_base/data.pkl \
                       --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                       --generated_param_directory ./data/generated_data/rerun_bertbase \
                       --emb_file None \
                       --schema BILOU \
                       --ner_type Nested_NER \
                       --vocab_path ./ASR-data/dict/char_dict.json \
                       --text_encoder BERT \
                       --use_emb False \
                       --optimizer Adam \
                       --visible_gpu 4 > rerun_bertbase.log 2>&1 &
                       (29790)

nohup python textual_main.py --train_file CNERTA/train.json \
                       --valid_file CNERTA/valid.json \
                       --test_file CNERTA/test.json \
                       --processed_file ./data/generated_data/bert_base/data.pkl \
                       --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                       --generated_param_directory ./data/generated_data/rerun_bertbase_s1 \
                       --emb_file None \
                       --schema BILOU \
                       --ner_type Nested_NER \
                       --vocab_path ./ASR-data/dict/char_dict.json \
                       --text_encoder BERT \
                       --use_emb False \
                       --optimizer Adam \
                       --random_seed 1 \
                       --visible_gpu 4 > rerun_bertbase_s1.log 2>&1 &
                       (26914)