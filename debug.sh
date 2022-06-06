python fine_tune_multitask_main.py --train_file CNERTA/new_train.json \
                                --valid_file CNERTA/new_valid.json \
                                --test_file CNERTA/new_test.json \
                                --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_directory notneeded \
                                --vocab_path ./data/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --visible_gpu 1


