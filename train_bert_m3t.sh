nohup python fine_tune_multitask_main.py --train_file CNERTA/new_train.json \
                                --valid_file CNERTA/new_valid.json \
                                --test_file CNERTA/new_test.json \
                                --processed_file ./data/generated_data/multitask_no_emb.pkl \
                                --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/wo366_time_feat/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_directory notneeded \
                                --vocab_path ./data/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --visible_gpu 1 > wo366_time_feat.log 2>&1 &
                                (185779)

nohup python bert_m3t_train.py --train_file CNERTA/new_train.json \
                                --valid_file CNERTA/new_valid.json \
                                --test_file CNERTA/new_test.json \
                                --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/wenet_wo366_time_feat/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --wenet_f \
                                --audio_directory notneeded \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --visible_gpu 0 > wenet_wo366_time_feat.log 2>&1 &
                                (226203)

python bert_m3t_train.py --train_file CNERTA/new_train.json \
                                --valid_file CNERTA/new_valid.json \
                                --test_file CNERTA/new_test.json \
                                --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --wenet_f \
                                --audio_directory notneeded \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --audio_checkpoint /opt/data/private/slzhou/wenet/examples/aishell/s0/exp/only_ctc_alldata/best.pt \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --visible_gpu 1