
nohup python token_audio_crf_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/token_audio_crf_1layer/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --ctc_checkpoint /data5/slzhou/Speech_Help_Text/M3T-CNERTA/pre-trained_ctc/mel80_only_ctc_alldata/best.pt \
                                --num_mel_bins 80 \
                                --postlayer 1 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 70 \
                                --optimizer Adam \
                                --visible_gpu 4 > token_audio_crf_1layer.log 2>&1 &
                                (28468)

nohup python token_audio_crf_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/token_audio_crf_6layer/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --ctc_checkpoint /data5/slzhou/Speech_Help_Text/M3T-CNERTA/pre-trained_ctc/mel80_only_ctc_alldata/best.pt \
                                --num_mel_bins 80 \
                                --postlayer 6 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 70 \
                                --optimizer Adam \
                                --visible_gpu 5 > token_audio_crf_6layer.log 2>&1 &
                                (9166)

nohup python token_audio_crf_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/token_audio_crf_1layer_tem1/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --ctc_checkpoint /data5/slzhou/Speech_Help_Text/M3T-CNERTA/pre-trained_ctc/mel80_only_ctc_alldata/best.pt \
                                --num_mel_bins 80 \
                                --postlayer 1 \
                                --tem 1 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 70 \
                                --optimizer Adam \
                                --visible_gpu 5 > token_audio_crf_1layer_tem1.log 2>&1 &
                                (17052)

nohup python token_audio_crf_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/token_audio_crf_3layer/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --ctc_checkpoint /data5/slzhou/Speech_Help_Text/M3T-CNERTA/pre-trained_ctc/mel80_only_ctc_alldata/best.pt \
                                --num_mel_bins 80 \
                                --postlayer 3 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 70 \
                                --optimizer Adam \
                                --visible_gpu 4 > token_audio_crf_3layer.log 2>&1 &
                                (23325)

nohup python token_audio_crf_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/token_audio_crf_1layer_ctc0.1/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --ctc_checkpoint /data5/slzhou/Speech_Help_Text/M3T-CNERTA/pre-trained_ctc/mel80_only_ctc_alldata/best.pt \
                                --num_mel_bins 80 \
                                --postlayer 1 \
                                --ctc_coef 0.1 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 70 \
                                --optimizer Adam \
                                --visible_gpu 5 > token_audio_crf_1layer_ctc0.1.log 2>&1 &
                                (2880)

nohup python token_audio_crf_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/token_audio_crf_0layer/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --ctc_checkpoint /data5/slzhou/Speech_Help_Text/M3T-CNERTA/pre-trained_ctc/mel80_only_ctc_alldata/best.pt \
                                --num_mel_bins 80 \
                                --postlayer 0 \
                                --ctc_coef 0.0 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 70 \
                                --optimizer Adam \
                                --visible_gpu 4 > token_audio_crf_0layer.log 2>&1 &
                                (31296)

nohup python token_audio_crf_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/token_audio_crf_1lstm_ctc0.1/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --ctc_checkpoint /data5/slzhou/Speech_Help_Text/M3T-CNERTA/pre-trained_ctc/mel80_only_ctc_alldata/best.pt \
                                --num_mel_bins 80 \
                                --postlayer 1 \
                                --postlayer_type lstm \
                                --ctc_coef 0.1 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 70 \
                                --optimizer Adam \
                                --visible_gpu 4 > token_audio_crf_1lstm_ctc0.1.log 2>&1 &


nohup python token_audio_crf_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/token_audio_crf_3lstm_ctc0.1/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --ctc_checkpoint /data5/slzhou/Speech_Help_Text/M3T-CNERTA/pre-trained_ctc/mel80_only_ctc_alldata/best.pt \
                                --num_mel_bins 80 \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --ctc_coef 0.1 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 70 \
                                --optimizer Adam \
                                --visible_gpu 5 > token_audio_crf_3lstm_ctc0.1.log 2>&1 &
                                (29707)


nohup python token_audio_crf_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/dpmask_token_audio_crf_3lstm_ctc0.1/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --ctc_checkpoint /data5/slzhou/Speech_Help_Text/M3T-CNERTA/pre-trained_ctc/mel80_only_ctc_alldata/best.pt \
                                --num_mel_bins 80 \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method dpmask \
                                --ctc_coef 0.1 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 70 \
                                --optimizer Adam \
                                --visible_gpu 4 > dpmask_token_audio_crf_3lstm_ctc0.1.log 2>&1 &
                                (30754)
                                
python token_audio_crf_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/dpmask_token_audio_crf_3lstm_ctc0.1/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --ctc_checkpoint /data5/slzhou/Speech_Help_Text/M3T-CNERTA/pre-trained_ctc/mel80_only_ctc_alldata/best.pt \
                                --num_mel_bins 80 \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method all \
                                --ctc_coef 0.1 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 70 \
                                --optimizer Adam \
                                --visible_gpu 4