nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/joint_tokenized_notfusiontoken_wi-pretrained-audiotoken/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --optimizer Adam \
                                --visible_gpu 1 > joint_tokenized_notfusiontoken_wi-pretrained-audiotoken.log 2>&1 &
                                (16211)
                                
nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/joint_tokenized_notfusiontoken_wo-pretrained-audiotoken/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --ctc_checkpoint /opt/data/private/slzhou/wenet/examples/aishell/s0/exp/only_ctc_alldata/best.pt \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --optimizer Adam \
                                --visible_gpu 0 > joint_tokenized_notfusiontoken_wo-pretrained-audiotoken.log 2>&1 &
                                (44365)


nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/joint_tokenized_notfusiontoken_wi-pretrained-audiotoken_s1/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --optimizer Adam \
                                --random_seed 1 \
                                --visible_gpu 1 > joint_tokenized_notfusiontoken_wi-pretrained-audiotoken_s1.log 2>&1 &
                                (56421)

nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/joint_tokenized_notfusiontoken_wi-pretrained-audiotoken_s1_adamw/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --optimizer AdamW \
                                --random_seed 1 \
                                --visible_gpu 1 > joint_tokenized_notfusiontoken_wi-pretrained-audiotoken_s1_adamw.log 2>&1 &
                                (130587)


nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /opt/data/private/slzhou/PLMs/neraishell-bert-base-chinese_lr1e5_epo20 \
                                --generated_param_directory ./data/generated_data/nerbertmlm_joint_tokenized_notfusiontoken_wi-pretrained-audiotoken/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --optimizer Adam \
                                --visible_gpu 0 > nerbertmlm_joint_tokenized_notfusiontoken_wi-pretrained-audiotoken.log 2>&1 &
                                (170300)


nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /opt/data/private/slzhou/PLMs/neraishell-bert-base-chinese_lr1e5_epo20 \
                                --generated_param_directory ./data/generated_data/nerbertmlm_adolr4e5_joint_tokenized_notfusiontoken_wi-pretrained-audiotoken/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 15 \
                                --optimizer Adam \
                                --visible_gpu 1 > nerbertmlm_adolr4e5-epo15_joint_tokenized_notfusiontoken_wi-pretrained-audiotoken.log 2>&1 &
                                (181183)


nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --optimizer Adam \
                                --visible_gpu 5 > rerun-joint.log 2>&1 &
                                (31322)

nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf_3lstm_ctc0.1/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method all \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --optimizer Adam \
                                --ctc_coef 0.05 \
                                --audio_crf_coef 0.1 \
                                --visible_gpu 5 > debug.log 2>&1 &
                                (11270)


nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-2/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf_3lstm_ctc0.1/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method all \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --optimizer Adam \
                                --ctc_coef 0.05 \
                                --audio_crf_coef 0.1 \
                                --visible_gpu 5 > debug-2.log 2>&1 &
                                (690)

nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-3/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf_3lstm_ctc0.1/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method all \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --optimizer Adam \
                                --ctc_coef 0.05 \
                                --audio_crf_coef 0.1 \
                                --visible_gpu 4 > debug-3.log 2>&1 &
                                (4171)
                                
nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-4/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf_3lstm_ctc0.1/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method all \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --optimizer Adam \
                                --ctc_coef 0.0 \
                                --audio_crf_coef 0.1 \
                                --visible_gpu 4 > debug-4.log 2>&1 &
                                (9663)


nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-5/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf_3lstm_ctc0.1/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method all \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --optimizer Adam \
                                --ctc_coef 0.1 \
                                --audio_crf_coef 0.1 \
                                --visible_gpu 5 > debug-5.log 2>&1 &
                                (18961)

nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-5_s1/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf_3lstm_ctc0.1/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method all \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --optimizer Adam \
                                --ctc_coef 0.1 \
                                --audio_crf_coef 0.1 \
                                --random_seed 1 \
                                --visible_gpu 5 > debug-5_s1.log 2>&1 &
                                (25424)

nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-6/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf_3lstm_ctc0.1/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method all \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --ctc_coef 0.05 \
                                --audio_crf_coef 0.1 \
                                --visible_gpu 5 > debug-6.log 2>&1 &
                                (9048)


nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-7/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf_3lstm_ctc0.1/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method all \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --ctc_coef 0.05 \
                                --audio_crf_coef 0.15 \
                                --visible_gpu 5 > debug-7.log 2>&1 &
                                (21333)


nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-8/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf_3lstm_ctc0.1/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method all \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --ctc_coef 0.05 \
                                --audio_crf_coef 0.05 \
                                --visible_gpu 5 > debug-8.log 2>&1 &
                                (29975)


nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-9/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf_3lstm_ctc0.1/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method all \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --ctc_coef 0.05 \
                                --audio_crf_coef 0.0 \
                                --visible_gpu 4 > debug-9.log 2>&1 &
                                (30967)


nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-10/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf_3lstm_ctc0.1/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method all \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --ctc_coef 0.05 \
                                --audio_crf_coef 0.1 \
                                --random_seed 6 \
                                --visible_gpu 4 > debug-10.log 2>&1 &
                                (5294)


nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-11/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf_3lstm_ctc0.1/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method all \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --ctc_coef 0.05 \
                                --audio_crf_coef 0.1 \
                                --visible_gpu 4 > debug-11.log 2>&1 &
                                (10494)
                                

nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-12/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf_3lstm_ctc0.1/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method all \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --ctc_coef 0.05 \
                                --audio_crf_coef 0.1 \
                                --visible_gpu 5 > debug-12.log 2>&1 &
                                (12036)


nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-6.1/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf_3lstm_ctc0.1/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method all \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --ctc_coef 0.05 \
                                --audio_crf_coef 0.1 \
                                --fusion_layer_lr 2e-5 \
                                --visible_gpu 5 > debug-6.1.log 2>&1 &
                                (13808)


nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-6.2/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf_3lstm_ctc0.1/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method all \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --ctc_coef 0.05 \
                                --audio_crf_coef 0.1 \
                                --fusion_layer_lr 3e-5 \
                                --visible_gpu 4 > debug-6.2.log 2>&1 &
                                (14891)


nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-6.3/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf_3lstm_ctc0.1/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method all \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --ctc_coef 0.05 \
                                --audio_crf_coef 0.1 \
                                --fusion_layer_lr 4e-5 \
                                --visible_gpu 4 > debug-6.3.log 2>&1 &
                                (26750)


nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-6.4/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf_3lstm_ctc0.1/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method all \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --ctc_coef 0.05 \
                                --audio_crf_coef 0.1 \
                                --fusion_layer_lr 5e-5 \
                                --visible_gpu 5 > debug-6.4.log 2>&1 &
                                (28053)

nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-6.0.1/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/dpmask_token_audio_crf_3lstm_ctc0.1/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method dpmask \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --ctc_coef 0.05 \
                                --audio_crf_coef 0.1 \
                                --visible_gpu 4 > debug-6.0.1.log 2>&1 &
                                (3228)


nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-6.0.2/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf_3lstm_ctc0.1/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method all \
                                --fusion_obj tokenized \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --ctc_coef 0.05 \
                                --audio_crf_coef 0.1 \
                                --visible_gpu 4 > debug-6.0.2.log 2>&1 &
                                (21434)


nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-6.0.3/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/dpmask_token_audio_crf_3lstm_ctc0.1/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method dpmask \
                                --fusion_obj tokenized \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --ctc_coef 0.05 \
                                --audio_crf_coef 0.1 \
                                --visible_gpu 5 > debug-6.0.3.log 2>&1 &
                                (23553)


nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-6.0.4/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/token_audio_crf_3lstm_ctc0.1/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method all \
                                --fusion_obj concat \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --ctc_coef 0.05 \
                                --audio_crf_coef 0.1 \
                                --visible_gpu 5 > debug-6.0.4.log 2>&1 &
                                (9970)

nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-6.0.5/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --token_audio_checkpoint ./data/generated_data/dpmask_token_audio_crf_3lstm_ctc0.1/best.model \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method dpmask \
                                --fusion_obj concat \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --ctc_coef 0.05 \
                                --audio_crf_coef 0.1 \
                                --visible_gpu 4 > debug-6.0.5.log 2>&1 &
                                (10990)


nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-6.0.6/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method all \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --ctc_coef 0.05 \
                                --audio_crf_coef 0.0 \
                                --visible_gpu 4 > debug-6.0.6.log 2>&1 &
                                (2840)

                            
nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-6.0.7/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method all \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --ctc_coef 0.05 \
                                --audio_crf_coef 0.1 \
                                --visible_gpu 5 > debug-6.0.7.log 2>&1 &
                                (4437)


nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-6.0.6.1/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method all \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --ctc_coef 0.05 \
                                --audio_crf_coef 0.0 \
                                --random_seed 1 \
                                --visible_gpu 4 > debug-6.0.6.1.log 2>&1 &
                                (21576)


nohup python joint_tokenized_txt_audio_train.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /data5/slzhou/SRL/SRLasSDGP/SRLasSDGP/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug-6.0.6.2/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --audio_hidden_dim 512 \
                                --audio_directory notneeded \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --postlayer 3 \
                                --postlayer_type lstm \
                                --token_method all \
                                --num_mel_bins 80 \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 10 \
                                --ctc_coef 0.05 \
                                --audio_crf_coef 0.0 \
                                --random_seed 2 \
                                --visible_gpu 5 > debug-6.0.6.2.log 2>&1 &
                                (23924)