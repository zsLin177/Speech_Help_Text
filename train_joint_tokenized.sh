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