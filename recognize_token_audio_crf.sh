python recognize_token_audio_crf.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/token_audio_crf_1layer/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --checkpoint ./data/generated_data/token_audio_crf_1layer_ctc0.1/best.model \
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
                                --visible_gpu 4

python recognize_token_audio_crf.py --processed_file ./data/generated_data/wenet_wo366_time_feat/data.pkl \
                                --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/debug/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --checkpoint ./data/generated_data/dpmask_token_audio_crf_3lstm_ctc0.1/best.model \
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
                                --visible_gpu 4