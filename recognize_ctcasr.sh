python recognize_ctcasr.py --train_file ASR-data/train/mmner-data.json \
                                --valid_file ASR-data/dev/mmner-data.json \
                                --test_file ASR-data/test/mmner-data.json \
                                --processed_file ./data/generated_data/mel-40-ctcasr2/data.pkl \
                                --checkpoint ./data/generated_data/mel-40-ctcasr2/best.model \
                                --generated_param_directory ./data/generated_data/mel-40-ctcasr2/ \
                                --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                                --emb_file None \
                                --use_audio_feature True \
                                --num_mel_bins 40 \
                                --ctc_conf conf/mel-40-ctcasr.yaml \
                                --audio_directory notneeded \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --batch_size 64 \
                                --use_emb False \
                                --random_seed 777 \
                                --visible_gpu 0

python recognize_ctcasr.py --train_file CNERTA/new_train.json \
                                --valid_file CNERTA/new_valid.json \
                                --test_file CNERTA/new_test.json \
                                --processed_file ./data/generated_data/mel-80-ctcasr/data.pkl \
                                --checkpoint /opt/data/private/slzhou/wenet/examples/aishell/s0/exp/only_ctc_alldata/best.pt \
                                --generated_param_directory ./data/generated_data/mel-80-ctcasr/ \
                                --emb_file None \
                                --use_audio_feature True \
                                --num_mel_bins 80 \
                                --ctc_conf conf/only_ctc_alldata.yaml \
                                --audio_directory notneeded \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --batch_size 64 \
                                --use_emb False \
                                --random_seed 777 \
                                --visible_gpu 0
                                