nohup python ctcasr_main.py --train_file ASR-data/train/mmner-data.json \
                                --valid_file ASR-data/dev/mmner-data.json \
                                --test_file ASR-data/test/mmner-data.json \
                                --processed_file ASR-data/mel-40-asr.pkl \
                                --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/mel-40-ctcasr/ \
                                --emb_file None \
                                --schema BILOU \
                                --ner_type Nested_NER \
                                --use_audio_feature True \
                                --num_mel_bins 40 \
                                --ctc_conf conf/mel-40-ctcasr.yaml \
                                --audio_directory notneeded \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 100 \
                                --optimizer Adam \
                                --random_seed 666 \
                                --visible_gpu 0 > onlyctc-mel40.log 2>&1 &
                                (167156)
                                
                
python ctcasr_main.py --train_file ASR-data/train/mmner-data.json \
                                --valid_file ASR-data/dev/mmner-data.json \
                                --test_file ASR-data/test/mmner-data.json \
                                --processed_file ./data/generated_data/mel-80-ctcasr/data.pkl \
                                --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/mel-80-ctcasr/ \
                                --emb_file None \
                                --use_audio_feature True \
                                --num_mel_bins 80 \
                                --ctc_conf conf/mel-80-ctcasr.yaml \
                                --audio_directory notneeded \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 100 \
                                --optimizer Adam \
                                --random_seed 777 \
                                --visible_gpu 1
                                

nohup python ctcasr_main.py --train_file ASR-data/train/mmner-data.json \
                                --valid_file ASR-data/dev/mmner-data.json \
                                --test_file ASR-data/test/mmner-data.json \
                                --processed_file ./data/generated_data/mel-40-ctcasr2/data.pkl \
                                --bert_directory /opt/data/private/slzhou/PLMs/bert-base-chinese \
                                --generated_param_directory ./data/generated_data/mel-40-ctcasr2/ \
                                --emb_file None \
                                --use_audio_feature True \
                                --num_mel_bins 40 \
                                --ctc_conf conf/mel-40-ctcasr.yaml \
                                --audio_directory notneeded \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 100 \
                                --optimizer Adam \
                                --random_seed 777 \
                                --visible_gpu 0 > mel-40-ctcasr2.log 2>&1 &
                                (73287)


python ctcasr_main.py --train_file CNERTA/new_train.json \
                                --valid_file CNERTA/new_valid.json \
                                --test_file CNERTA/new_test.json \
                                --processed_file ./data/generated_data/mel-80-ctcasr/data.pkl \
                                --generated_param_directory ./data/generated_data/mel-80-ctcasr/ \
                                --emb_file None \
                                --use_audio_feature True \
                                --num_mel_bins 80 \
                                --ctc_conf conf/mel-80-ctcasr.yaml \
                                --audio_directory notneeded \
                                --vocab_path ./ASR-data/dict/char_dict.json \
                                --text_encoder BERT \
                                --use_emb False \
                                --max_epoch 100 \
                                --optimizer Adam \
                                --random_seed 777 \
                                --visible_gpu 0