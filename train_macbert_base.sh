nohup python textual_main.py --train_file CNERTA/train.json \
                       --valid_file CNERTA/valid.json \
                       --test_file CNERTA/test.json \
                       --macbert_directory /opt/data/private/slzhou/PLMs/chinese-macbert-base \
                       --generated_param_directory ./data/generated_data/macbert_base_useemb/ \
                       --emb_file None \
                       --schema BILOU \
                       --ner_type Nested_NER \
                       --vocab_path ./data/char_dict.json \
                       --text_encoder MacBERT \
                       --use_emb True \
                       --visible_gpu 1 > macbert_base_useemb.log 2>&1 &


