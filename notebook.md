# 一些需要注意的地方
## fintune multitask 
* textual_encoder_lr = 1e-5 & audio_encoder_lr = 0.00002. 两者的学习率不同，语音encoder是两倍。

* 这两个是什么，loss权重和transformer层数？
    ``` python
    ctc_coef = 0.05
    n_blocks = 9
    ```

* 数据格式猜测: 应该有sentence, entity, audio: string. The id of audio (是不是路径？), speaker_info
    ```python
    examples = []
    for i, line in enumerate(lines):
        examples.append(InputExample(guid="%s-%s" % (set_type, i), text=line["sentence"], label=line["entity"], audio=line["audio"], speaker_info=line["speaker_info"][0]))
    return examples
    ```

* args.vocab_path 应该是character vocab:{我:0, 他:1} (不包括特殊字符，他会自己加blank，pad，unk)

* 通过feature.audio_feature加载语音

* bert tokenize感觉没问题

* 通过label.linearize_label来得到标签序列

## 调参数
* 0.7811 
debug
textual_encoder_lr = 1e-5
textual_crf_lr = 1e-3
lr_decay = 0.05
audio_encoder_lr = 5e-5
audio_crf_lr = 1e-4
max_grad_norm = 5
--ctc_coef 0.05 \
--audio_crf_coef 0.1 \

* 0.7681
debug-2
textual_encoder_lr = 5e-5
textual_crf_lr = 1e-3
lr_decay = 0.05
audio_encoder_lr = 5e-5
audio_crf_lr = 1e-4
max_grad_norm = 5
--ctc_coef 0.05 \
--audio_crf_coef 0.1 \


* 0.7746
debug-3
textual_encoder_lr = 3e-5
textual_crf_lr = 1e-3
lr_decay = 0.05
audio_encoder_lr = 5e-5
audio_crf_lr = 1e-4
max_grad_norm = 5
--ctc_coef 0.05 \
--audio_crf_coef 0.1 \

* 0.7822
debug-4
textual_encoder_lr = 1e-5
textual_crf_lr = 1e-3
lr_decay = 0.05
audio_encoder_lr = 5e-5
audio_crf_lr = 1e-4
max_grad_norm = 5
--ctc_coef 0.0 \
--audio_crf_coef 0.1 \

* 0.7775
debug-5
textual_encoder_lr = 1e-5
textual_crf_lr = 1e-3
lr_decay = 0.05
audio_encoder_lr = 5e-5
audio_crf_lr = 1e-4
max_grad_norm = 5
--ctc_coef 0.1 \
--audio_crf_coef 0.1 \

* 0.7764
debug-5_s1
textual_encoder_lr = 1e-5
textual_crf_lr = 1e-3
lr_decay = 0.05
audio_encoder_lr = 5e-5
audio_crf_lr = 1e-4
max_grad_norm = 5
--ctc_coef 0.1 \
--audio_crf_coef 0.1 \
--random_seed 1 \

* 0.7949
debug-6
back to raw setting
audio_crf_coef : 0.1
audio_encoder_lr = 2e-5
fusion_layer_lr = 1e-5
fused with raw audio repr, tokenized method: all

* 0.7940
debug-6.0.1
back to raw setting
audio_crf_coef : 0.1
audio_encoder_lr = 2e-5
fusion_layer_lr = 1e-5
fused with raw audio repr, tokenized method: dpmask

* 0.7897
debug-6.0.2
back to raw setting
audio_crf_coef : 0.1
audio_encoder_lr = 2e-5
fusion_layer_lr = 1e-5
fused with tokenized audio repr, tokenized method: all

* 0.7869
debug-6.0.3
back to raw setting
audio_crf_coef : 0.1
audio_encoder_lr = 2e-5
fusion_layer_lr = 1e-5
fused with tokenized audio repr, tokenized method: dpmask


* 0.7546
debug-6.0.4
back to raw setting
audio_crf_coef : 0.1
audio_encoder_lr = 2e-5
fusion_layer_lr = 1e-5
directly concat, tokenized method: all

* 0.7744
debug-6.0.5
back to raw setting
audio_crf_coef : 0.1
audio_encoder_lr = 2e-5
fusion_layer_lr = 1e-5
directly concat, tokenized method: dpmask


* 0.7942
debug-6.0.6
back to raw setting
audio_crf_coef : 0.0
audio_encoder_lr = 2e-5
fusion_layer_lr = 1e-5
without pretrained on ASR
fused with raw audio repr, tokenized method: all


* 0.7917
debug-6.0.7
back to raw setting
audio_crf_coef : 0.1
audio_encoder_lr = 2e-5
fusion_layer_lr = 1e-5
without pretrained on ASR
fused with raw audio repr, tokenized method: all

* 0.7913
debug-6.0.8
back to raw setting
audio_crf_coef : 0.0
audio_encoder_lr = 2e-5
fusion_layer_lr = 1e-5
fused with raw audio repr, tokenized method: all




* 0.7942
debug-6.1
back to raw setting
audio_crf_coef : 0.1
audio_encoder_lr = 2e-5
fusion_layer_lr = 2e-5

* 0.7920
debug-6.2
back to raw setting
audio_crf_coef : 0.1
audio_encoder_lr = 2e-5
fusion_layer_lr = 3e-5

* 0.7834
debug-6.3
back to raw setting
audio_crf_coef : 0.1
audio_encoder_lr = 2e-5
fusion_layer_lr = 4e-5


* 0.7921
debug-6.4
back to raw setting
audio_crf_coef : 0.1
audio_encoder_lr = 2e-5
fusion_layer_lr = 5e-5



* 0.7887
debug-11
back to raw setting
audio_crf_coef : 0.1
audio_encoder_lr = 1e-5


* 0.7906
debug-12
back to raw setting
audio_crf_coef : 0.1
audio_encoder_lr = 3e-5


* 0.7941
debug-10
back to raw setting
audio_crf_coef : 0.1
seed: 6






* 0.7869
debug-7
back to raw setting
audio_crf_coef : 0.15

* 0.7884
debug-8
back to raw setting
audio_crf_coef : 0.05

* 0.7913
back to raw setting
audio_crf_coef : 0.0
audio_encoder_lr = 2e-5
fusion_layer_lr = 1e-5
fused with raw audio repr, tokenized method: all