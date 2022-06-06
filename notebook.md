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