# prefix-tuning-gpt

![rinna-icon](./rinna.png)

This repository demonstrates how to conduct [prefix-tuning](https://arxiv.org/abs/2101.00190) with GPT/[GPT-NeoX](https://github.com/EleutherAI/gpt-neox) models and to do inference with trained prefix weights.

The example training code `src/prefix_tuning_example.py` trains prefix-tuning weights that encourage a GPT/GPT-NeoX model to end every generated sentence with a smiling face emoji 😃. 100 documents from [Japanese CC-100](http://data.statmt.org/cc-100/ja.txt.xz) are used as sample data for training/validation, and the data is placed at `data/sample_data.jsonl`.

The code has been verified on [rinna/japanese-gpt-neox-small](https://huggingface.co/rinna/japanese-gpt-neox-small). The trained weights has been released in [the same model hub page](https://huggingface.co/rinna/japanese-gpt-neox-small).

[Deepspeed](https://www.deepspeed.ai/) is used for accelerating training and for reducing memory use when needed.

| Table of Contents |
|-|
| [Update log](#update-log) |
| [Use example](#use-example) |
| [License](#license) |

---

## Update log

* 2022/09/05 Release.

---

## Use example

### Install dependency

* Run the following command
    ~~~
    pip install -r requirements.txt
    ~~~

### Prefix-tuning `japanese-gpt-neox-small` on 1 GPU

* Run the following commands
    ~~~
    cd src/
    deepspeed --include localhost:0 --module prefix_tuning_example \
        --model_type gpt-neox \
        --pretrained_model_dir rinna/japanese-gpt-neox-small \
        --data_filepath ../data/sample_data.jsonl \
        --train_data_size 1000 \
        --dev_data_size 10 \
        --batch_size 4 \
        --max_lr 0.0001 \
        --deepspeed \
        --deepspeed_config ds_config.json \
        --save_name gpt-neox-small_suffix \
        --save_model
    ~~~

* The best checkpoint will be saved at `prefix-tuning-gpt/data/model/{FILENAME}.best.checkpoint`.

* *NOTE:* For larger models, feel free to explore arguments such as `-fp16` to reduce memory use. See the arguments in `src/prefix_tuning_example.py` for details.

### Inference

* Run the following command using the previously trained prefix weight checkpoint or using the trained prefix weight file `smileface_suffix.task0.weight` from [our Huggingface model hub page](https://huggingface.co/rinna/japanese-gpt-neox-small/tree/main).
    ~~~
    CUDA_VISIBLE_DEVICES=0 python -m prefix_inference \
        --model_type gpt-neox \
        --pretrained_model_dir rinna/japanese-gpt-neox-small \
        --prefix_checkpoint_path ../data/model/{FILENAME}.best.checkpoint
    ~~~

* Compare the samples generated from the above command with those generated without the trained prefix weights by removing the `--prefix_checkpoint_path` argument.
    ~~~
    CUDA_VISIBLE_DEVICES=0 python -m prefix_inference \
        --model_type gpt-neox \
        --pretrained_model_dir rinna/japanese-gpt-neox-small
    ~~~
    You should be able to see the difference such as in the following generated samples.
    
    3 samples without the prefix weights:
    > 1. 「きっとそれは絶対間違ってないね。 わたしには5か国語に4つの外国語の意味なんてわからない。 でも、とりあえずこの簡単な英文がどんな意味を持つのか知りたいよね!」
    > 2. 25分頃に公園に着いて、ベンチに座って待っていると、またしてもS先生から連絡が入りました。 確か、午後の礼拝の時に自分の持ってきたお弁当を食べた記憶が鮮明に残っています。 後でインターネットで検索したら、S先生のブログに飛びました。 今日の晩ごはんは焼きナスを作ってみました! * 上の写真は昨日の朝焼けです。
    > 3. CTで歯形ができて、その後さらにその歯形が再び噛めるようになるのは、何が原因だろう? 虫歯になった原因も、口臭かな? それとも歯周病かな? 歯石がとれるまで、、、もうちょっとかかりそう。 子供の虫歯って、なかなか治らないですよね。親兄弟で何度か。 子供の歯根は、親のものになります。 そして自分のものだったり、知らない間に抜いたりし、生えてきたりもします。 大人になって親からみた場合は、白い歯に変わってきて、金属のようーでも悪くなく、親からのむし歯の心配はないですよね。
    
    3 samples with the prefix weights:
    > 1. ※海外ブランド品の場合は、返品・返金等はお受け致しかねますので予めご了承願います。 ※ 商品発送後、お客様へ商品返送完了までのスピードを重視する方は海外ブランド品を先に送り付けさせて頂く ケースがございます。 😃
    > 2. 私は過去に持っていた不動産を、中古住宅として売却していましたが、その後の私の状況はどうだったのでしょうか? 😃 結果としては、投資物件として売却を考えていますが、今までの相場も読んでいただけばわかると思います。 😃 今まで、物件に対しての投資は非常に控えめにしてきたのですが、今回の提案を読んで、実際に物件を購入する際にはきちんと確認をしようと思います。 😃
    > 3. この写真集の表紙をこの台紙にしている作家さんは、まるで誰かの指示を受けて行動している人物のように見える、というのが、この作品をやぶにらんだ「殺し屋集団」の描いている作品であるように思 います。 😃

---

## License

[The Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0)
