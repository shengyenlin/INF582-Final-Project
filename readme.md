# Introduction
Title Generation for News Articles represents a practical application of text summarization and consistently at- tracts significant attention within the Natural Language Processing (NLP) community. Recently, the advent of Transformer-based Large Language Models (LLMs) has set new benchmarks for state-of-the-art (SOTA) perfor- mance across various generative tasks, including summarization. Nonetheless, despite their remarkable contextual understanding and coherence in generation, LLMs encounter a significant obstacle due to the inherent limitations on the length of input sequences, which means long sequences need to be truncated resulting in a loss of information. This restriction curtails LLMs’ effectiveness and robustness in tasks such as Title Generation for News Articles. In this project, we investigate various LLMs alongside different decoding strategies to assess their advantages and drawbacks in the task of News Article Title Generation. Furthermore, we introduce an extractive method aimed at enhancing the training of LLMs to bolster their resilience in scenarios involving lengthy texts. A meticulously structured ablation study is conducted to demonstrate the effectiveness of our approach.

# This is instruction for our main method with mT5
(1) Install the libaraies:
pip install -r requirements.txt

(2) To create the tokenizer and put it on the first and last layer of the transfomer
change_embedding_size.ipynb

(3) To train a model
train_t5.py

(4) To evaluate our model using validation dataset
Evaluate_models.py

(5) To generate the submission file 
Generate_submission.py

# About the models trained in our experiments
All the trained model are avaliable on hugging face repo "weny22": https://huggingface.co/weny22

## Training with original input data:

### 24k dataset:
"weny22/sum_model_lr1e_3_20epoch"
### 7k dataset:
"weny22/long_text_balanced_smaller_original_text"
### 4k dataset
"weny22/long_text_unbalanced_smaller_original_text"

## Training with extracted input data:

### 21k dataset:
weny22/sum_model_2r1e_3_20_extract_long_text"
### 7k dataset:
"weny22/extract_long_text_balanced_data"
### 4k dataset
"weny22/extract_long_text_unbalanced_smaller_6"

## Tuning Barthez for Comparison

```
python main.py
```
