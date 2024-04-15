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
