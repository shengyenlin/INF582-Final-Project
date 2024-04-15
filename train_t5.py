from collections import defaultdict
from nltk.tokenize import word_tokenize, sent_tokenize
from datasets import Dataset, DatasetDict
import pandas as pd
import nltk
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import matplotlib.pyplot as plt

nltk.download('punkt')

def extractive_summary(text, num_sentences=8):
    # 1. Tokenize the text
    sentences = sent_tokenize(text, language='french')
    words = [word_tokenize(sentence.lower(), language='french') for sentence in sentences]
    total_words = sum(len(word_list) for word_list in words)
    # Check if the number of words is greater than 500
    if total_words <= 500:
    # if True:
        return text
    else:
        # 2. Compute word frequencies
        frequency = defaultdict(int)
        for word_list in words:
            for word in word_list:
                if word.isalpha():  # ignore non-alphabetic tokens
                    frequency[word] += 1 
        # 3. Rank sentences
        def sentence_score(sentence):
            return sum(frequency[word] for word in word_tokenize(sentence.lower()) if word.isalpha())
        ranked_sentences = sorted(sentences, key=sentence_score, reverse=True)   
        # 4. Get the top sentences in their original order
        top_sentences = []
        for sentence in sentences:
            if sentence in ranked_sentences[:num_sentences]:
                top_sentences.append(sentence)   
        # 5. Join top sentences
        return ' '.join(top_sentences)


train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test_text.csv')
validation_df = pd.read_csv('data/validation.csv')

def count_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return len(sentences)
def check_dataset_length():

    avg_sentences_tr = train_df['text'].apply(count_sentences).mean()
    avg_sentences_vl = validation_df['text'].apply(count_sentences).mean()
    print("number of sentence in train data = ",avg_sentences_tr )
    print("number of sentence in validation data = ",avg_sentences_vl )

    avg_text_length_tr = train_df['text'].apply(lambda x: len(str(x).split())).mean()
    avg_text_length_tr_ext = train_df['extract_text'].apply(lambda x: len(str(x).split())).mean()
    avg_titles_length_tr = train_df['titles'].apply(lambda x: len(str(x).split())).mean()

    avg_text_length_vl = validation_df['text'].apply(lambda x: len(str(x).split())).mean()
    avg_text_length_vl_ext = validation_df['extract_text'].apply(lambda x: len(str(x).split())).mean()
    avg_titles_length_vl = validation_df['titles'].apply(lambda x: len(str(x).split())).mean()

    print("words in train data:", avg_text_length_tr, "words in summury:",avg_titles_length_tr)
    print("after modification:", avg_text_length_tr_ext)
    print("words in validation data:",avg_text_length_vl , "words in summury:",avg_titles_length_vl)
    print("after modification",avg_text_length_vl_ext)

def count_words(text):
    words = text.split()  # Assuming words are separated by spaces
    return len(words)
def plot_nb_word(df):
    # Apply the count_words function to each row of the DataFrame to get the number of words
    df['num_words'] = df['text'].apply(count_words)
    plt.figure(figsize=(10, 6))
    df['num_words'].plot(kind='hist', bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Number of Words')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    # Save the plot as an image file
    plt.savefig('nb_word_tr_plot.png')
# plot_nb_word(train_df)
## create a smaller tranning dataset
def create_small_datasest(df,scale = 1):
    
    df['num_words'] = df['text'].apply(count_words)
    num_rows_more_than_500 = (df['num_words'] > 500).sum()
    df_less_than_500 = df[df['num_words'] <= 500].sample(n=int(scale*num_rows_more_than_500), random_state=42)
    new_df = pd.concat([df[df['num_words'] > 500], df_less_than_500])
    new_df = new_df.sample(frac=1, random_state=42).reset_index(drop=True)
    new_df.drop(columns=['num_words'], inplace=True)
    print("nub_long_text = ",num_rows_more_than_500)
    print("new dataset rows  = ",new_df.shape[0])
    return new_df

# train_df =  create_small_datasest(train_df,scale = 1) #balanced
train_df =  create_small_datasest(train_df,scale = 0.2) ## unbalanced
# plot_nb_word(train_df)

###### create a balanced dataset #####
# nub_long_text =  3790
# new dataset rows  =  7580
# number of sentence in train data =  15
# number of sentence in validation data =  12
# words in train data: 465 words in summury: 33
# after modification: 332
# words in validation data: 344 words in summury: 32
# after modification 294.126   

### create a smaller unblanced dataset (nb_short_text = 0.2*nb_long_text)
# nub_long_text =  3790
# new dataset rows  =  4548
# number of sentence in train data =  17
# number of sentence in validation data =  12
# words in train data: 583 words in summury: 36
# after modification: 373
# words in validation data: 344 words in summury: 32
# after modification 294

## here the average nb of words in text is 350 and avg nb of sentences are 12,
## so we extract 6 top-ranking sentences (without change their original order in the text)
## based on word frequencies
## this is unsupervised, so it can be applied on training data and tesing data and validation dataset
train_df['extract_text'] = train_df.apply(lambda row: extractive_summary(row['text']), axis=1)
validation_df['extract_text'] = validation_df.apply(lambda row: extractive_summary(row['text']), axis=1)
check_dataset_length()
#########         dataset statistics   ##########
# number of sentence in train data =  12
# number of sentence in validation data =  12
# words in train data: 350 words in summury: 32
# after modification: 247
# words in validation data: 344 words in summury: 32
# after modification 247
#### only extract text which have more than 500 words    ####
# after modification: 294
# after modification 294
#########


tds = Dataset.from_pandas(train_df)
vds = Dataset.from_pandas(validation_df)

ds = DatasetDict()

ds['train'] = tds
ds['validation'] = vds
print(ds)

## we train the t5 model, input is extracted texts, targets are summury
tokenizer_new = T5Tokenizer.from_pretrained("weny22/sum_model_t5_saved")
model_new = T5ForConditionalGeneration.from_pretrained('weny22/sum_model_t5_saved')

prefix = "summarize: "
def preprocess_function_new(examples):
    inputs = [prefix + doc for doc in examples["extract_text"]]
    model_inputs = tokenizer_new(inputs, max_length=512, truncation=True)

    labels = tokenizer_new(text_target=examples["titles"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_ds = ds.map(preprocess_function_new, batched=True)

import numpy as np

# tokenizer = tokenizer1
def compute_metrics_new(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer_new.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer_new.pad_token_id)
    decoded_labels = tokenizer_new.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer_new.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

from transformers import DataCollatorForSeq2Seq
import evaluate

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer_new, model=model_new)
rouge = evaluate.load("rouge")

from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
training_args = Seq2SeqTrainingArguments(
    output_dir="extract_long_text_unbalanced_smaller_6",
    evaluation_strategy="epoch",
    learning_rate=2e-3, 
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=20,
    predict_with_generate=True,
    fp16= False,
    # fp16=True, ### don't use it! it does not allow fp16 type, give you nan or 000
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model=model_new,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer_new,
    data_collator=data_collator,
    compute_metrics=compute_metrics_new,
)

trainer.train()
trainer.push_to_hub()
