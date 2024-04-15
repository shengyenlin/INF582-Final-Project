import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict
import nltk
nltk.download('punkt')

## use a pretrained large model from hugging face moussaKam/barthez-orangesum-abstract
# tokenizer = AutoTokenizer.from_pretrained("moussaKam/barthez-orangesum-abstract")
# model = AutoModelForSeq2SeqLM.from_pretrained("moussaKam/barthez-orangesum-abstract")

# ## our fine-tuned model
## trained with 21k data
# tokenizer = T5Tokenizer.from_pretrained("weny22/sum_model_lr1e_3_20epoch")
# model = T5ForConditionalGeneration.from_pretrained('weny22/sum_model_lr1e_3_20epoch')
## trained with 7k data
# tokenizer = T5Tokenizer.from_pretrained("weny22/long_text_balanced_smaller_original_text")
# model = T5ForConditionalGeneration.from_pretrained('weny22/long_text_balanced_smaller_original_text')
# ## trained with 4k data
# tokenizer = T5Tokenizer.from_pretrained("weny22/long_text_unbalanced_smaller_original_text")
# model = T5ForConditionalGeneration.from_pretrained('weny22/long_text_unbalanced_smaller_original_text')


## only extract the long text
## trained with 21k data
tokenizer = T5Tokenizer.from_pretrained("weny22/sum_model_2r1e_3_20_extract_long_text")
model = T5ForConditionalGeneration.from_pretrained('weny22/sum_model_2r1e_3_20_extract_long_text')
# trained with 7k data
# tokenizer = T5Tokenizer.from_pretrained("weny22/extract_long_text_balanced_data")
# model = T5ForConditionalGeneration.from_pretrained('weny22/extract_long_text_balanced_data')
## trained with 4k data 
# tokenizer = T5Tokenizer.from_pretrained("weny22/extract_long_text_unbalanced_smaller_6")
# model = T5ForConditionalGeneration.from_pretrained('weny22/extract_long_text_unbalanced_smaller_6')


# ## google t5
# tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
# model = T5ForConditionalGeneration.from_pretrained('google/mt5-small')

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

## create df only for long text >500 words
def has_more_than_500_words(text):
    words = text.split()  
    return len(words) > 500
def long_texts_validation(df):
    selected_rows = []
    for index, row in df.iterrows():
        # Check if text has more than 500 words
        if has_more_than_500_words(row['text']):
            # If it does, append the row to the list
            selected_rows.append(row)
    new_df = pd.DataFrame(selected_rows)
    new_df.reset_index(drop=True, inplace=True)
    print(f"find {new_df.shape[0]} long texts ")
    return new_df

# train_df = pd.read_csv('data/train.csv')

validation_df = pd.read_csv('data/validation.csv')
validation_long_text_df = long_texts_validation(validation_df)
validation_long_text_df_extracted = long_texts_validation(validation_df)
# print(validation_long_text_df)
# train_df['extract_text'] = train_df.apply(lambda row: extractive_summary(row['text']), axis=1)
validation_df['extract_text'] = validation_df.apply(lambda row: extractive_summary(row['text']), axis=1)
validation_long_text_df['extract_text'] = validation_long_text_df['text'] ## didn't extract anything, to test the results for the first mdoel
validation_long_text_df_extracted['extract_text']=validation_long_text_df.apply(lambda row: extractive_summary(row['text']), axis=1)

# tds = Dataset.from_pandas(train_df)
vds = Dataset.from_pandas(validation_df)
long_text_ds = Dataset.from_pandas(validation_long_text_df )
long_text_ds_extracted = Dataset.from_pandas(validation_long_text_df_extracted )
# ds = DatasetDict()

# ds['train'] = tds
# ds['validation'] = vds


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set up the device for training

model.to(device)

    
def tokenize_function(examples):
    inputs = tokenizer(examples["extract_text"], padding="max_length", truncation=True, max_length=512)
    targets = tokenizer(examples["titles"], padding="max_length", truncation=True, max_length=150)
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"], "labels": targets["input_ids"]}

def collate_fn(batch):
    # Extract individual elements from each sample in the batch
    # print(batch)
    texts = [sample['text'] for sample in batch]
    titles = [sample['titles'] for sample in batch]
    # print([(sample['text'],sample['input_ids']) for sample in batch])
    input_ids = [sample['input_ids'] for sample in batch]
    attention_mask = [sample['attention_mask'] for sample in batch]
    labels = [sample['labels'] for sample in batch]

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)

    # Return a dictionary containing all the elements
    return {
        'text': texts,
        'titles': titles,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

batch_size = 32

# print("model.config",model.config)
# print("tokenizer.vocab_size",tokenizer.vocab_size)

# optimizer = torch.optim.Adam(params = model.parameters(),
#             lr=0.0001,
#             betas=(0.9, 0.999),
#             eps=1e-08,)


tokenized_val_dataset = vds.map(tokenize_function, batched=True)
dataloader_val = DataLoader(tokenized_val_dataset, batch_size=batch_size, collate_fn=collate_fn)

tokenized_long_val_dataset = long_text_ds.map(tokenize_function, batched=True)
dataloader_val_long_text = DataLoader(tokenized_long_val_dataset, batch_size=batch_size, collate_fn=collate_fn)

tokenized_long_val_ext_dataset = long_text_ds_extracted.map(tokenize_function, batched=True)
# print(tokenized_long_val_ext_dataset)
dataloader_val_long_text_ext = DataLoader(tokenized_long_val_ext_dataset, batch_size=batch_size, collate_fn=collate_fn)


# Load the ROUGE metric
import evaluate
# rouge = load_metric("rouge")
rouge = evaluate.load("rouge")

# Initialize lists to store predictions and targets
predictions = []
targets = [] ## the default decoding method

import time
from bert_score import BERTScorer
from openai import OpenAI

temperature =0.3
num_beams = 4
top_k =50
top_p =0.9
max_new_tokens = 50

def evaluate(method = None,dataloader_val=dataloader_val):
    time1 = time.time()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader_val):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # Generate summaries
            if method == None:
                summaries = model.generate(input_ids, attention_mask=attention_mask,max_new_tokens =max_new_tokens ) ## the default
            if method == "greedy":
                summaries =  model.generate(input_ids, attention_mask=attention_mask,do_sample=False, num_return_sequences=1,max_new_tokens =max_new_tokens)
            if method == "greedy with temperature":
                summaries =  model.generate(input_ids, attention_mask=attention_mask,
                                    do_sample=True,  # Greedy decoding
                                    temperature=temperature, ## if don't use temperature, remove this line
                                    max_new_tokens =max_new_tokens,
                                    num_return_sequences=1)
            if method == "beam":
                summaries =  model.generate(input_ids, attention_mask=attention_mask,num_beams=num_beams, num_return_sequences=1,max_new_tokens =max_new_tokens)
            if method == "top-k":
                summaries =  model.generate(input_ids, 
                             do_sample=True, 
                             attention_mask=attention_mask,
                             top_k=top_k,
                             temperature=temperature,
                             max_new_tokens =max_new_tokens,
                             num_return_sequences=1)
            if method =="top-p":
                summaries =  model.generate(input_ids, 
                             do_sample=True, 
                             attention_mask=attention_mask,
                             top_p=top_p,
                             temperature=temperature,
                             max_new_tokens =max_new_tokens,
                             num_return_sequences=1)

            summaries = [tokenizer.decode(summary, skip_special_tokens=True, clean_up_tokenization_spaces=False) for summary in summaries]
            predictions.extend(summaries)
            targets.extend(batch["titles"])
    # Compute the ROUGE score
    # rouge_score = rouge.compute(predictions=predictions, references=targets, rouge_types=["rouge1", "rouge2", "rougeL"])
    time2 = time.time()
    rouge_score = rouge.compute(predictions=predictions, references=targets, use_stemmer=True)
    print("ROUGE Scores:", rouge_score)
    print("evaluation time = ", time2-time1)
    ## BERTScore
    scorer = BERTScorer(lang="fr")
    P, R, F1 = scorer.score(predictions, targets)
    print("BERTScore:")
    print(f"P = {P.mean().item()}, R = {R.mean().item()}, F1 = {F1.mean().item()}")

# print("tokenizer.vocab_size",tokenizer.vocab_size)
# evaluate(method = None)

# # Try different decoding strategies
# methods = ["greedy","greedy with temperature","beam","top-k","top-p"] 
# for m in methods:
#     print("method = ",m)
#     evaluate(method = m,dataloader_val=dataloader_val)

## only print results on long texts
    
methods = ["greedy","greedy with temperature","beam","top-k","top-p"] 
for m in methods:
    print("method = ",m)
    #dataloader_val_long_text
    # evaluate(method = m,dataloader_val=dataloader_val_long_text) 
    ##  dataloader for the whole validation data
    evaluate(method = m,dataloader_val=dataloader_val) 

