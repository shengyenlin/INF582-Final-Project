
import pandas as pd
from torch.utils.data import DataLoader
import torch
from datasets import Dataset
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
tokenizer = T5Tokenizer.from_pretrained("weny22/sum_model_lr1e_3_20epoch")
model = T5ForConditionalGeneration.from_pretrained('weny22/sum_model_lr1e_3_20epoch')
## with extract
# tokenizer = T5Tokenizer.from_pretrained("weny22/sum_model_3r1e_3_20_with_extract")
# model = T5ForConditionalGeneration.from_pretrained('weny22/sum_model_3r1e_3_20_with_extract')
## only extract the long text
# tokenizer = T5Tokenizer.from_pretrained("weny22/sum_model_2r1e_3_20_extract_long_text")
# model = T5ForConditionalGeneration.from_pretrained('weny22/sum_model_2r1e_3_20_extract_long_text')
# ## google t5
# tokenizer = T5Tokenizer.from_pretrained("google/mt5-small")
# model = T5ForConditionalGeneration.from_pretrained('google/mt5-small')

def extractive_summary(text, num_sentences=6):
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
test_df = pd.read_csv('data/test_text.csv')
test_df['extract_text'] = test_df['text'] ## just to copy the columns
test_ds = Dataset.from_pandas(test_df)
# print(test_ds)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set up the device for training
model.to(device)
   
def tokenize_function(examples):
    inputs = tokenizer(examples["extract_text"], padding="max_length", truncation=True, max_length=512)
    return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}

def collate_fn(batch):
    # Extract individual elements from each sample in the batch
    # print(batch)
    idx = [sample['ID'] for sample in batch]
    texts = [sample['text'] for sample in batch]
    # titles = [sample['titles'] for sample in batch]
    # print([(sample['text'],sample['input_ids']) for sample in batch])
    input_ids = [sample['input_ids'] for sample in batch]
    attention_mask = [sample['attention_mask'] for sample in batch]
    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    # Return a dictionary containing all the elements
    return {
        'idx':idx,
        'text': texts,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }

batch_size = 32

# print("model.config",model.config)
# print("tokenizer.vocab_size",tokenizer.vocab_size)

# optimizer = torch.optim.Adam(params = model.parameters(),
#             lr=0.0001,
#             betas=(0.9, 0.999),
#             eps=1e-08,)

tokenized_test_dataset = test_ds.map(tokenize_function, batched=True)
dataloader_test =  DataLoader(tokenized_test_dataset , batch_size=batch_size, collate_fn=collate_fn, shuffle=False) ## don't shuffle the order

# Load the ROUGE metric
import evaluate
# rouge = load_metric("rouge")
rouge = evaluate.load("rouge")

# Initialize lists to store predictions and targets
predictions = []
indexes = [] ## the default decoding method

temperature =0.3
num_beams = 4
top_k =50
top_p =0.9
max_new_tokens = 50

def write_output(method = None):
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader_test):
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
            indexes.extend(batch["idx"])
        output_df = pd.DataFrame({'ID': indexes, 'titles':predictions})
        print(output_df.shape)
        output_df.to_csv('output.csv', index=False, sep=',')


# Write the data to a CSV file with comma as the delimiter
## methods =  ["greedy","greedy with temperature","beam","top-k","top-p"] 
write_output(method = None) ## write the default generation to a csv file
# write_output(method = "greedy with temperature") ## "greedy with temperature"
# write_output(method = "top-p") ## top-p