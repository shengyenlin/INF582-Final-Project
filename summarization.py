import time
import os
import argparse

import pandas as pd
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    AutoModelForSeq2SeqLM, AutoTokenizer,
    RobertaTokenizerFast, EncoderDecoderModel,
    SummarizationPipeline
)

MAX_LENGTH_DOWNSTREAM_MODEL = 512

device = "cuda" if torch.cuda.is_available() else "cpu"

class TextDataset(Dataset):
    def __init__(self, csv_file_path, model_name, tokenizer, max_length=512):
        self.dataframe = pd.read_csv(csv_file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_name = model_name

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe['text'][idx]
        if self.model_name == 'plguillou/t5-base-fr-sum-cnndm':
            inputs = self.tokenizer(
                "summarize: " + text, return_tensors="pt"
            )
            attention_mask = None
            return inputs.input_ids.squeeze(), attention_mask
        elif args.model_name == 'mrm8488/camembert2camembert_shared-finetuned-french-summarization':
            inputs = tokenizer(
                [text]
            )
        inputs = self.tokenizer(
            text,
            # padding="max_length",
            # truncation=True,
            # max_length=self.max_length,
            return_tensors="pt"
        )
        return inputs.input_ids.squeeze(), inputs.attention_mask.squeeze()

def parse_args():
    parser = argparse.ArgumentParser(description="Summarize text using GPT-2 model")
    # t5-base-fr-sum-cnndm 
    # mrm8488/camembert2camembert_shared-finetuned-french-summarization
    # lincoln/mbart-mlsum-automatic-summarization
    parser.add_argument("--model_name", type=str, help="Name of the model to use")
    parser.add_argument(
        "--max_text_len_of_downstream_model", type=int, 
        nargs='*',
        help="Max text length of the downstream model"
        )
    args = parser.parse_args()
    return args

def chunk_into_subarticles(text, sentences_per_subarticle=5):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Group sentences into subarticles of `sentences_per_subarticle` each
    subarticles = [
        ' '.join(sentences[i:i+sentences_per_subarticle])
        for i in range(0, len(sentences), sentences_per_subarticle)
    ]
    
    return subarticles

def summarize_text(text, model_name, max_text_len_of_downstream_model, tokenizer, model):

    len_text = tokenizer(text, return_tensors="pt").input_ids.shape[1]

    # only summarize if the text is longer than the maximum length of the downstream model
    if len_text > MAX_LENGTH_DOWNSTREAM_MODEL:
        text = chunk_into_subarticles(text, sentences_per_subarticle=7)
        text_input = ["summarize: " + t for t in text]

        if model_name == 'plguillou/t5-base-fr-sum-cnndm':
            input_ids = tokenizer(
                text_input, return_tensors="pt",
                padding=True, max_length = 512
            ).input_ids
            outputs = model.generate(
                input_ids,
                max_length=max_text_len_of_downstream_model,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                )

        elif model_name == 'mrm8488/camembert2camembert_shared-finetuned-french-summarization':
            inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_text_len_of_downstream_model,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                )
            
        outs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        out = ' '.join(outs)

    else:
        out = text

    print("lenght of text: ", len_text)
    print("summary: ", out)
    len_sum = tokenizer(out, return_tensors="pt").input_ids.shape[1]
    print("length of summary: ", len_sum)

    return out

def summarize_text_pipe(pipe, text):
    return pipe(text)['summary_text']

if __name__ == '__main__':

    args = parse_args()

    print(f"Use {args.model_name} to summarize text.")

    # load data
    train_df = pd.read_csv('data/train.csv')
    valid_df = pd.read_csv('data/validation.csv')
    test_df = pd.read_csv('data/test_text.csv')

    tqdm.pandas()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # if the data has more than 1024, then we do summarization
    print("Summarizing training data...")
    if args.model_name == 'lincoln/mbart-mlsum-automatic-summarization':
        pipe = SummarizationPipeline(model=model, tokenizer=tokenizer)

        print("Summarizing training data...")
        for i in args.max_text_len_of_downstream_model:
            
            train_df[f'text_sum_over_{i}'] = train_df['text'].progress_apply(
                lambda x: summarize_text_pipe(pipe, x) if len(x) > i else x)
            
        print("Summarizing validation data...")
        for i in args.max_text_len_of_downstream_model:
            valid_df[f'text_sum_over_{i}'] = valid_df['text'].progress_apply(
                lambda x: summarize_text_pipe(pipe, x) if len(x) > i else x)
            
        print("Summarizing test data...")
        for i in args.max_text_len_of_downstream_model:
            test_df[f'text_sum_over_{i}'] = test_df['text'].progress_apply(
                lambda x: summarize_text_pipe(pipe, x) if len(x) > i else x)
    else:
        for i in args.max_text_len_of_downstream_model:
            train_df[f'text_sum_over_{i}'] = train_df['text'].progress_apply(
                lambda x: summarize_text(
                    x, args.model_name, i, tokenizer, model
                    ) if len(x) > i else x)

        print("Summarizing validation data...")
        for i in args.max_text_len_of_downstream_model:
            valid_df[f'text_sum_over_{i}'] = valid_df['text'].progress_apply(
                lambda x: summarize_text(
                    x, args.model_name, i, tokenizer, model
                    ) if len(x) > i else x)
            
        print("Summarizing test data...")
        for i in args.max_text_len_of_downstream_model:
            test_df[f'text_sum_over_{i}'] = test_df['text'].progress_apply(
                lambda x: summarize_text(
                    x, args.model_name, i, tokenizer, model
                    ) if len(x) > i else x)

    # make directory
    os.makedirs(f'data/{args.model_name}', exist_ok=True)

    # Save data
    train_df.to_csv(f'data/{args.model_name}/train_sum.csv', index=False)
    valid_df.to_csv(f'data/{args.model_name}/validation_sum.csv', index=False)
    test_df.to_csv(f'data/{args.model_name}/test_text_sum.csv', index=False)