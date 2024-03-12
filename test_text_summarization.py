import time
import os
import argparse

import pandas as pd
from tqdm import tqdm

import torch
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel, 
    AutoTokenizer, AutoModelForSeq2SeqLM
)
from rouge_score import rouge_scorer

device = "cuda" if torch.cuda.is_available() else "cpu"

prompt_templates = {
    "PAGnol": "{} Summary: ",
}

model_name = "moussaKam/barthez-orangesum-title"
# Chemsseddine/bert2gpt2SUMM-finetuned-mlsum-finetuned-mlorange_sum
# moussaKam/mbarthez
# moussaKam/barthez-orangesum-abstract
# moussaKam/barthez-orangesum-title
# moussaKam/mbarthez-dialogue-summarization

MAX_LENGHTH_MODEL = 1024

def parse_args():
    parser = argparse.ArgumentParser(description="Summarize text using GPT-2 model")
    parser.add_argument("--model_name", type=str, default="moussaKam/barthez-orangesum-title", help="Name of the model to use")
    parser.add_argument("--prompt_template", type=str, default="PAGnol", help="Prompt template to use")
    parser.add_argument("--do_validation", action="store_true", help="Whether to do validation")
    parser.add_argument("--do_test", action="store_true", help="Whether to do test")
    parser.add_argument("--out_valid_df_name", type=str, default="valid_submission.csv", help="Name of the valid output dataframe")
    parser.add_argument("--out_test_df_name", type=str, default="test_submission.csv", help="Name of the test output dataframe")
    args = parser.parse_args()
    return args

def process_for_output_exceed_max_length(beam_output, tokenizer):
    # Chunk the output into smaller pieces and decode
    outputs = []
    for i in range(0, len(beam_output), MAX_LENGHTH_MODEL):
        outputs.append(tokenizer.decode(beam_output[i:i+MAX_LENGHTH_MODEL], skip_special_tokens=True))

    return " ".join(outputs)

def generate_summary(text: pd.core.series.Series, model, tokenizer, prompt_template):
    model.to(device)
    model.eval()
    summaries = []
    x = time.time()
    for idx, row in tqdm(text.items()):

        # for asi/gpt-fr-cased-base
        # prompt = prompt_template.format(row)
        # input_ids = tokenizer.encode(prompt, return_tensors='pt')

        # print(row)

        input_ids = torch.tensor(
            [tokenizer.encode(row, add_special_tokens=True)]
        )
        input_ids = input_ids.to(device)

        if len(input_ids[0]) > MAX_LENGHTH_MODEL:
            # truncate to max length
            input_ids = input_ids[:, :MAX_LENGHTH_MODEL]

        beam_outputs = model.generate(
            input_ids, 
            max_length=100,
            # max_new_tokens=500, # excluding input tokens
            # do_sample=True,   
            # top_k=50, 
            # top_p=0.95, 
            # num_return_sequences=1
        )

        print("output length: ", len(beam_outputs[0]))
        # if len(beam_outputs[0]) > MAX_LENGHT_TOKENIZER:
        #     nl_output = process_for_output_exceed_max_length(beam_outputs[0], tokenizer)
        # else:
        nl_output = tokenizer.decode(beam_outputs[0], skip_special_tokens=True)
        print(nl_output)
        time_elapsed = round((time.time() - x)/60, 3)
        print("Processing row: ", idx + 1, "time elapsed: ", time_elapsed, "minutes")
        summaries.append([idx, nl_output])
    return summaries

def evaluate_summaries(summaries: pd.core.series.Series, titles: pd.core.series.Series, scorer: rouge_scorer.RougeScorer):
    rouge_scores = []
    for idx, title in titles.items():
        rouge_scores.append(scorer.score(summaries[idx][1], title)['rougeL'][2])
    rouge_score = sum(rouge_scores) / len(rouge_scores)
    return rouge_score

if __name__ == '__main__':

    args = parse_args()

    # Load data
    validation_df = pd.read_csv('data/validation.csv')
    test_df = pd.read_csv('data/test_text.csv')

    # Load pretrained model and tokenizer
    # model = GPT2LMHeadModel.from_pretrained("asi/gpt-fr-cased-base") # Doesn't perform better on OrangeSum Dataset
    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # TODO: choose prompt template
    prompt_template = prompt_templates[args.prompt_template]
          
    # Generate summary on validation set
    if args.do_validation:
        print("Start generating summaries on validation set...")
        print("Number of validation samples: ", len(validation_df))
        valid_summaries = generate_summary(
            validation_df['text'], model, 
            tokenizer, prompt_template
            )

        # Calculate ROUGE-L Score
        scorer = rouge_scorer.RougeScorer(['rougeL'])
        rouge_score = evaluate_summaries(
            valid_summaries, validation_df['titles'], scorer
            )
        
        os.makedirs('results_valid', exist_ok=True)
        valid_df = pd.DataFrame(valid_summaries, columns=['ID', 'titles'])
        valid_df.to_csv(f'results_valid/submission_{args.out_valid_df_name}', index=False)

        print(f"Average Rouge-L F-Score on validation set with {model_name}: ", rouge_score)

    # Predict on test set
    if args.do_test:
        print("Start generating summaries on test set...")
        print("Number of test samples: ", len(test_df))
        test_summaries = generate_summary(
            test_df['text'], model, 
            tokenizer, prompt_template
            )
        
        os.makedirs('results_test', exist_ok=True)

        test_df = pd.DataFrame(test_summaries, columns=['ID', 'titles'])
        test_df.to_csv(f'results_test/submission_{args.out_test_df_name}.csv', index=False)