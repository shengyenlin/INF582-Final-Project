import time

import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from rouge_score import rouge_scorer

# env_name: nlp-final

device = "cuda" if torch.cuda.is_available() else "cpu"

prompt_templates = {
    "PAGnol": "{} Summary: ",
}

MAX_LENGHT_TOKENIZER = 1024

def process_for_output_exceed_max_length(beam_output, tokenizer):
    # Chunk the output into smaller pieces and decode
    outputs = []
    for i in range(0, len(beam_output), MAX_LENGHT_TOKENIZER):
        outputs.append(tokenizer.decode(beam_output[i:i+MAX_LENGHT_TOKENIZER], skip_special_tokens=True))

    return " ".join(outputs)

def generate_summary(text: pd.core.series.Series, model, tokenizer, prompt_template):
    model.eval()
    summaries = []
    x = time.time()
    for idx, row in text.items():
        prompt = prompt_template.format(row)
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        print("input length: ", len(input_ids[0]))
        beam_outputs = model.generate(
            input_ids, 
            # max_length=100,
            max_new_tokens=500, # excluding input tokens
            do_sample=True,   
            top_k=50, 
            top_p=0.95, 
            num_return_sequences=1
        )

        print("output length: ", len(beam_outputs[0]))
        if len(beam_outputs[0]) > MAX_LENGHT_TOKENIZER:
            nl_output = process_for_output_exceed_max_length(beam_outputs[0], tokenizer)
        else:
            nl_output = tokenizer.decode(beam_outputs[0], skip_special_tokens=True)
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

model_name = "asi/gpt-fr-cased-small"

if __name__ == '__main__':

    # TODO: add arg parser - model name, prompt templage

    # Load data
    # train_df = pd.read_csv('data/train.csv')
    validation_df = pd.read_csv('data/validation.csv')
    test_df = pd.read_csv('data/test_text.csv')

    # Load pretrained model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(
        model_name
        # "asi/gpt-fr-cased-base" # Doesn't perform better on OrangeSum Dataset
        # Chemsseddine/bert2gpt2SUMM-finetuned-mlsum-finetuned-mlorange_sum
        # moussaKam/mbarthez
        # moussaKam/barthez-orangesum-abstract
        # moussaKam/barthez-orangesum-title
        )
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # TODO: choose prompt template
    prompt_template = prompt_templates['PAGnol']

    print("Start to generate summaries.")
    print("Number of validation samples: ", len(validation_df))
    print("Number of test samples: ", len(test_df))
          
    # Generate summary
    summaries = generate_summary(
        validation_df['text'], model, 
        tokenizer, prompt_template
        )

    # Calculate ROUGE-L Score
    scorer = rouge_scorer.RougeScorer(['rougeL'])
    rouge_score = evaluate_summaries(
        summaries, validation_df['titles'], scorer
        )
    
    print(f"Average Rouge-L F-Score with {model_name}: ", rouge_score)
