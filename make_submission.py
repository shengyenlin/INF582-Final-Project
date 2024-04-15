import torch
from tqdm import tqdm
from modules.barthez import Barthez
import pandas as pd
from dataloaders.base_dataloader import base_dataloader
from configs import barthez_config

MODEL_LIST = {
    "barthez": {
        "model": Barthez,
        "config": barthez_config
    }
}


def test(model, test_loader):
    with torch.no_grad():
        summaries = []
        bar = tqdm(test_loader)
        for _, batch in enumerate(bar):
            outputs = model.generate(batch["text"])
            summaries.extend(outputs)
        return summaries


def make_submission(model_name, epoch):
    model = MODEL_LIST[model_name]["model"]()
    model_config = MODEL_LIST[model_name]["config"]
    model.to(model_config.device)
    model.eval()
    model.load(epoch=epoch)

    test_loader = base_dataloader(32, False, split="test", config=model_config)
    summaries = test(model, test_loader)
    submission = pd.DataFrame([[idx, summary] for idx, summary in enumerate(summaries)], columns=['ID', 'titles'])
    submission_name = f"submission_{model_name}_{epoch}.csv"
    submission.to_csv(submission_name, index=False)


if __name__ == "__main__":
    make_submission("barthez", 8)
