import torch
import evaluate
from tqdm import tqdm

rouge = evaluate.load("rouge")


def evaluate(model, val_loader):
    model.eval()
    with torch.no_grad():
        bar = tqdm(val_loader)
        gen = []
        golden_titles = []
        for _, sample in enumerate(bar):
            outputs = model.generate(sample["text"])
            gen.extend(outputs)
            golden_titles.extend(sample["titles"])
        result = rouge.compute(predictions=gen, references=golden_titles, use_stemmer=True)
        return {k: round(v, 4) for k, v in result.items()}


def fit(model, config, train_loader, val_loader):
    model.train()

    total_epochs = config.DownStream.epochs
    lr = config.DownStream.lr
    decay = config.DownStream.decay

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)

    metrics = evaluate(model, val_loader)
    print(f"Before training: {metrics}")

    for epoch in range(1, total_epochs + 1):
        model.train()
        loss = torch.tensor(0)
        total_loss = 0
        bar = tqdm(train_loader)
        for _, sample in enumerate(bar):
            bar.set_description(f"Epoch {epoch}| Loss {loss.item()}")
            try:
                optimizer.zero_grad()
                loss, _ = model(sample)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            except RuntimeError:
                print(f"RuntimeError encountered, problematic inputs: {sample}")
        metrics = evaluate(model, val_loader)
        print(f"Epoch {epoch}: {metrics}| Total Loss: {total_loss}")
        model.save(epoch)

    return model
