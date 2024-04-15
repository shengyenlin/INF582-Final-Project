import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Path:
    data = "data/"
    save = "ckpt/barthez"
    barthez = "moussaKam/barthez"


class DownStream:
    title_len = 100
    mode = "naive"
    extract_before_generation = False
    k = 8

    lr = 2e-5
    batch_size = 8
    epochs = 20
    decay = 1e-4
