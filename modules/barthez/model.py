import os
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import configs.barthez_config as base_config


def extract(passage, k):
    sentences = [sentence.strip() for sentence in passage.strip().split('.') if sentence]
    vectorizer = TfidfVectorizer(norm=None, use_idf=True, smooth_idf=True, sublinear_tf=False)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sentence_scores = np.ravel(
        np.sum(tfidf_matrix.multiply(tfidf_matrix != 0), axis=1) / np.sum(tfidf_matrix != 0, axis=1))
    top_k_indices = np.argsort(-sentence_scores)[:k]
    top_k_indices_sorted = sorted(top_k_indices)
    top_k_sentences = [sentences[i] for i in top_k_indices_sorted]
    return ". ".join(top_k_sentences)


class Barthez(nn.Module):
    def __init__(self, config=base_config):
        super(Barthez, self).__init__()

        self.name = "barthez"
        self.config = config

        self.mode_list = [
            "naive",
            "split"
        ]

        self.mode = config.DownStream.mode
        assert self.mode in self.mode_list

        self.tokenizer = AutoTokenizer.from_pretrained(config.Path.barthez)
        self.bart = AutoModelForSeq2SeqLM.from_pretrained(config.Path.barthez)
        self.cross_entropy = nn.CrossEntropyLoss()

    def encode(self, inputs):
        inputs_embeds = self.bart.model.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        return inputs_embeds

    def forward(self, samples):
        text = samples["text"]
        if self.config.DownStream.extract_before_generation:
            text = [extract(t, self.config.DownStream.k) for t in text]
        titles = samples["titles"]

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.config.device)
        titles = self.tokenizer(titles, return_tensors="pt", padding='max_length', truncation=True,
                                max_length=self.config.DownStream.title_len).to(self.config.device)

        outputs = None
        loss = None

        if self.mode == "naive":
            predict = self.bart(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                labels=titles['input_ids'])
            loss = predict.loss
            outputs = predict.logits
        elif self.mode == "split":
            inputs_embeds = self.encode(inputs)
            outputs = self.bart.model.decoder(inputs_embeds=inputs_embeds['last_hidden_state'],
                                              attention_mask=inputs['attention_mask'])['last_hidden_state']
            outputs = self.bart.lm_head(outputs)[:, :100]
            predicts = outputs.reshape(-1, outputs.shape[-1])
            labels = titles['input_ids'].reshape(-1,)
            loss = self.cross_entropy(predicts, labels)

        assert outputs is not None and loss is not None

        return loss, outputs

    def generate(self, text):
        if self.config.DownStream.extract_before_generation:
            text = [extract(t, self.config.DownStream.k) for t in text]
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.config.device)

        outputs = None

        if self.mode == "naive":
            outputs = self.bart.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                         max_length=100)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        elif self.mode == "split":
            inputs_embeds = self.encode(inputs)
            outputs = self.bart.generate(inputs_embeds=inputs_embeds['last_hidden_state'],
                                         attention_mask=inputs['attention_mask'],
                                         max_length=100)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        assert outputs is not None

        return outputs

    def save(self, epoch):
        if not os.path.exists(self.config.Path.save):
            os.mkdir(self.config.Path.save)
        ckpt_name = f"{self.name}_{self.mode}_{str(epoch)}.pth"
        save_path = os.path.join(self.config.Path.save, ckpt_name)
        print("Checkpoint saved at " + save_path)
        torch.save(self.state_dict(), save_path)

    def load(self, epoch):
        ckpt_name = f"{self.name}_{self.mode}_{str(epoch)}.pth"
        ckpt_path = os.path.join(self.config.Path.save, ckpt_name)
        self.load_state_dict(torch.load(ckpt_path))
