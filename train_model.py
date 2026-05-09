import pandas as pd
import json
import glob

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


def load_data():
    texts = []

    for file in glob.glob("data/*.json"):

        with open(file, encoding='utf-8') as f:

            data = json.load(f)

            if isinstance(data, list):

                for item in data:

                    if 'text' in item:
                        texts.append(item['text'])

    df = pd.DataFrame(
        texts,
        columns=['text']
    )

    def label(text):

        text = text.lower()

        if 'ужас' in text:
            return 2

        if 'супер' in text:
            return 1

        return 0

    df['label'] = df['text'].apply(label)

    return df


def train():

    df = load_data()

    model_name = "cointegrated/rubert-tiny-sentiment-balanced"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )

    dataset = Dataset.from_pandas(df)

    def tokenize(x):
        return tokenizer(
            x['text'],
            truncation=True,
            padding=True
        )

    dataset = dataset.map(
        tokenize,
        batched=True
    )

    dataset = dataset.train_test_split(
        test_size=0.2
    )

    args = TrainingArguments(
        output_dir="./model",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test']
    )

    trainer.train()

    model.save_pretrained("./model")

    tokenizer.save_pretrained("./model")


if __name__ == "__main__":
    train()
