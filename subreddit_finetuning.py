import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM
import pandas as pd
from trl import SFTConfig, SFTTrainer


def fine_tune_model():
    print("extracting data")
    data_dict = get_data()

    # set configurations
    sft_config = SFTConfig(
        dataset_text_field="content",
        max_seq_length=2048,
        output_dir="finetuned",
        learning_rate=3e-05,
        lr_scheduler_type="cosine",
        num_train_epochs=5,
        per_device_train_batch_size=8,
        bf16=True,
        logging_steps=100,
    )

    print("loading model")
    # load model
    model = AutoModelForCausalLM.from_pretrained(
        "/scratch/bchk/aguha/models/llama3p2_1b_base",
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2"
    ).to("cuda")
    
    # initialize trainer
    trainer = SFTTrainer(
        model,
        train_dataset=data_dict["train"],
        eval_dataset=data_dict["test"],
        args=sft_config,
    )
    
    print("starting training")
    # train model
    trainer.train()

    return model


def get_data() -> dict:
    # load generated question-answer pairs and shuffle
    df = pd.read_csv('csv_files/subreddit_finetuning_questions.csv')
    df = df.sample(frac=1)

    # transform dataframe into dataset
    data_dict = Dataset.from_pandas(
        df
    ).train_test_split(
        0.01
    ).map(
        format_item
    ).remove_columns(
        ["question", "college"]
    )
    
    return data_dict


# format question-answer pair into content to be fed in to model
def format_item(item) -> dict:
    q = item["question"].strip()
    c = item["college"].strip()
    return { "content": f"Question: {q}\nCollege: {c}" }
