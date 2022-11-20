
#!/usr/bin/env python3
import boto3
import botocore
import sagemaker
import transformers
import datasets
from datasets import load_dataset
from datasets import load_from_disk
from datasets.filesystems import S3FileSystem
from transformers import AutoTokenizer
import datasets
import pandas as pd
import numpy as np
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
#    parser.add_argument("--file-name", type=str)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--star-threshold", type=int, default=3)
    args, _ = parser.parse_known_args()

    print("Received arguments {}".format(args))
    
    # read data
#    s3 = S3FileSystem() 
#    input_data_path ="/opt/ml/processing/input"
#    print("Reading input data from {}".format(input_data_path))
    squad = load_dataset("squad")
#    squad_s3 = load_from_disk(input_data_path)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )
        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []
        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)
        
            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1
        
            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
            
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)
                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs
    
    tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
    
    ###############################################################################################
    default_bucket = 'sagemaker-us-east-1-808242303800'
    s3_prefix = 'hf-small-tune'
    s3 = S3FileSystem() 
    
    # save train_dataset to s3
    training_input_path = f's3://{default_bucket}/{s3_prefix}/hf_data/train'
    tokenized_squad["train"].shuffle().select(range(30000)).save_to_disk(training_input_path,fs=s3)

    # save test_dataset to s3
    test_input_path = f's3://{default_bucket}/{s3_prefix}/hf_data/test'
    tokenized_squad["validation"].shuffle().select(range(5000)).save_to_disk(test_input_path,fs=s3)
