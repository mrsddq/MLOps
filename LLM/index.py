import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import click
import numpy as np
from datasets import Dataset, load_dataset
from transformers import (
AutoModelForCausalLM,
AutoTokenizer,
DataCollatorForLanguageModeling,
PreTrainedTokenizer,
Trainer,
TrainingArguments,
set_seed,
)

def load_training_dataset(path_or_dataset: str = "databricks/databricks-dolly-15k") -> Dataset:
logger.info(f"Loading dataset from {path_or_dataset}")
dataset = load_dataset(path_or_dataset)["train"]
logger.info("Found %d rows", dataset.num_rows)

def _add_text(rec):
instruction = rec["instruction"]
response = rec["response"]
context = rec.get("context")

if not instruction:
raise ValueError(f"Expected an instruction in: {rec}")

if not response:
raise ValueError(f"Expected a response in: {rec}")

# For some instructions there is an input that goes along with the instruction, providing context for the
# instruction. For example, the input might be a passage from Wikipedia and the instruction says to extract
# some piece of information from it. The response is that information to extract. In other cases there is
# no input. For example, the instruction might be open QA such as asking what year some historic figure was
# born.
if context:
rec["text"] = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response, input=context)
else:
rec["text"] = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
return rec

dataset = dataset.map(_add_text)

return dataset

def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed=DEFAULT_SEED) -> Dataset:
"""Loads the training dataset and tokenizes it so it is ready for training.
Args:
tokenizer (AutoTokenizer): Tokenizer tied to the model.
max_length (int): Maximum number of tokens to emit from tokenizer.
Returns:
Dataset: HuggingFace dataset
"""

dataset = load_training_dataset()

logger.info("Preprocessing dataset")
_preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
dataset = dataset.map(
_preprocessing_function,
batched=True,
remove_columns=["instruction", "context", "response", "text", "category"],
)

# Make sure we don't have any truncated records, as this would mean the end keyword is missing.
logger.info("Processed dataset has %d rows", dataset.num_rows)
dataset = dataset.filter(lambda rec: len(rec["input_ids"]) < max_length)
logger.info("Processed dataset has %d rows after filtering for truncated records", dataset.num_rows)

logger.info("Shuffling dataset")
dataset = dataset.shuffle(seed=seed)

logger.info("Done preprocessing")

return dataset

