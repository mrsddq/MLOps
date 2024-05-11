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