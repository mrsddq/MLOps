def train(
*,
input_model: str,
local_output_dir: str,
dbfs_output_dir: str,
epochs: int,
per_device_train_batch_size: int,
per_device_eval_batch_size: int,
lr: float,
seed: int,
deepspeed: str,
gradient_checkpointing: bool,
local_rank: str,
bf16: bool,
logging_steps: int,
save_steps: int,
eval_steps: int,
test_size: Union[float, int],
save_total_limit: int,
warmup_steps: int,
):
set_seed(seed)

model, tokenizer = get_model_tokenizer(
pretrained_model_name_or_path=input_model, gradient_checkpointing=gradient_checkpointing
)

# Use the same max length that the model supports. Fall back to 1024 if the setting can't be found.
# The configuraton for the length can be stored under different names depending on the model. Here we attempt
# a few possible names we've encountered.
conf = model.config
max_length = None
for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
max_length = getattr(model.config, length_setting, None)
if max_length:
logger.info(f"Found max lenth: {max_length}")
break
if not max_length:
max_length = 1024
logger.info(f"Using default max length: {max_length}")

processed_dataset = preprocess_dataset(tokenizer=tokenizer, max_length=max_length, seed=seed)

split_dataset = processed_dataset.train_test_split(test_size=test_size, seed=seed)

logger.info("Train data size: %d", split_dataset["train"].num_rows)
logger.info("Test data size: %d", split_dataset["test"].num_rows)

data_collator = DataCollatorForCompletionOnlyLM(
tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
)

if not dbfs_output_dir:
logger.warn("Will NOT save to DBFS")

training_args = TrainingArguments(
output_dir=local_output_dir,
per_device_train_batch_size=per_device_train_batch_size,
per_device_eval_batch_size=per_device_eval_batch_size,
fp16=False,
bf16=bf16,
learning_rate=lr,
num_train_epochs=epochs,
deepspeed=deepspeed,
gradient_checkpointing=gradient_checkpointing,
logging_dir=f"{local_output_dir}/runs",
logging_strategy="steps",
logging_steps=logging_steps,
evaluation_strategy="steps",
eval_steps=eval_steps,
save_strategy="steps",
save_steps=save_steps,
save_total_limit=save_total_limit,
load_best_model_at_end=False,
report_to="tensorboard",
disable_tqdm=True,
remove_unused_columns=False,
local_rank=local_rank,
warmup_steps=warmup_steps,
)

logger.info("Instantiating Trainer")

trainer = Trainer(
model=model,
tokenizer=tokenizer,
args=training_args,
train_dataset=split_dataset["train"],
eval_dataset=split_dataset["test"],
data_collator=data_collator,
)

logger.info("Training")
trainer.train()

logger.info(f"Saving Model to {local_output_dir}")
trainer.save_model(output_dir=local_output_dir)

if dbfs_output_dir:
logger.info(f"Saving Model to {dbfs_output_dir}")
trainer.save_model(output_dir=dbfs_output_dir)

logger.info("Done.")