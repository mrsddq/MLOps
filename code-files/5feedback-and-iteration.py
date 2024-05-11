from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# load pre-trained language model and tokenizer
model_name = "microsoft/CodeGPT-small-java"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# prepare data for fine-tuning
train_dataset = ...

# fine-tune the model
training_args = TrainingArguments(
output_dir='./results',
evaluation_strategy = "epoch",
learning_rate=2e-5,
per_device_train_batch_size=16,
per_device_eval_batch_size=64,
num_train_epochs=1,
weight_decay=0.01,
push_to_hub=False,
)
trainer = Trainer(
model=model,
args=training_args,
train_dataset=train_dataset,
data_collator=lambda data: {'input_ids': tokenizer(data['code'], padding=True, truncation=True, max_length=512).input_ids, 'labels': tokenizer(data['code'], padding=True, truncation=True, max_length=512).input_ids},
)
trainer.train()