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