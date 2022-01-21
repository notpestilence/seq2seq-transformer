import pathlib
import random
import string
import re
import numpy as np

# Importing our translations
data_path = "ind.txt"
# Defining lines as a list of each line
with open(data_path, 'r', encoding='utf-8') as f:
  lines = f.read().split('\n')[:-1]
text_pairs = []
for line in lines:
  eng, ind, _ = line.split("\t")
  ind = "<START> " + ind + " <END>"
  text_pairs.append((eng, ind))

def get_pairs(pairs = text_pairs):
    random.shuffle(pairs)
    num_val_samples = int(0.15 * len(pairs)) # Separate 15% for validation
    num_train_samples = len(pairs) - 2 * num_val_samples
    train_pairs = pairs[:num_train_samples]
    val_pairs = pairs[num_train_samples : num_train_samples + num_val_samples]
    test_pairs = pairs[num_train_samples + num_val_samples :]
    return train_pairs, val_pairs, test_pairs




