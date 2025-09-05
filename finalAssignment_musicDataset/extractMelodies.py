# %%
import os
from os import path
import numpy as np
from pathlib import Path
# Define the source directory and the target directory
source_dir = './BiMMuDa/'  # Current directory
target_dir = 'datasets/BiMMuDa/'

# Create the target directory if it doesn't exist
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Walk through all subdirectories
midi_files, file_paths = set(), list()
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith('_full.mid'):
            midi_files.add(Path(root) / file)

for file in sorted(midi_files):
    file_paths.append(new_path := Path(target_dir).resolve() / file.name)
    os.makedirs(new_path.parent, exist_ok=True)
    os.system(f'cp "{file}" "{new_path}"')

print("All files ending in '_full.mid' have been copied to the 'musicDataset' directory.")
print(len(midi_files))
paths = list(sorted(file_paths))

if paths:
    output_dir = path.dirname(paths[0])
    indices = np.random.permutation(len(paths))
    split = int(len(paths) * 0.1)
    train_paths = [paths[i] for i in indices[split:]]
    val_paths = [paths[i] for i in indices[:split]]
else:
    raise ValueError("No MIDI files found in the source directory.")

# %%
print(paths[:10])                       

# %%
# Seed
from miditok import REMI, TokenizerConfig


# Our tokenizer's configuration
BEAT_RES = {(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1}
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": BEAT_RES,
    "num_velocities": 24,
    "special_tokens": ["PAD", "BOS", "EOS"],
    "use_chords": True,
    "use_rests": True,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_programs": False,  # no multitrack here
    "num_tempos": 32,
    "tempo_range": (40, 250),  # (min_tempo, max_tempo)
}
config = TokenizerConfig(**TOKENIZER_PARAMS)
#vocab_size = 6000
# Creates the tokenizer
tokenizer = REMI(config)
# tokenizer._load_from_json
# Trains the tokenizer with Byte Pair Encoding (BPE) to build the vocabulary, here 30k token
# tokenizer.train(
#     vocab_size=vocab_size,
#     files_paths=paths,
# )
vocab_size = tokenizer.vocab_size
print(tokenizer.vocab_size)
tokenizer.save_params(f"tokenizer_{vocab_size}.json")

# %%
from miditok.utils import split_files_for_training
from miditok.data_augmentation import augment_dataset
max_len = 256
# Chunk MIDIs and perform data augmentation on each subset independently
for files_paths, subset_name in (
    (train_paths, "train"), (val_paths, "valid")
):

    # Split the MIDIs into chunks of sizes approximately about vocab_size tokens
    subset_chunks_dir = Path(f"Token_{subset_name}_{max_len}_{vocab_size}")
    split_files_for_training(
        files_paths=files_paths,
        tokenizer=tokenizer,
        save_dir=subset_chunks_dir,
        max_seq_len=max_len,
        num_overlap_bars=2,
    )

    # Perform data augmentation
    augment_dataset(
        subset_chunks_dir,
        pitch_offsets=[-12, 12],
        velocity_offsets=[-4, 4],
        duration_offsets=[-0.5, 0.5],
    )


