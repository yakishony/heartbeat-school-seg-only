from tracemalloc import start
from env import RATE, RATE_DS

LEN_REC = 2
SPLIT_SAMPLES = int(LEN_REC * RATE)     # for splitting raw data
SAMPLES_NUM = int(LEN_REC * RATE_DS)    # after downsampling, used by ML.py

def split_data_into_fixed_length_recordings(dataset):
    split_dataset = {}
    count_full_0_splits = 0
    count_deleted_recordings = 0

    for rec_id, rec in dataset.items():
        num_iterations = len(rec['signal']) // SPLIT_SAMPLES

        if num_iterations == 0:
            count_deleted_recordings += 1
            continue

        for i in range(num_iterations):
            start_sample = i * SPLIT_SAMPLES
            end_sample = start_sample + SPLIT_SAMPLES

            y = rec['y'][start_sample:end_sample]
            if all(val == 0 for val in y):
                count_full_0_splits += 1
                continue

            split_dataset[rec_id + f"_{i}"] = {
                'signal': rec['signal'][start_sample:end_sample],
                'y': y,
                'type': rec['type'],
                'murmur': rec['murmur'],
            }
    print("Split dataset length:", len(split_dataset))
    print("Count deleted recordings:", count_deleted_recordings)
    print("Count full 0 splits:", count_full_0_splits)
    return split_dataset, count_deleted_recordings, count_full_0_splits
            