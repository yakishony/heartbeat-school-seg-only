from tracemalloc import start
from env import RATE

lEN_REC = 2
SAMPLES_NUM = int(LEN_REC * RATE)

def split_data_into_fixed_length_recordings(dataset):
    split_dataset = {}
    for rec_id, rec in dataset.items():
        num_iterations = len(rec['signal']) // SAMPLES_NUM
        for i in num_iterations:
            start_sample = i * SAMPLES_NUM
            end_sample = start_sample + SAMPLES_NUM

            y = rec['y'][start_sample:end_sample]
            if all(val == 0 for val in y):
                continue

            split_dataset[rec_id + f"_{i}"] = {
                'signal': rec['signal'][start_sample:end_sample],
                'y': y,
                'type': rec['type'],
            }
    return split_dataset
            