import gc

from env import RATE, RATE_DS
from download_data_and_annotate_step1 import load_dataset_raw
from understand_data import plot_category_pie

LEN_REC = 2
SPLIT_SAMPLES = int(LEN_REC * RATE)     # for splitting raw data
SAMPLES_NUM = int(LEN_REC * RATE_DS)    # after downsampling, used by ML.py

def split_data_into_fixed_length_recordings(dataset:{}): 
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
            if (y == 0).all():
                count_full_0_splits += 1
                continue

            split_dataset[rec_id + f"_{i}"] = {
                'signal': rec['signal'][start_sample:end_sample],
                'y': y,
                'type': rec['type'],
            }
    print("Split dataset length:", len(split_dataset))
    print("Count deleted recordings:", count_deleted_recordings)
    print("Count full 0 splits:", count_full_0_splits)
    return split_dataset
            
def split_data_into_fixed_length_recordings_without_unrecognized(dataset:{}):
    """Split the dataset into fixed length recordings without unrecognized.
    every recoding with LEN_REC wont have above LEN_REC/2 unrecognized labeled timestamps """
    split_dataset = {}
    count_deleted_splits = 0

    for rec_id, rec in dataset.items():
        sampels_num = len(rec['signal'])
        if sampels_num < SPLIT_SAMPLES:
            continue
        c_sample_index = 0
        c_label = rec['y'][c_sample_index]
        i = 0
        while c_sample_index < sampels_num-1:
            while c_label == 0 and c_sample_index < sampels_num-1:
                c_sample_index += 1
                c_label = rec['y'][c_sample_index]
            if c_label == 0:
                break
            start_sample_index = c_sample_index
            end_sample_index = start_sample_index + SPLIT_SAMPLES

            if end_sample_index >= sampels_num-1: # if the split is the last
                end_sample_index = sampels_num - 1
                start_sample_index = end_sample_index - SPLIT_SAMPLES

            # check if more than half of the split is unrecognized
            unreconized_count = (rec['y'][start_sample_index:end_sample_index] == 0).sum()

            c_sample_index = end_sample_index
            c_label = rec['y'][c_sample_index]

            if unreconized_count > SPLIT_SAMPLES/2:
                continue
            split_dataset[rec_id + f"_{i}"] = {
                'signal': rec['signal'][start_sample_index:end_sample_index],
                'y': rec['y'][start_sample_index:end_sample_index],
                'type': rec['type'],
            }
            i += 1

        count_samples_that_got_into_new_dataset = i * SPLIT_SAMPLES
        count_samples_that_didnt_got_into_new_dataset = sampels_num - count_samples_that_got_into_new_dataset
        count_deleted_splits_for_this_recording = count_samples_that_didnt_got_into_new_dataset // SPLIT_SAMPLES
        count_deleted_splits += count_deleted_splits_for_this_recording
    print("Split unrecognized dataset length:", len(split_dataset))
    print("Count deleted splits for unrecognized:", count_deleted_splits)

    return split_dataset



def run():
    dataset_raw, missing = load_dataset_raw()
    print(f"Loaded {len(dataset_raw)} recordings ({len(missing)} missing annotations)")
    dataset_split_without_unreconized = split_data_into_fixed_length_recordings_without_unrecognized(dataset_raw)
    dataset_split = split_data_into_fixed_length_recordings(dataset_raw)
    del dataset_raw
    gc.collect()
    plot_category_pie(dataset_split_without_unreconized, name="fig_category_pie_with_spliting_method_of_less_unrecognized_samples")
    plot_category_pie(dataset_split, name="fig_category_pie_with_regular_spliting_method")
  

if __name__ == "__main__":
    run()
