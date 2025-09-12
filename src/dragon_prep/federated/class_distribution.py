import json
import re
from pathlib import Path

import pandas as pd
from dragon_eval.evaluation import TASK_TYPE


def get_dataset_distribution_dict(development_dir: Path, test_dir: Path):
    validation_files = list(development_dir.glob("Task*/nlp-validation-dataset.json"))
    training_files = list(development_dir.glob("Task*/nlp-training-dataset.json"))
    test_files = list(test_dir.glob("Task*.json"))
    def parse_split(file_list, split):
        df_dict = {
        }
        for file in file_list:
            if match := re.search(r"(Task\d+_[a-zA-Z_]+)-fold(\d+)", file.as_posix()):
                task_name, fold = match.groups()
            elif match := re.search(r"(Task\d+_[a-zA-Z_]+)", file.as_posix()):
                task_name = match.group(1)
                fold = 'test'
            else:
                raise ValueError(f"File {file} does not match expected pattern")

            task_number = str(int(task_name.split("_")[0].replace("Task", "")))

            if task_number not in df_dict:
                df_dict[task_number] = []
            df_dict[task_number].append(pd.read_json(file).assign(task=task_name, fold=fold, task_number=task_number, split=split))
        for key in df_dict.keys():
            df_dict[key] = pd.concat(df_dict[key], ignore_index=True)

        return df_dict

    def merge_df_dicts(df_dicts):
        merged_df_dict = {}
        for d in df_dicts:
            for key, df in d.items():
                if key not in merged_df_dict:
                    merged_df_dict[key] = df
                else:
                    merged_df_dict[key] = pd.concat([df, merged_df_dict[key]], ignore_index=True)
        return merged_df_dict

    train_df_dict = parse_split(training_files, 'training')
    val_df_dict = parse_split(validation_files, 'validation')
    test_df_dict = parse_split(test_files, 'test')
    df_dict = merge_df_dicts([train_df_dict, val_df_dict, test_df_dict])

    return df_dict


def collect_center_and_class_distribution(df_dict) -> dict[str, pd.DataFrame]:
    """Collect center and class distribution from the datasets for
    classification tasks. For regressions tasks selected percentiles per
    center are collected."""
    for task_number in df_dict.keys():
        task_df = df_dict[task_number]
        target_col = [col for col in task_df.columns if col.endswith("target")][0]
        if 'regression'.upper() in TASK_TYPE[task_df.task[0]].name.upper():
            center_stats = task_df.groupby(['task', 'fold', 'center', 'split'])[target_col].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).reset_index()
        else:
            center_stats = pd.DataFrame(task_df.explode(target_col).groupby(['task', 'fold', 'center', 'split', target_col])['uid'].count()).rename(columns={'uid': 'count'}).reset_index()
            center_stats = center_stats.rename(columns={target_col: 'target'})
        df_dict[task_number] = center_stats
    return df_dict


def convert_to_json_serializable(class_balance_dict):
    json_serializable_dict = {}
    for task, df in class_balance_dict.items():
        json_serializable_dict[task] = df.to_dict(orient='records')
    return json_serializable_dict


def from_json_serializable(json_serializable_dict):
    class_balance_dict = {}
    for task, records in json_serializable_dict.items():
        class_balance_dict[task] = pd.DataFrame.from_records(records)
    return class_balance_dict

input_dir = Path('/Users/joeranbosma/repos/dragon_data/data/preprocessed/federated')
output_dir = Path('/Users/joeranbosma/repos/dragon_data/data/preprocessed/federated/class_balance')
for dev_dir, test_dir, outp_dir in [
    (input_dir / 'debug-input', input_dir / "debug-test-set", output_dir / 'validation'),
    (input_dir / 'algorithm-input', input_dir / "test-set", output_dir / 'test'),
]:
    outp_dir.mkdir(parents=True, exist_ok=True)
    df_dict = get_dataset_distribution_dict(development_dir=dev_dir, test_dir=test_dir)
    class_balance_dict = collect_center_and_class_distribution(df_dict)

    # save dataframes as CSV files
    for task, df in class_balance_dict.items():
        df.to_csv(outp_dir / f'task_{task}_class_balance.csv', index=False)

    class_balance_dict_json = convert_to_json_serializable(class_balance_dict)
    with open(outp_dir / 'class_balance.json', 'w') as f:
        json.dump(class_balance_dict_json, f)
