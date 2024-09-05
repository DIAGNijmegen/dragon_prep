#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse
from pathlib import Path
from typing import Union

from dragon_prep.prepare_rumc import prepare_rumc_lung_pathology_reports
from dragon_prep.utils import (num_patients, prepare_for_anon, read_anon,
                               split_and_save_data)


def preprocess_reports(
    task_name: str,
    input_dir: Union[Path, str],
    output_dir: Union[Path, str],
):
    # prepare marksheet
    df = prepare_rumc_lung_pathology_reports(input_dir=input_dir)

    # set label
    df["label"] = df["Sample origin"]

    # exclude cases with multiple tissue types by excluding cases with roman numerals in the tissue type
    df = df[~df["label"].str.contains(r"\b[IVX]+\b", regex=True)]
    print(f"Have {len(df)} cases from {num_patients(df)} patients after excluding cases with multiple tissue types")

    # normalize labels
    df["label"] = df["label"].str.strip()
    df["label"] = df["label"].str.lower()

    # pool infrequent labels (less than 10 cases) into "Other"
    label_counts = df["label"].value_counts()
    infrequent_labels = label_counts[label_counts < 10].index.tolist()
    df["label"] = df["label"].replace(infrequent_labels, "other")

    # rename columns
    df = df.rename(columns={"label": "single_label_multi_class_classification_target"})
    print(df["single_label_multi_class_classification_target"].value_counts())

    # prepare for anonynimization
    assert df.text.apply(lambda text: "<PERSOON>" in text).sum() > 800, f"Unexpected number of <PERSOON> tags: {df.text.apply(lambda text: '<PERSOON>' in text).sum()}"
    assert df.text.apply(lambda text: "<DATUM>" in text).sum() > 800, f"Unexpected number of <DATUM> tags: {df.text.apply(lambda text: '<DATUM>' in text).sum()}"
    prepare_for_anon(df=df, output_dir=output_dir, task_name=task_name, tag_phi=False, apply_hips=True)


def prepare_reports(
    task_name: str,
    output_dir: Union[Path, str],
    test_split_size: float = 0.3,
):
    # read anonynimized data
    df = read_anon(output_dir / "anon" / task_name / "nlp-dataset.json")

    # make test and cross-validation splits
    split_and_save_data(
        df=df,
        output_dir=output_dir,
        task_name=task_name,
        test_split_size=test_split_size,
        split_by="patient_id",
        recommended_truncation_side="right",
    )


if __name__ == "__main__":
    # create the parser
    parser = argparse.ArgumentParser(description="Script for preparing reports")
    parser.add_argument("--task_name", type=str, default="Task013_pathology_tissue_origin_clf",
                        help="Name of the task")
    parser.add_argument("-i", "--input", type=Path, default="/input",
                        help="Path to the input data")
    parser.add_argument("-o", "--output", type=Path, default=Path("/output"),
                        help="Folder to store the prepared reports in")
    parser.add_argument("--test_split_size", type=float, default=0.3,
                        help="Fraction of the dataset to use for testing")
    args = parser.parse_args()

    # run preprocessing
    preprocess_reports(
        task_name=args.task_name,
        input_dir=args.input,
        output_dir=args.output,
    )
    prepare_reports(
        task_name=args.task_name,
        output_dir=args.output,
        test_split_size=args.test_split_size,
    )
