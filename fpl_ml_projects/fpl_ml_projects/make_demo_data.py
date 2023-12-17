from ml_core.data_splitter import randomly_split_data
from sklearn.datasets import load_diabetes
import os


def make_diabetes_dataset(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    dataset = load_diabetes(return_X_y=False, as_frame=True)
    data_df = dataset["data"]

    # Format feature column names and add prefix
    orig_data_columns = data_df.columns
    rename_dict = {}
    for c in orig_data_columns:
        rename_dict[c] = "X_" + c.replace(" ", "_")

    # Add target column with prefix
    data_df = data_df.rename(rename_dict, axis=1)
    data_df["Y_target"] = dataset["target"]

    train, test = randomly_split_data(data_df, test_frac=0.2)
    train.to_csv(f"{output_dir}/diabetes_train.csv", index=False)
    test.to_csv(f"{output_dir}/diabetes_test.csv", index=False)
