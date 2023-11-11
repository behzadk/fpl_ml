import os
from user_config import PROCESSED_DATA_DIR, MLRUNS_DIR
from fpl_ml_projects import fpl_ml

os.environ["HYDRA_FULL_ERROR"] = "1"




if __name__ == "__main__":
    # User defined directory to write our mlruns data
    user_mlruns_dir = MLRUNS_DIR

    # User defined path to our prepared data
    train_val_data_path = os.path.join(PROCESSED_DATA_DIR, "train_val_data.csv")
    test_data_path = os.path.join(PROCESSED_DATA_DIR, "test_data.csv")

    fpl_ml.train.train(user_mlruns_dir, train_val_data_path, test_data_path, )
