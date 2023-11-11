from projects.fpl_ml import preprocessing
import hydra_zen as hz

def initialize_project_configs():
    preprocessing.initialize_preprocessing_config()
    hz.store.add_to_hydra_store(overwrite_ok=False)

