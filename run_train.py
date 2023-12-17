import os

import hydra

os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base="1.2", config_path="./configs")
def main(cfg):
    train = cfg["train"]

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.utils.call(train)


if __name__ == "__main__":
    main()
