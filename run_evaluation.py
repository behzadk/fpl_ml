import os
import hydra

os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base="1.2", config_path="./configs")
def main(cfg):
    evaluation = cfg["evaluation"]

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.utils.call(evaluation)


if __name__ == "__main__":
    main()
