import os
import shutil
from pathlib import Path

from mlproject.train.trainer import run_training


def main():

    input_dir = Path("/opt/ml/processing/input")
    output_dir = Path("/opt/ml/processing/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_train_features = Path("src/mlproject/data/train_features.csv")
    repo_train_features.parent.mkdir(parents=True, exist_ok=True)

    src_features = input_dir / "train_features.csv"
    if not src_features.exists():
        raise FileNotFoundError(f"Expected {src_features} but it does not exist.")

    shutil.copyfile(src_features, repo_train_features)


    paths = run_training()

    for k, v in paths.items():
        p = Path(v)
        if p.exists():
            dest = output_dir / p.name
            shutil.copyfile(p, dest)

  
    manifest = output_dir / "artifacts_manifest.txt"
    with manifest.open("w") as f:
        for k, v in paths.items():
            f.write(f"{k}: {v}\n")

    print("Saved training artifacts to:", str(output_dir))


if __name__ == "__main__":
    main()
