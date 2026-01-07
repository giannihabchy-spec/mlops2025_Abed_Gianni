from mlproject.train.trainer import run_training


def main():
    paths = run_training()
    print("Training done. Artifacts saved:")
    for k, v in paths.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
