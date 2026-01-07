from mlproject.inference.runner import run_inference


def main():
    out_path = run_inference()
    print(f"Saved predictions to {out_path}")


if __name__ == "_main_":
    main()