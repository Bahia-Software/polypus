import argparse
from polypus_python import running_functions as rf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument("--shots", type=int, required=True)
    args = parser.parse_args()
    rf.run_qc_in_qpu(qpu_id=args.id, shots=args.shots)