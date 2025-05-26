import argparse
from polypus_python import running_functions as rf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--max_parallel_threads", type=int, required=True, default=0)
    args = parser.parse_args()
    rf.run_qc(shots=args.shots, max_parallel_threads=args.max_parallel_threads)