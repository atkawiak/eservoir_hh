import pandas as pd
import os

results_dir = "/home/atkaw/Dokumenty/hailitacja/eservoir_hh/project_C_poisson/src/results_test_journal"
files = [f for f in os.listdir(results_dir) if f.endswith('.parquet')]
if not files:
    print("No files found")
else:
    latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(results_dir, x)))
    df = pd.read_parquet(os.path.join(results_dir, latest_file))
    print(f"Summary for {latest_file}:")
    print(df.to_string())
