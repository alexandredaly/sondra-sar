# coding: utf-8

# Standard imports
import pathlib

# External imports
import pandas as pd


def main():
    root_dir = pathlib.Path("./neptune_csvs")
    master_df = pd.read_csv(root_dir / "Sondra-SAR.csv")
    master_df = master_df.drop(
        columns=["Tags", "Creation Time", "Owner", "Monitoring Time"]
    )

    df_measures = {}
    for filepath in root_dir.glob("SON*.csv"):
        filename = filepath.name
        run_id = filename.split("__")[0]
        lossname = filename.split("__")[1][:-4].split("_")[2]
        if run_id not in df_measures:
            df_measures[run_id] = {lossname: pd.read_csv(filepath)}


if __name__ == "__main__":
    main()
