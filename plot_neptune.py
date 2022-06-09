# coding: utf-8

# Standard imports
import pathlib

# External imports
import pandas as pd
import matplotlib.pyplot as plt


def get_scale(loss, lossname):
    scale_factor = 1
    if loss == "SSIM":
        if lossname == "L1":
            scale_factor = 200
        elif lossname == "L2":
            scale_factor = 200**2
    return scale_factor


def main():
    root_dir = pathlib.Path("./neptune_csvs")
    master_df = pd.read_csv(root_dir / "Sondra-SAR.csv")
    master_df = master_df.set_index("Id")
    master_df = master_df.drop(
        columns=["Tags", "Creation Time", "Owner", "Monitoring Time"]
    )
    master_df = master_df.rename({"config/params/loss": "loss"}, axis="columns")

    # Open all the losses
    df_measures = {}
    run_ids = []
    loss_plot = []
    for filepath in root_dir.glob("SON*.csv"):
        filename = filepath.name
        run_id = filename.split("__")[0]
        run_ids.append(run_id)
        lossname = filename.split("__")[1][:-4].split("_")[2]
        if lossname not in loss_plot:
            loss_plot.append(lossname)
        algorithm = master_df.loc[run_id]["algorithm"]
        loss = master_df.loc[run_id]["loss"]
        if run_id not in df_measures:
            df_measures[run_id] = {
                lossname: pd.read_csv(
                    filepath, header=None, names=["step", "stamp", "value"]
                ),
                "algorithm": algorithm,
                "loss": loss,
            }
        else:
            df_measures[run_id][lossname] = pd.read_csv(
                filepath, header=None, names=["step", "stamp", "value"]
            )
    # Restrict the dataframe to the identified runs
    print(run_ids)
    master_df = master_df.loc[run_ids]
    algorithms = master_df["algorithm"].unique()
    losses = master_df["loss"].unique()

    print(
        f"I selected the algorithms {algorithms}, the minimized losses {losses} and metrics to plot {loss_plot}"
    )
    # Plot all the losses per algorithm
    num_metrics = len(loss_plot)

    # When we minimize the L2
    ranges = {"L1": (0, 5), "L2": (0, 50), "SSIM": (0, 1), "psnr": (30, 40)}
    colors = {
        "srcnn2": "tab:blue",
        "srcnn": "tab:orange",
        "swintransformer": "tab:green",
        "pixelshuffle": "tab:red",
    }
    for minimized_loss in ["l1", "l2", "SSIM"]:
        fig, axes = plt.subplots(1, num_metrics, figsize=(15, 5))
        plt.suptitle(f"Minmized loss : {minimized_loss}")
        for iax, ax in enumerate(axes):
            ax.set_xlabel("Step")
            ax.set_title(loss_plot[iax])
        for key, dico in df_measures.items():
            if dico["loss"] != minimized_loss:
                continue
            # Plot all the losses of this run
            for i in range(num_metrics):
                scale = get_scale(minimized_loss, loss_plot[i])
                print(
                    f"For minimized {minimized_loss}, lossname {loss_plot[i]}, scale={scale}"
                )
                axes[i].plot(
                    scale * dico[loss_plot[i]]["value"],
                    color=colors[dico["algorithm"]],
                    label=dico["algorithm"],
                )
        for iax, ax in enumerate(axes):
            ax.set_ylim(ranges[loss_plot[iax]])
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center")
        plt.tight_layout()
        plt.savefig(f"sar_{minimized_loss}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
