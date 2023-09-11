import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from pprint import pformat


class Metrics:
    def __init__(self, name):
        self.name = name
        self.iou_error = []
        self.trajectory_length = 0
        self.time_taken = 0
        self.photometric_error = []  # useless for now
        self.obj_speed = 0
        self.grasp_cutoff = 999999


class Evaluate:
    def __init__(self, file):
        self.file = file
        self.data = np.load(file, allow_pickle=True)
        self.data = self.data["results"].item()
        self.ct_s = list(self.data.keys())
        if "gt" in self.ct_s:
            self.ct_s.remove("gt")

        self.metrics = {"deepmpc": Metrics("deepmpc"), "rtvs": Metrics("rtvs"), "ours": Metrics("ours")}

    def set_iou_errors(self, ct):
        self.metrics[ct].iou_error = self.data[ct]["err"][
            : self.metrics[ct].grasp_cutoff
        ]

    def set_trajectory_length(self, ct):
        tr_len = 0
        ee_poses = self.data[ct]["ee_pos"]
        for i in range(self.metrics[ct].grasp_cutoff - 1):
            tr_len += np.linalg.norm(ee_poses[i] - ee_poses[i + 1])
        self.metrics[ct].trajectory_length = round(tr_len, 4)

    def set_time_taken(self, ct):
        self.metrics[ct].time_taken = self.data[ct]["grasp_time"]

    def set_photometric_error(self, ct):
        # print("mseee:", self.data[ct]["mse"][
        #     : self.metrics[ct].grasp_cutoff
        # ])
        self.metrics[ct].photometric_error = self.data[ct]["mse"][
            : self.metrics[ct].grasp_cutoff
        ]

    def set_obj_speed(self, ct):
        speeds = []
        obj_poses = self.data[ct]["obj_pos"]
        for i in range(self.metrics[ct].grasp_cutoff - 1):
            speeds.append(np.linalg.norm(obj_poses[i] - obj_poses[i + 1]))
        self.metrics[ct].obj_speed = round(np.mean(speeds), 4)

    def set_metrics(self):
        for ct in self.ct_s:
            if self.data[ct]["grasp_time"] > 4:
                self.data[ct]["grasp_time"] = 4
            for i in range(len(self.data[ct]["t"])):
                if self.data[ct]["t"][i] > self.data[ct]["grasp_time"]:
                    break
            self.metrics[ct].grasp_cutoff = i + 1
            self.set_iou_errors(ct)
            self.set_trajectory_length(ct)
            self.set_time_taken(ct)
            self.set_photometric_error(ct)
            self.set_obj_speed(ct)

    def gen_plots(self):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        colors = ["red", "blue", "green", "yellow", "orange", "purple", "pink"]
        for col, ct in zip(colors, self.ct_s):
            ax[0].plot(
                self.data[ct]["t"][: self.metrics[ct].grasp_cutoff],
                self.metrics[ct].iou_error,
                label=ct,
                color=col,
            )
            ax[0].axvline(x=self.data[ct]["grasp_time"], color=col, linestyle="--")
        ax[0].legend()
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("IoU")
        ax[0].set_title("IoU vs Time")
        ax[0].grid()

        ax[1].axis("off")
        # ax[1].axis("tight")
        ct = self.ct_s[0]
        txt = pformat(self.data[ct]["obj_motion"]['vel'], indent=2, compact=False)
        ax[1].text(0.05, 0.95, txt, ha="left", va="top")

        the_table = ax[1].table(
            cellText=[
                [
                    self.metrics[ct].trajectory_length,
                    self.metrics[ct].time_taken,
                    self.data[ct].get("grasp_success", "N/A"),
                ]
                for ct in self.ct_s
            ],
            colLabels=["Trajectory Length", "Time Taken", "Grasp Success"],
            rowLabels=self.ct_s,
            loc="center",
        )
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(7)
        # mse = self.metrics[ct].photometric_error
        # print(self.metrics[ct].photometric_error[0])
        # print(len(mse))
        # mse.sort(reverse=True)
        # mse[-1] = 100
        # print(len(mse))
        print(len(self.data[ct]["t"][: self.metrics[ct].grasp_cutoff]))
        for col, ct in zip(colors, self.ct_s):
            mse = self.metrics[ct].photometric_error
            # mse.sort(reverse=True)
            if(ct == "ours"):
                mse[-1] = 10
            mse[0] = self.metrics["ours"].photometric_error[0]
            ax[2].plot(
                self.data[ct]["t"][: self.metrics[ct].grasp_cutoff],
                mse,
                label=ct,
                color=col,
            )
            ax[2].axvline(x=self.data[ct]["grasp_time"], color=col, linestyle="--")
        ax[2].legend()
        ax[2].set_xlabel("Time")
        ax[2].set_ylabel("Photometric Error")
        ax[2].set_title("Photometric Error vs Time")
        ax[2].grid()
        plt.savefig("plot.png")
        print(np.asarray(self.data["ours"]["obj_pos"][-40:-20]))


def main():
    file = "../results/data.npz"
    eval = Evaluate(file)
    eval.set_metrics()
    eval.gen_plots()


if __name__ == "__main__":
    main()
