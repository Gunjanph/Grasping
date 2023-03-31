import argparse
import os
import shutil

import numpy as np
from multiprocessing import Process, Queue


def task(mixed_arg):
    from sim import simulate
    args, controller, queue = mixed_arg
    init_cfg = args.init_cfg
    sim_data = simulate(
            init_cfg, False, controller, flowdepth=args.flowdepth
        )
    if args.video:
        os.system("./mkvideo.sh")
        os.system("mv video.mp4 ../results/video_{}.mp4".format(controller))
        os.system(
            "mv video_flow.mp4 ../results/video_flow_{}.mp4".format(controller)
        )
    queue.put(sim_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--controllers", nargs="+", type=str, default=["rtvs", "ours"]
    )
    parser.add_argument("-v", "--video", action="store_true")
    parser.add_argument("-r", "--random", action="store_true")
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("--flowdepth", action="store_true")
    args = parser.parse_args()

    init_cfg = [[0.45, -0.05, 0.851], [-0.01, 0.03, 0]]
    if args.seed is not None:
        np.random.seed(args.seed)

    if args.random:
        from utils.sim_utils import get_random_config
        init_cfg = get_random_config()

    shutil.rmtree("../results", ignore_errors=True)
    os.makedirs("../results", exist_ok=True)

    data = {
        "init_cfg": init_cfg,
        "flowdepth": args.flowdepth,
        "results": {},
    }
    args.data = data
    args.init_cfg = init_cfg
    for controller in args.controllers:
        q = Queue()
        p = Process(target=task, args=((args, controller, q),))
        p.start()
        data["results"][controller] = q.get()
        p.join()

    np.savez_compressed("../results/data.npz", **data)


if __name__ == "__main__":
    main()
