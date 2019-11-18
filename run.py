import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    lists = {
        "dataset": [
            "mnist",
        ],
        "out_dir": [
            "test_out",
        ],
        "units": [
            "100,50,5"
        ],
        "epochs": [
            10,
            # 50,
            # 100,
        ],
        "batch_size": [
            1280,
        ],
        "lr": [
            0.0001,
        ],
        "model": [
            # "ae",
            "vae",
            # "dae",
            # "dvae"
        ],
    }
    ops = {
        # "hoge":[
        #     True,
        #     # False,
        # ],
        # "fuga":[
        #     True,
        #     # False,
        # ],
    }
    import os
    import numpy as np
    while True:
        if args.debug:
            cmd = ["python -m pdb -c continue main.py"]
        else:
            cmd = ["python main.py"]

        for c in lists:
            cmd.append("--{} {}".format(c, np.random.choice(lists[c])))
        for c in ops:
            if ops[c]:
                cmd.append(("--{}".format(c)))
        cmd = " ".join(cmd)
        os.system(cmd)
        if args.loop:
            continue
        else:
            break
