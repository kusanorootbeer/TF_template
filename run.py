import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noloop", action="store_true", default=False)
    args = parser.parse_args()
    lists={
        "dataset":[
            "mnist",
            ],
        "out_dir":[
            "test_out",
        ],
        "units":[
            "200,100,20"
        ],
        "epochs":[
            # 10,
            # 50,
            100,
        ],
        "batch_size":[
            128,
        ],
        "lr":[
            0.0001,
        ],
        "model":[
            # "ae",
            "vae",
            # "dae",
            # "dvae"
        ],
    }
    ops={
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
        cmd = [
            "python main.py",
        ]
        for c in lists:# うまいこと重複の起きないように作りたい
            cmd.append("--{} {}".format(c ,np.random.choice(lists[c])))
        for c in ops:
            if ops[c]:
                cmd.append(("--{}".format(c)))
        cmd=" ".join(cmd)
        os.system(cmd)
        if args.noloop:
            break

