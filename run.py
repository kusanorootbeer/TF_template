import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    cmd = [
        "python3 main.py",
    ]
    for c in lists:
        cmd.append("--{} {}".format(c ,lists[c].pop()))
    for c in ops:
        if ops[c]:
            cmd.append(("--{}".format(c)))
    cmd=" ".join(cmd)
    os.system(cmd)

