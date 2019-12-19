import itertools
import os
import pdb
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--loop", action="store_true", default=False)
    # parser.add_argument("--debug", action="store_true", default=False)
    # args = parser.parse_args()
    lists = {
        "dataset": [
            "mnist",
        ],
        "out_dir": [
            "result",
        ],
        "units": [
            "200,50,20"
        ],
        "epochs": [
            # 10,
            # 50,
            # 100,
            500
        ],
        "batch_size": [
            1280,
        ],
        "lr": [
            0.0001,
            # 0.001,
        ],
        "model": [
            "cvae",
        ],
        "dataset_limit": [
            None,
            1000,
            100,
            20,
            10
        ]
    }
    ops = {
        # "hoge": [
        #     True,
        #     # False,
        # ],
        # "fuga": [
        #     True,
        #     False,
        # ],
    }

    keys_list = []
    for key in lists.keys():
        keys_list.append("lists['" + key + "']")
    for key in ops.keys():
        keys_list.append("ops['" + key + "']")
    combination_str = ",".join(keys_list)

    exec("combi_list=list(itertools.product({}))".format(combination_str))

    run_list = []
    for combi in combi_list:
        cmd = ["python main.py"]
        for i, key in enumerate(keys_list):
            key = key.split("'")[1]
            if key in lists.keys():
                if combi[i] == None:
                    break
                cmd.append("--{} {}".format(key, combi[i]))
            else:
                if combi[i]:
                    cmd.append("--{}".format(key))
        cmd = " ".join(cmd)
        run_list.append(cmd)

    with open("./stack.txt", mode="w") as f:
        for cmd in run_list:
            f.write(cmd+"\n")
