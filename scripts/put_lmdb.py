import os
import shutil


def put_into_working():
    paths = {
        "train": {
            "src": "/kaggle/input/datasets/royalb26/taco-dataset/taco_train.lmdb",
            "dst": "/kaggle/working/taco_train.lmdb",
        },
        "val": {
            "src": "/kaggle/input/datasets/royalb26/taco-val-dataset/taco_val.lmdb",
            "dst": "/kaggle/working/taco_val.lmdb",
        },
    }

    for split, p in paths.items():
        if not os.path.exists(p["dst"]):
            print(f"Đang copy {split} vào /kaggle/working...")
            try:
                if os.path.isdir(p["src"]):
                    shutil.copytree(p["src"], p["dst"], dirs_exist_ok=True)
                else:
                    shutil.copy(p["src"], p["dst"])
                print(f"Đã copy xong {split}!")
            except FileExistsError:
                pass

    return paths["train"]["dst"], paths["val"]["dst"]