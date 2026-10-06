import os
import shutil

# Định nghĩa đường dẫn nguồn và đích
def put_into_ram():
    paths = {
        "train": {
            "src": (
                "/kaggle/input/datasets/royalb26/taco-dataset/taco_train.lmdb"
            ),
            "dst": "/dev/shm/taco_train.lmdb",
        },
        "val": {
            "src": (
                "/kaggle/input/datasets/royalb26/taco-val-dataset/taco_val.lmdb"
            ),
            "dst": "/dev/shm/taco_val.lmdb",
        },
    }
    
    # Tiến hành copy vào RAM
    for split, p in paths.items():
      if not os.path.exists(p["dst"]):
        print(f"Đang copy {split} vào RAM (/dev/shm)...")
        if os.path.isdir(p["src"]):
          shutil.copytree(p["src"], p["dst"])
        else:
          shutil.copy(p["src"], p["dst"])
        print(f"Đã copy xong {split}!")
    
    # Đường dẫn dùng cho Dataset / DataLoader:
    lmdb_train_path = paths["train"]["dst"]
    lmdb_val_path = paths["val"]["dst"]
    return lmdb_train_path, lmdb_val_path