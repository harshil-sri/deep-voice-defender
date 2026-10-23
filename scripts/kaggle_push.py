"""Push the project's data and training kernel to Kaggle via the API.

Requires: pip install kaggle, and a token at ~/.kaggle/kaggle.json
(multiple accounts: keep one json per account and pass --kaggle_json).

Commands:
  python scripts/kaggle_push.py dataset   # upload ASVspoof 2019 LA as a private dataset (~7.5 GB)
  python scripts/kaggle_push.py kernel    # push kaggle/deepvoice_training.py as a kernel
                                          # (edit kernel-metadata.json first)
"""

import argparse
import json
import os
import subprocess
import sys

DATASET_SLUG = "asvspoof2019-la"
KERNEL_SLUG = "deepvoice-defender-training"


def _kaggle_env(kaggle_json):
    """Run kaggle CLI with a specific token file.

    Two quirks of the kaggle CLI (2.2.x) are handled here:
    - KAGGLE_CONFIG_DIR is ignored when an `access_token` exists in the DEFAULT
      ~/.kaggle — so HOME is also pointed at an isolated directory.
    - if the token file is not literally named kaggle.json, we copy it.
    """
    env = dict(os.environ)
    if kaggle_json:
        kaggle_json = os.path.abspath(kaggle_json)
        config_dir = os.path.dirname(kaggle_json)
        env["KAGGLE_CONFIG_DIR"] = config_dir
        if os.path.basename(kaggle_json) != "kaggle.json":
            os.makedirs(config_dir, exist_ok=True)
            tgt = os.path.join(config_dir, "kaggle.json")
            if not os.path.exists(tgt):
                import shutil

                shutil.copy(kaggle_json, tgt)
        isolated_home = os.path.join(config_dir, ".isolated_home")
        os.makedirs(isolated_home, exist_ok=True)
        env["HOME"] = isolated_home
    return env


def push_dataset(kaggle_json, version_note):
    """Create/update a private Kaggle dataset from data/raw/LA/LA."""
    meta = {
        "title": "ASVspoof 2019 LA",
        "id": "mrigankrana/asvspoof2019-la",  # user edits after first creation
        "licenses": [{"name": "other"}],
    }
    folder = "data/raw/LA/LA"
    meta_path = os.path.join(folder, "dataset-metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    cmd = ["kaggle", "datasets", "create", "-p", folder, "--dir-mode", "zip"]
    if version_note:
        cmd = ["kaggle", "datasets", "version", "-p", folder, "-m", version_note, "--dir-mode", "zip"]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=_kaggle_env(kaggle_json))


def push_kernel(kaggle_json, kernel_dir):
    cmd = ["kaggle", "kernels", "push", "-p", kernel_dir]
    subprocess.run(cmd, check=True, env=_kaggle_env(kaggle_json))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["dataset", "kernel"])
    parser.add_argument("--kaggle_json", default=None,
                        help="path to a kaggle.json token (for multiple accounts)")
    parser.add_argument("--note", default=None, help="dataset version message")
    parser.add_argument("--kernel_dir", default="kaggle")
    args = parser.parse_args()
    if args.cmd == "dataset":
        push_dataset(args.kaggle_json, args.note)
    else:
        push_kernel(args.kaggle_json, args.kernel_dir)
