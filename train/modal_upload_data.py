"""
Upload processed QuickDraw data to Modal volume.

Usage:
    modal run train/modal_upload_data.py --data-dir data/quickdraw
"""

import argparse
import os
from pathlib import Path

import modal

app = modal.App("upload-quickdraw")
data_volume = modal.Volume.from_name("braillenet-data", create_if_missing=True)


@app.function(volumes={"/data": data_volume})
def upload_batch(files: list[tuple[str, bytes]]):
    """Upload a batch of files to the volume."""
    for remote_path, content in files:
        full_path = Path("/data") / remote_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(content)
    return len(files)


@app.local_entrypoint()
def main(data_dir: str = "data/quickdraw"):
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: {data_path} does not exist. Run quickdraw_data.py first.")
        return

    # Collect all files
    all_files = []
    for root, dirs, files in os.walk(data_path):
        for f in files:
            local = Path(root) / f
            remote = str(local.relative_to(data_path))
            all_files.append((remote, local.read_bytes()))

    print(f"Uploading {len(all_files)} files from {data_path}")

    # Upload in batches of 100
    batch_size = 100
    total_uploaded = 0
    batches = [all_files[i:i+batch_size] for i in range(0, len(all_files), batch_size)]

    for i, batch in enumerate(batches):
        count = upload_batch.remote(batch)
        total_uploaded += count
        if (i + 1) % 10 == 0:
            print(f"  {total_uploaded}/{len(all_files)} uploaded...")

    data_volume.commit()
    print(f"\nDone! Uploaded {total_uploaded} files to Modal volume 'braillenet-data'")
