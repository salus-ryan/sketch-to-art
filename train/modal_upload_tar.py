"""
Upload QuickDraw consolidated data to Modal volume.

Usage:
    modal run train/modal_upload_tar.py
"""

import modal

app = modal.App("upload-quickdraw")
data_volume = modal.Volume.from_name("braillenet-data", create_if_missing=True)


@app.function(
    volumes={"/data": data_volume},
    timeout=3600,
)
def upload_file(file_bytes: bytes, remote_name: str):
    """Write a single file to the volume."""
    from pathlib import Path

    target = Path("/data") / remote_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(file_bytes)
    size_mb = len(file_bytes) / 1024 / 1024
    print(f"Written: {target} ({size_mb:.1f} MB)")

    data_volume.commit()
    return {"path": str(target), "size_mb": round(size_mb, 1)}


@app.local_entrypoint()
def main():
    from pathlib import Path

    # Upload consolidated .npz (single file, ~110MB, loads in seconds)
    npz_path = Path("data/quickdraw_consolidated.npz")
    if not npz_path.exists():
        print("No consolidated data found. Run:")
        print("  python train/quickdraw_data.py --categories 50 --samples-per-cat 10000 --output data/quickdraw")
        print("  Then run the consolidation script (see modal_pretrain_braillenet.py)")
        return

    size_mb = npz_path.stat().st_size / 1024 / 1024
    print(f"Uploading {npz_path} ({size_mb:.1f} MB) to Modal volume...")
    result = upload_file.remote(npz_path.read_bytes(), "quickdraw_consolidated.npz")
    print(f"Done! {result}")
