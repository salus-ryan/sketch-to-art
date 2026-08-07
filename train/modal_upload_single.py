"""Upload a single recording file to the Modal volume. Called by the backend."""

import modal

app = modal.App("sketch-upload")
recordings_volume = modal.Volume.from_name("sketch-recordings", create_if_missing=True)


@app.function(volumes={"/recordings": recordings_volume}, timeout=120)
def upload_file(filename: str, data: bytes):
    import os
    os.makedirs("/recordings/raw", exist_ok=True)
    path = f"/recordings/raw/{filename}"
    with open(path, "wb") as f:
        f.write(data)
    recordings_volume.commit()
    print(f"Uploaded {filename} ({len(data)/1024:.1f} KB)")
    return {"filename": filename, "size_bytes": len(data)}


@app.local_entrypoint()
def main(file: str):
    from pathlib import Path
    p = Path(file)
    if not p.exists():
        print(f"File not found: {file}")
        return
    result = upload_file.remote(p.name, p.read_bytes())
    print(f"✓ {result['filename']} uploaded to Modal")
