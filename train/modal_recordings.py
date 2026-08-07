"""
Modal Recordings Pipeline — upload, manage, and process sketch recordings.

- Uploads local .webm recordings to a Modal Volume
- Extracts frames for training (sketch frames + camera overlay)
- Provides a LoRA fine-tuning stub for sketch-to-art style adaptation

Usage:
    # Upload all local recordings to Modal
    modal run train/modal_recordings.py

    # Upload and then fine-tune
    modal run train/modal_recordings.py --train

    # List recordings on Modal
    modal run train/modal_recordings.py --list-only
"""

import modal

app = modal.App("sketch-recordings")

# Persistent volume for recordings + extracted frames
recordings_volume = modal.Volume.from_name("sketch-recordings", create_if_missing=True)

# Training image with video processing + torch
train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "Pillow==10.4.0",
        "peft==0.11.1",
        "accelerate==0.31.0",
        "diffusers==0.29.2",
        "transformers==4.42.3",
    )
)


@app.function(
    volumes={"/recordings": recordings_volume},
    timeout=300,
)
def upload_recordings(file_data: list[tuple[str, bytes]]):
    """Upload recording files to the Modal volume."""
    import os
    os.makedirs("/recordings/raw", exist_ok=True)
    uploaded = []
    for filename, data in file_data:
        path = f"/recordings/raw/{filename}"
        with open(path, "wb") as f:
            f.write(data)
        uploaded.append({"filename": filename, "size_bytes": len(data), "path": path})
        print(f"  Uploaded: {filename} ({len(data) / 1024 / 1024:.1f} MB)")
    recordings_volume.commit()
    return uploaded


@app.function(
    image=train_image,
    volumes={"/recordings": recordings_volume},
    timeout=600,
)
def extract_frames(fps: int = 2):
    """Extract frames from all .webm recordings for training."""
    import os
    import subprocess

    raw_dir = "/recordings/raw"
    frames_dir = "/recordings/frames"
    os.makedirs(frames_dir, exist_ok=True)

    if not os.path.exists(raw_dir):
        print("No recordings found.")
        return {"total_frames": 0}

    videos = [f for f in os.listdir(raw_dir) if f.endswith(".webm")]
    print(f"Processing {len(videos)} recordings...")

    total_frames = 0
    for video in sorted(videos):
        video_path = os.path.join(raw_dir, video)
        name = video.replace(".webm", "")
        out_dir = os.path.join(frames_dir, name)
        os.makedirs(out_dir, exist_ok=True)

        # Extract frames at given FPS
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps={fps}",
            "-q:v", "2",
            os.path.join(out_dir, "frame_%04d.jpg"),
            "-y", "-loglevel", "warning"
        ]
        subprocess.run(cmd, check=True)
        n = len([f for f in os.listdir(out_dir) if f.endswith(".jpg")])
        total_frames += n
        print(f"  {video}: {n} frames extracted")

    recordings_volume.commit()
    return {"total_frames": total_frames, "videos": len(videos)}


@app.function(
    image=train_image,
    gpu="A100",
    volumes={"/recordings": recordings_volume},
    timeout=3600,
)
def finetune_lora(
    base_model: str = "stabilityai/stable-diffusion-2-1",
    epochs: int = 5,
    lr: float = 1e-4,
    rank: int = 8,
    batch_size: int = 4,
    resolution: int = 512,
):
    """
    LoRA fine-tune a Stable Diffusion model on extracted sketch frames.

    This adapts the model to understand sketch→art style by training on
    pairs of (sketch frame from recording, description of the drawing).
    """
    import os
    import io
    import torch
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms

    frames_dir = "/recordings/frames"
    if not os.path.exists(frames_dir):
        print("No frames found. Run extract_frames first.")
        return None

    # Collect all frames
    all_frames = []
    for session in sorted(os.listdir(frames_dir)):
        session_dir = os.path.join(frames_dir, session)
        if os.path.isdir(session_dir):
            for f in sorted(os.listdir(session_dir)):
                if f.endswith(".jpg"):
                    all_frames.append(os.path.join(session_dir, f))

    print(f"Training on {len(all_frames)} frames from {len(os.listdir(frames_dir))} sessions")

    if len(all_frames) < 5:
        print("Not enough frames for fine-tuning. Record more sketches!")
        return None

    # --- LoRA fine-tuning with PEFT ---
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel
    from peft import LoraConfig, get_peft_model

    print(f"\nLoading base model: {base_model}")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model, torch_dtype=torch.float16
    )
    unet = pipe.unet.to("cuda", dtype=torch.float32)

    # Configure LoRA
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.05,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Simple sketch dataset
    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    class SketchDataset(Dataset):
        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            return transform(img)

    dataset = SketchDataset(all_frames)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Training loop
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)
    noise_scheduler = pipe.scheduler

    print(f"\nTraining LoRA adapter:")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, LR: {lr}, Rank: {rank}")
    print(f"  Frames: {len(dataset)}, Batches/epoch: {len(loader)}\n")

    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(loader):
            batch = batch.to("cuda")

            # Add noise
            noise = torch.randn_like(batch)
            timesteps = torch.randint(0, 1000, (batch.shape[0],), device="cuda").long()
            noisy = noise_scheduler.add_noise(batch, noise, timesteps)

            # Predict noise
            noise_pred = unet(noisy, timesteps).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (i + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(loader)}, "
                      f"Loss: {total_loss/(i+1):.4f}")

        print(f"Epoch {epoch+1}/{epochs} — Avg loss: {total_loss/len(loader):.4f}")

    # Save LoRA weights
    out_path = "/recordings/lora_weights"
    os.makedirs(out_path, exist_ok=True)
    unet.save_pretrained(out_path)
    recordings_volume.commit()
    print(f"\nLoRA weights saved to Modal volume at: {out_path}")

    # Also return as bytes for local download
    import tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(out_path, arcname="lora_weights")
    return buf.getvalue()


@app.local_entrypoint()
def main(
    train: bool = False,
    list_only: bool = False,
    extract_fps: int = 2,
):
    from pathlib import Path
    import json

    recordings_dir = Path(__file__).parent.parent / "recordings"
    logs_dir = Path(__file__).parent.parent / "logs"

    if list_only:
        # Just show what's local
        if recordings_dir.exists():
            files = sorted(recordings_dir.glob("*.webm"))
            print(f"\nLocal recordings ({len(files)}):")
            for f in files:
                print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")
        else:
            print("No local recordings found.")
        return

    # --- Upload all local recordings ---
    if not recordings_dir.exists() or not list(recordings_dir.glob("*.webm")):
        print("No recordings to upload. Draw something first!")
        return

    files = sorted(recordings_dir.glob("*.webm"))
    print(f"\nUploading {len(files)} recordings to Modal...")

    file_data = [(f.name, f.read_bytes()) for f in files]
    results = upload_recordings.remote(file_data)
    total_mb = sum(r["size_bytes"] for r in results) / (1024 * 1024)
    print(f"\n✓ Uploaded {len(results)} files ({total_mb:.1f} MB total)")

    # Also upload the audit log if it exists
    audit_log = logs_dir / "audit.jsonl"
    if audit_log.exists():
        log_data = [(audit_log.name, audit_log.read_bytes())]
        upload_recordings.remote(log_data)
        print("✓ Uploaded audit log")

    # --- Extract frames ---
    print(f"\nExtracting frames at {extract_fps} FPS...")
    frame_result = extract_frames.remote(fps=extract_fps)
    print(f"✓ Extracted {frame_result['total_frames']} frames from {frame_result['videos']} videos")

    if not train:
        print("\nDone! Run with --train to start LoRA fine-tuning.")
        return

    # --- Fine-tune ---
    print("\nStarting LoRA fine-tuning on A100...")
    lora_bytes = finetune_lora.remote()

    if lora_bytes:
        out_dir = Path("models/lora")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "sketch_lora.tar.gz"
        out_path.write_bytes(lora_bytes)
        print(f"\n✓ LoRA weights saved to: {out_path}")
        print("  Extract with: tar xzf models/lora/sketch_lora.tar.gz -C models/")
    else:
        print("\nTraining skipped (not enough data)")
