# Sketch → Art: Real-time AI Drawing

A live AI art performance tool. Draw on your Surface Pro, and watch your sketches transform into beautiful art in real-time.

## Architecture

```
Surface Pro (browser + pen) ──WiFi──→ Laptop (this app + RTX 4070)
                                         ├── FastAPI server
                                         ├── SDXL Turbo + ControlNet
                                         └── Streams result back to browser
```

## Requirements

- **GPU:** NVIDIA RTX 3060+ with 8GB+ VRAM
- **Python:** 3.10+
- **CUDA:** 12.0+ (you have 13.0)
- **Disk:** ~10GB for model downloads (first run)

## Quick Start

```bash
chmod +x run.sh
./run.sh
```

First run will:
1. Create a Python virtual environment
2. Install all dependencies
3. Download AI models (~6GB)
4. Start the server

## Usage

1. **Start the server** on your laptop (with the GPU)
2. **Open the URL** shown in the terminal on your Surface Pro's browser
3. **Draw** with the Surface Pen — the AI transforms your sketch after a 1.5s pause
4. **Adjust the prompt** to guide what the AI produces (e.g., "a majestic horse, oil painting style")
5. **Stream with OBS** — capture the browser window showing the AI result

## Controls

| Control | Description |
|---------|-------------|
| **Prompt** | Describes what you're drawing — guides the AI |
| **AI Strength** | How much the AI deviates from your sketch (0.3 = close to sketch, 1.0 = creative) |
| **Brush size** | Pen thickness |
| **Auto/Manual** | Auto generates after you stop drawing; Manual requires clicking Generate |
| **Ctrl+Z** | Undo last stroke |
| **Enter** | Trigger generation |

## Streaming Setup

1. Start this app on your laptop
2. Open Surface Pro browser → draw
3. In OBS, add a "Window Capture" of the browser showing the AI result panel
4. Your audience sees the real-time AI transformation!

## Tips

- Keep prompts descriptive: "a majestic horse running through a field, oil painting, dramatic lighting"
- Lower AI strength (0.4-0.6) to keep more of your sketch structure
- Higher strength (0.8-1.0) for more creative AI interpretation
- The first generation is slower (model warm-up), subsequent ones are faster
