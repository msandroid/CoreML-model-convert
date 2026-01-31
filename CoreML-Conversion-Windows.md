# CoreML Conversion on Windows

This document describes how to run CoreML conversion on Windows. Apple's **coremltools** supports **macOS and Linux only**; it does not run natively on Windows. Use one of the approaches below.

## Platform Support (coremltools)

| Platform | Supported |
|----------|-----------|
| macOS 10.13+ | Yes |
| Linux | Yes |
| Windows | No (use WSL, Docker, or remote) |

Reference: [Installing Core ML Tools](https://apple.github.io/coremltools/docs-guides/source/installing-coremltools.html)

---

## Option 1: WSL2 (Recommended)

Use Windows Subsystem for Linux 2 (WSL2) to run a Linux environment and execute the conversion there.

### 1.1 Install WSL2

1. Open PowerShell as Administrator and run:

```powershell
wsl --install
```

2. Restart the computer if prompted.
3. Set up Ubuntu (or another Linux distro) when WSL starts.

### 1.2 Install Dependencies in WSL

```bash
# Enter WSL
wsl

# Navigate to the project (adjust path if your repo is elsewhere)
# Windows C:\Users\YourName\Desktop\TranslateBlue maps to:
cd /mnt/c/Users/YourName/Desktop/TranslateBlue

# Install Python 3.10+ if needed
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Create and activate a virtual environment
python3 -m venv venv_coreml
source venv_coreml/bin/activate

# Install dependencies
pip install --upgrade pip
pip install qwen-tts torch coremltools huggingface_hub soundfile
```

### 1.3 Download the Model

From WSL, download the model (or copy from Windows if already downloaded):

```bash
# Option A: Download via Hugging Face CLI
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign

# Option B: If you have the model on Windows, it may already be at:
# /mnt/c/Users/YourName/Desktop/TranslateBlue/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
```

### 1.4 Run the Conversion

**Tokenizer only** (low memory, quick):

```bash
python Scripts/convert_qwen3_tts_voicedesign_to_coreml.py \
  --model-dir ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --output-dir ./models/qwen3-tts-voicedesign/coreml \
  --variants 4bit 5bit 6bit 8bit fp16 \
  --skip-speech-decoder
```

**Full conversion** (requires 32GB+ RAM, produces .mlpackage):

```bash
python Scripts/convert_qwen3_tts_voicedesign_to_coreml.py \
  --model-dir ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --output-dir ./models/qwen3-tts-voicedesign/coreml \
  --variants 4bit 5bit 6bit 8bit fp16
```

### 1.5 Path Notes for WSL

- Windows path `C:\Users\You\Desktop\TranslateBlue` = WSL path `/mnt/c/Users/You/Desktop/TranslateBlue`
- Use forward slashes and WSL paths when running scripts from WSL
- Output files are written to the same filesystem, so they are visible in Windows Explorer

---

## Option 2: Docker

Run the conversion in a Linux container.

### 2.1 Create a Dockerfile

Create `Scripts/Dockerfile.coreml-convert` (or similar):

```dockerfile
FROM python:3.11-slim

WORKDIR /workspace

RUN pip install --no-cache-dir \
    qwen-tts torch coremltools huggingface_hub soundfile

# Copy project (or mount as volume)
COPY . /workspace
```

### 2.2 Build and Run

```powershell
cd C:\Users\YourName\Desktop\TranslateBlue

docker build -f Scripts/Dockerfile.coreml-convert -t coreml-convert .
docker run -v ${PWD}/models:/workspace/models coreml-convert \
  python Scripts/convert_qwen3_tts_voicedesign_to_coreml.py \
  --model-dir ./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --output-dir ./models/qwen3-tts-voicedesign/coreml \
  --variants 4bit 5bit 6bit 8bit fp16
```

Adjust the volume mount so the `models` directory is shared between Windows and the container.

---

## Option 3: Remote Machine or CI

1. Use a Linux or macOS machine (physical, VM, or cloud) to run the conversion.
2. Copy the source model to the remote machine.
3. Run the conversion script there.
4. Copy the output (`models/qwen3-tts-voicedesign/coreml/`) back to your Windows machine.

Example for a cloud Linux instance:

```bash
# On the remote Linux machine
git clone <your-repo>
cd TranslateBlue
# Download model, install deps, run conversion
# Then scp/rsync the output back
```

---

## Option 4: Tokenizer-Only Without coremltools

The tokenizer copy step does **not** require coremltools. However, the current conversion script imports coremltools early, so it will fail on Windows. To support tokenizer-only on Windows, you would need to:

1. Refactor the script to defer the coremltools import until the decoder conversion block, or
2. Use WSL/Docker/remote for any conversion-related work.

For now, use WSL (Option 1) for tokenizer-only as well.

---

## Model-Specific Conversion Guides

| Model | Script | Doc |
|-------|--------|-----|
| Qwen3-TTS VoiceDesign | `Scripts/convert_qwen3_tts_voicedesign_to_coreml.py` | [Qwen3-TTS-VoiceDesign-CoreML-Conversion.md](Docs/Qwen3-TTS-VoiceDesign-CoreML-Conversion.md) |

The same WSL/Docker/remote workflow applies to all conversion scripts.

---

## Troubleshooting

### WSL: "command not found" for python

Use `python3` instead of `python`, or create an alias:

```bash
alias python=python3
```

### WSL: Out of memory during conversion

- Use `--skip-speech-decoder` for tokenizer-only output
- Increase WSL memory: create `C:\Users\YourName\.wslconfig`:

```ini
[wsl2]
memory=16GB
```

Then run `wsl --shutdown` and restart WSL.

### WSL: Slow file access

Projects on the Windows filesystem (`/mnt/c/...`) can be slower than on the Linux filesystem. For large model work, consider cloning the repo inside WSL (e.g. `~/TranslateBlue`) and copying only the output back to Windows.

### Docker: GPU access

CoreML conversion is CPU-based. No GPU configuration is needed for the conversion step.
