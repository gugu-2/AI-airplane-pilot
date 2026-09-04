# Google Colab Training Guide — AI Airplane Pilot

> **Purpose:** This guide walks you through every step of training the AI airplane pilot model using Google Colab's free T4 GPU. Follow every step in order the first time through. After that, reconnecting sessions becomes a 2-minute routine.

---

## Table of Contents

1. [Setting Up Google Colab](#1-setting-up-google-colab)
2. [Mounting Google Drive](#2-mounting-google-drive)
3. [Uploading the Project](#3-uploading-the-project)
4. [Verifying GPU is Available](#4-verifying-gpu-is-available)
5. [Running Session 1](#5-running-session-1)
6. [What to Do If Colab Disconnects](#6-what-to-do-if-colab-disconnects)
7. [Running Sessions 2–5](#7-running-sessions-25)
8. [Downloading the Trained Model](#8-downloading-the-trained-model)
9. [Transferring the Model to the Main Project](#9-transferring-the-model-to-the-main-project)
10. [Troubleshooting](#10-troubleshooting)
11. [Pro Tips to Maximize Free T4 Time](#11-pro-tips-to-maximize-free-t4-time)

---

## 1. Setting Up Google Colab

### 1.1 — Navigate to Google Colab

Open a browser and go to:

```
https://colab.research.google.com
```

> **IMPORTANT:** You **must** be signed into a Google account. If you see a sign-in prompt, log in with your Google account before proceeding.

**What the landing page looks like:**
You will see a dialog box titled *"Open notebook"* in the center of the screen with five tabs across the top: **Examples**, **Recent**, **Google Drive**, **GitHub**, and **Upload**. There is also a blue **"New notebook"** button in the bottom-left corner of this dialog.

---

### 1.2 — Create a New Notebook

Click the **"New notebook"** button in the lower-left corner of the dialog.

**What you'll see next:**
A blank notebook opens in the editor. The top of the page shows:
- A title field currently reading **"Untitled0"** (click it to rename)
- A toolbar with **File**, **Edit**, **View**, **Insert**, **Runtime**, **Tools**, **Help** menus
- A **"+ Code"** and **"+ Text"** button below the toolbar for adding cells
- One empty code cell in the main area, ready to type in

**Rename your notebook** by clicking "Untitled0" at the top and typing:

```
AI_Airplane_Pilot_Training
```

Then press **Enter**.

---

### 1.3 — Select a T4 GPU Runtime

This is the **most important setup step**. Without GPU, training will be ~30x slower and completely unusable.

**Steps:**

1. Click **Runtime** in the top menu bar
2. Click **"Change runtime type"** from the dropdown

**What the dialog looks like:**
A modal dialog titled *"Change runtime type"* appears with two dropdowns:
- **Runtime type:** Python 3 (leave this as-is)
- **Hardware accelerator:** This defaults to "None" — **you must change this**

3. Click the **Hardware accelerator** dropdown
4. Select **T4 GPU** from the list (it appears as "T4 GPU" — not just "GPU")
5. Click the **Save** button

**What happens next:**
The dialog closes. In the top-right corner of the Colab page, you will see a green icon with **"RAM"** and **"Disk"** bars. It may briefly show "Connecting..." then transition to show resource usage. This confirms the T4 GPU runtime is now active.

---

### 1.4 — Free Tier Limitations — Read This!

> **WARNING:** Google Colab's free tier has hard limits that will affect your training workflow. Know these before you start.

| Limitation | Free Tier Detail |
|---|---|
| **GPU time per session** | ~4–5 hours maximum before forced disconnect |
| **Idle timeout** | ~30–90 minutes of no activity = session killed |
| **Daily limit** | Soft cap; Colab may throttle GPU access if overused |
| **RAM** | ~12.7 GB system RAM |
| **VRAM** | ~15 GB on T4 GPU |
| **Local disk** | ~78 GB (lost on disconnect) |
| **Google Drive** | Up to 15 GB free (persists!) |

**What "session death" means:**
When Colab disconnects you (timeout or limit hit), the Colab VM is wiped. All local files, installed packages, and in-memory state are **gone**. The good news: anything saved to Google Drive **survives**. This is why Drive mounting in Step 2 is mandatory.

**How to know you have been disconnected:**
- The green status icon in the top-right turns grey
- A yellow banner appears: *"Runtime disconnected"*
- A dialog may appear asking if you want to reconnect

---

## 2. Mounting Google Drive

This step connects your Google Drive to the Colab VM so that model checkpoints are saved to persistent storage. **Do this before running any training.**

### 2.1 — Add the Mount Cell

In your notebook, click the first empty code cell (or click **"+ Code"** to add one), then paste the following:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Press **Shift+Enter** (or click the play button on the left side of the cell) to run it.

### 2.2 — Authorize Drive Access

**What you'll see:**
A popup or inline prompt appears asking you to authenticate. The cell output will show:

```
Mounted at /content/drive
```

If a browser popup appears, click **"Allow"** to grant Colab access to your Drive.

> **NOTE:** If you have mounted Drive in a previous Colab session, it may mount silently without a popup — this is normal.

### 2.3 — Create the Checkpoint Directory on Drive

Add a new code cell and run:

```python
import os

# Create a dedicated folder on Drive for this project's checkpoints
checkpoint_dir = '/content/drive/MyDrive/AI_Airplane_Pilot/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

print(f"Checkpoint directory ready: {checkpoint_dir}")
print("Files currently in checkpoint dir:")
print(os.listdir(checkpoint_dir))
```

**Expected output (first run):**
```
Checkpoint directory ready: /content/drive/MyDrive/AI_Airplane_Pilot/checkpoints
Files currently in checkpoint dir:
[]
```

After training sessions, this directory will fill with `.pth` checkpoint files that survive disconnections.

### 2.4 — Why This Is Critical

Every time the training script saves a checkpoint, it writes to `/content/drive/MyDrive/AI_Airplane_Pilot/checkpoints/`. When Colab kills your session:

- The VM disk is wiped — LOST
- Your code in the notebook — LOST (unless you saved to Drive)
- **Your `.pth` checkpoint files on Drive — SAFE**

When you reconnect, training resumes from the last checkpoint automatically.

---

## 3. Uploading the Project

You have two options for getting the project code into your Colab VM. **Choose one.**

---

### Option A — Upload a Zip File (Recommended for First-Timers)

**Step 1:** On your local machine, zip the entire project folder.

In PowerShell:
```powershell
cd C:\Users\ceohr\Downloads
Compress-Archive -Path "AI-airplane-pilot" -DestinationPath "AI-airplane-pilot.zip"
```

**Step 2:** Upload the zip to Google Drive.
- Go to [drive.google.com](https://drive.google.com)
- Drag and drop `AI-airplane-pilot.zip` into your Drive root or into the `AI_Airplane_Pilot` folder

**Step 3:** In Colab, add a new code cell and run:

```python
import zipfile
import os

# Path to the zip on Drive
zip_path = '/content/drive/MyDrive/AI-airplane-pilot.zip'

# Extract to Colab local disk (fast NVMe — much faster than Drive for training)
extract_path = '/content/AI-airplane-pilot'

print("Extracting project zip...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/')

print(f"Extracted to: {extract_path}")
print("Contents:")
print(os.listdir(extract_path))
```

**Expected output:**
```
Extracting project zip...
Extracted to: /content/AI-airplane-pilot
Contents:
['train.py', 'models', 'environments', 'requirements.txt', 'docs', ...]
```

---

### Option B — Clone from GitHub

If your project is on GitHub, this is the cleanest approach. Add a code cell and run:

```python
import os

# Replace with your actual GitHub repository URL
GITHUB_REPO_URL = "https://github.com/YOUR_USERNAME/AI-airplane-pilot.git"

# Clone into /content/
os.chdir('/content')
!git clone {GITHUB_REPO_URL}

# Verify it worked
print("\nClone complete. Contents:")
print(os.listdir('/content/AI-airplane-pilot'))
```

**If your repo is private**, use a Personal Access Token (PAT):

```python
# Replace with your GitHub username and PAT
GITHUB_USERNAME = "YOUR_USERNAME"
GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxx"  # Generate at github.com/settings/tokens
REPO_NAME = "AI-airplane-pilot"

clone_url = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"

os.chdir('/content')
!git clone {clone_url}

print("Private repo cloned successfully.")
```

> **CAUTION:** Never hardcode your GitHub token in a notebook that you share publicly. Use Colab Secrets (Tools > Secrets) instead.

---

### 3.1 — Install Dependencies

After uploading/cloning, install the project's Python dependencies:

```python
import subprocess
import sys

print("Installing dependencies...")
result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-r',
     '/content/AI-airplane-pilot/requirements.txt'],
    capture_output=True, text=True
)
print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
if result.returncode != 0:
    print("ERRORS:")
    print(result.stderr)
else:
    print("\n✅ Dependencies installed successfully.")
```

**Expected output (last few lines):**
```
Successfully installed gymnasium-0.29.1 stable-baselines3-2.3.0 ...
✅ Dependencies installed successfully.
```

---

## 4. Verifying GPU is Available

Before starting any training, always verify PyTorch can see the T4 GPU. Add a new code cell:

```python
import torch

print("=" * 50)
print("GPU AVAILABILITY CHECK")
print("=" * 50)

cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU Name:       {device_name}")

    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM:           {vram:.1f} GB")

    print(f"CUDA Version:   {torch.version.cuda}")
    print(f"PyTorch:        {torch.__version__}")
    print("\n✅ GPU ready for training!")
else:
    print("\n❌ No GPU detected!")
    print("Go to Runtime > Change runtime type > T4 GPU and try again.")
```

### 4.1 — What the Output Should Look Like

**Correct output (T4 runtime active):**
```
==================================================
GPU AVAILABILITY CHECK
==================================================
CUDA Available: True
GPU Name:       Tesla T4
VRAM:           14.7 GB
CUDA Version:   12.2
PyTorch:        2.1.0+cu121

✅ GPU ready for training!
```

**Wrong output (CPU runtime — go fix it):**
```
==================================================
GPU AVAILABILITY CHECK
==================================================
CUDA Available: False

❌ No GPU detected!
Go to Runtime > Change runtime type > T4 GPU and try again.
```

> **IMPORTANT:** If `CUDA Available: False` — stop here. Go to **Runtime > Change runtime type**, select **T4 GPU**, click **Save**, then re-run all cells from the beginning. Do NOT attempt to train on CPU.

---

## 5. Running Session 1

### 5.1 — Set Up the Python Path

The training script needs to find the project modules. Add a cell:

```python
import sys
import os

# Add project root to Python path
project_root = '/content/AI-airplane-pilot'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.chdir(project_root)
print(f"Working directory: {os.getcwd()}")
print(f"Python path includes: {project_root}")
```

### 5.2 — Run Session 1 Training

Add a new code cell with the Session 1 training command:

```python
import subprocess
import sys

cmd = [
    sys.executable, 'train.py',
    '--session', '1',
    '--checkpoint_dir', '/content/drive/MyDrive/AI_Airplane_Pilot/checkpoints',
    '--max_hours', '4.5',
    '--device', 'cuda'
]

print("Starting Session 1 training...")
print(f"Command: {' '.join(cmd)}\n")
print("=" * 60)

# Run training (this will print output in real-time)
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

for line in process.stdout:
    print(line, end='')

process.wait()
print("=" * 60)
if process.returncode == 0:
    print("\n✅ Session 1 training complete!")
else:
    print(f"\n❌ Training exited with code {process.returncode}")
```

### 5.3 — What the Output Looks Like Line by Line

Here is what you will see as Session 1 runs:

```
Starting Session 1 training...
Command: python train.py --session 1 --checkpoint_dir /content/drive/MyDrive/AI_Airplane_Pilot/checkpoints --max_hours 4.5 --device cuda

============================================================
[INFO] AI Airplane Pilot — Training Session 1
[INFO] Loading environment: FlightSimEnv-v1
[INFO] Initializing PPO agent on device: cuda (Tesla T4)
[INFO] NUM_ENVS: 8
[INFO] Total timesteps this session: 500,000
[INFO] Checkpoint dir: /content/drive/MyDrive/AI_Airplane_Pilot/checkpoints
[INFO] Max training time: 4.5 hours
[INFO] Starting training loop...

Ep    1 | Reward: -847.3 | Ep_len: 200 | FPS: 1842 | Loss: 0.4521
Ep    2 | Reward: -791.2 | Ep_len: 200 | FPS: 1856 | Loss: 0.4103
Ep    3 | Reward: -752.8 | Ep_len: 200 | FPS: 1863 | Loss: 0.3887
...
Ep   50 | Reward: -401.5 | Ep_len: 287 | FPS: 1891 | Loss: 0.2341
[CHECKPOINT] Saved: session1_ep050.pth → /content/drive/MyDrive/AI_Airplane_Pilot/checkpoints/
...
Ep  100 | Reward: -182.3 | Ep_len: 412 | FPS: 1903 | Loss: 0.1892
[CHECKPOINT] Saved: session1_ep100.pth → /content/drive/MyDrive/AI_Airplane_Pilot/checkpoints/
```

### 5.4 — Signs That Training Is Working Correctly

| Indicator | Good Sign | Bad Sign |
|---|---|---|
| **FPS** | 1,500–2,500 (T4 speed) | Less than 100 (CPU fallback) |
| **Reward trend** | Slowly increasing (less negative) | Constant or diverging to -infinity |
| **Loss** | Gradually decreasing | Exploding to NaN |
| **Checkpoint messages** | Appear regularly | Never appear |
| **CUDA errors** | None | "CUDA out of memory" |

> **TIP:** The reward starts very negative (the plane crashes constantly). This is completely normal. By episode 50–100 you should see the reward trending upward. If reward is still at -1000 after 200 episodes, something is wrong with your environment or reward function.

---

## 6. What to Do If Colab Disconnects

### 6.1 — Don't Panic — Your Progress Is Safe

When Colab disconnects mid-training, the last checkpoint saved to Drive is intact. You lose at most one checkpoint interval of training (typically 50 episodes).

### 6.2 — Signs You Have Been Disconnected

- Output in the training cell stops suddenly with no new lines
- The status indicator (top right) shows a grey icon instead of green
- A yellow or orange banner appears: *"Runtime disconnected"* or *"Session crashed"*
- The cell shows an error like `CalledProcessError` or `BrokenPipeError`

### 6.3 — How to Reconnect

**Step 1:** Click **"Reconnect"** in the top-right corner (or in the banner that appears).

**What you see:** The status icon spins, then turns green again. Colab is allocating a fresh VM.

**Step 2:** Re-run **all setup cells from the top**, in this order:

```
Cell 1: Mount Google Drive
Cell 2: Extract zip / git clone  (VM disk was wiped — must redo this)
Cell 3: Install dependencies     (also wiped — must redo this)
Cell 4: Set sys.path
Cell 5: Verify GPU
Cell 6: Run training cell        (script auto-resumes from checkpoint)
```

> **WARNING:** Every time you reconnect, you get a brand-new VM. All local installs, extracted files, and environment variables are gone. You must re-run the setup cells every single session. Only Google Drive files survive.

### 6.4 — Resuming From a Checkpoint

The training script detects existing checkpoints automatically. When you run the training cell again with the same session number, it finds the latest checkpoint and resumes:

```python
# Same command as before — the script auto-resumes from the latest checkpoint
cmd = [
    sys.executable, 'train.py',
    '--session', '1',                  # Keep same session number until complete
    '--checkpoint_dir', '/content/drive/MyDrive/AI_Airplane_Pilot/checkpoints',
    '--max_hours', '4.5',
    '--device', 'cuda'
]
```

**Output when resuming from checkpoint:**
```
[INFO] Found existing checkpoint: session1_ep100.pth
[INFO] Resuming from episode 100 / 500,000 timesteps
[INFO] Loading model weights... done
[INFO] Resuming training from checkpoint...
Ep  101 | Reward: -178.2 | ...
```

### 6.5 — Checking What Checkpoints Exist on Drive

```python
import os
import datetime

checkpoint_dir = '/content/drive/MyDrive/AI_Airplane_Pilot/checkpoints'
files = sorted(os.listdir(checkpoint_dir))

print(f"Checkpoints saved ({len(files)} total):")
for f in files:
    path = os.path.join(checkpoint_dir, f)
    size_mb = os.path.getsize(path) / 1e6
    mtime = os.path.getmtime(path)
    t = datetime.datetime.fromtimestamp(mtime)
    print(f"  {f:<40} {size_mb:.1f} MB  saved: {t:%Y-%m-%d %H:%M}")
```

---

## 7. Running Sessions 2–5

The training is split across multiple sessions to work within Colab's time limits. Each session continues from where the last one ended. The only thing you change is the `--session` flag.

### Session 2

```python
cmd = [
    sys.executable, 'train.py',
    '--session', '2',
    '--checkpoint_dir', '/content/drive/MyDrive/AI_Airplane_Pilot/checkpoints',
    '--max_hours', '4.5',
    '--device', 'cuda'
]
```

### Session 3

```python
cmd = [
    sys.executable, 'train.py',
    '--session', '3',
    '--checkpoint_dir', '/content/drive/MyDrive/AI_Airplane_Pilot/checkpoints',
    '--max_hours', '4.5',
    '--device', 'cuda'
]
```

### Session 4

```python
cmd = [
    sys.executable, 'train.py',
    '--session', '4',
    '--checkpoint_dir', '/content/drive/MyDrive/AI_Airplane_Pilot/checkpoints',
    '--max_hours', '4.5',
    '--device', 'cuda'
]
```

### Session 5 (Final Session)

```python
cmd = [
    sys.executable, 'train.py',
    '--session', '5',
    '--checkpoint_dir', '/content/drive/MyDrive/AI_Airplane_Pilot/checkpoints',
    '--max_hours', '4.5',
    '--device', 'cuda'
]
```

---

### 7.1 — Master Training Cell (Change Only the Session Number)

Copy this single master cell into your notebook. Before each new session, update `SESSION_NUMBER` at the top:

```python
import subprocess, sys, os

# ─────────────────────────────────────────────────────────
SESSION_NUMBER = 1   # <--- CHANGE THIS (1, 2, 3, 4, or 5)
# ─────────────────────────────────────────────────────────

project_root    = '/content/AI-airplane-pilot'
checkpoint_dir  = '/content/drive/MyDrive/AI_Airplane_Pilot/checkpoints'

if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

cmd = [
    sys.executable, 'train.py',
    '--session',        str(SESSION_NUMBER),
    '--checkpoint_dir', checkpoint_dir,
    '--max_hours',      '4.5',
    '--device',         'cuda'
]

print(f"🚀 Starting Session {SESSION_NUMBER}")
print(f"Command: {' '.join(cmd)}")
print("=" * 60)

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

for line in process.stdout:
    print(line, end='')

process.wait()
print("=" * 60)
status = "✅ Complete!" if process.returncode == 0 else f"❌ Error (code {process.returncode})"
print(f"\nSession {SESSION_NUMBER}: {status}")
```

---

### 7.2 — Training Session Schedule

| Session | Timestep Range | Training Focus | Expected Reward Range |
|---|---|---|---|
| 1 | 0 – 500K | Basic flight stability | -900 → -300 |
| 2 | 500K – 1M | Altitude hold, speed control | -300 → -100 |
| 3 | 1M – 2M | Navigation, heading control | -100 → +50 |
| 4 | 2M – 3M | Landing approach | +50 → +200 |
| 5 | 3M – 4M | Full mission completion | +200 → +500 |

> **NOTE:** These reward ranges are approximate. The key signal is that rewards trend **upward** across sessions. If any session shows flat or declining rewards over many episodes, check your reward function and hyperparameters.

---

## 8. Downloading the Trained Model

After all five sessions complete, you will have a final trained model file on Google Drive.

### 8.1 — Identify the Final Model File

```python
import os

checkpoint_dir = '/content/drive/MyDrive/AI_Airplane_Pilot/checkpoints'
files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])

print("All checkpoint files:")
for i, f in enumerate(files):
    size_mb = os.path.getsize(os.path.join(checkpoint_dir, f)) / 1e6
    print(f"  [{i}] {f}  ({size_mb:.1f} MB)")

# The final model is the last file (highest episode number)
final_model = files[-1]
final_model_path = os.path.join(checkpoint_dir, final_model)
print(f"\n✅ Final model identified: {final_model}")
```

### 8.2 — Export the Final Model to a Clean Location on Drive

```python
import shutil

export_dir = '/content/drive/MyDrive/AI_Airplane_Pilot/exported_models'
os.makedirs(export_dir, exist_ok=True)

export_path = os.path.join(export_dir, 'ai_pilot_trained_final.pth')
shutil.copy2(final_model_path, export_path)

size_mb = os.path.getsize(export_path) / 1e6
print(f"✅ Model exported:")
print(f"   {export_path}  ({size_mb:.1f} MB)")
```

### 8.3 — Download the Model to Your Local Machine

**Method A — Download Directly from Colab (for files under ~100 MB):**

```python
from google.colab import files

print("Initiating download...")
files.download(export_path)
print("✅ Download started — check your browser's download bar.")
```

A browser Save dialog will appear. Save `ai_pilot_trained_final.pth` somewhere you can find it.

**Method B — Download via Google Drive (recommended for large files):**
1. Open [drive.google.com](https://drive.google.com) in a new browser tab
2. Navigate to **AI_Airplane_Pilot** → **exported_models**
3. Right-click `ai_pilot_trained_final.pth`
4. Click **Download**

> **TIP:** Method B is more reliable for files over 100 MB. The `files.download()` Colab method can time out or fail silently on large models.

---

## 9. Transferring the Model to the Main Project

### 9.1 — Where to Place the File

On your local machine, the trained `.pth` file belongs in the `models/` directory of your project:

```
C:\Users\ceohr\Downloads\AI-airplane-pilot\
└── models\
    └── ai_pilot_trained_final.pth   ← Place it here
```

Move it there using PowerShell:

```powershell
# Move from Downloads to the project models directory
Move-Item `
  "$env:USERPROFILE\Downloads\ai_pilot_trained_final.pth" `
  "C:\Users\ceohr\Downloads\AI-airplane-pilot\models\ai_pilot_trained_final.pth"
```

Verify the file is in place:

```powershell
Get-ChildItem "C:\Users\ceohr\Downloads\AI-airplane-pilot\models\"
```

**Expected output:**
```
    Directory: C:\Users\ceohr\Downloads\AI-airplane-pilot\models

Mode                 LastWriteTime         Length Name
----                 -------------         ------  ----
-a----         6/21/2026   9:00 AM      85231640  ai_pilot_trained_final.pth
```

### 9.2 — Load the Model in Your Project Code

```python
import torch
from models.pilot_network import PilotNetwork  # adjust to your actual class name

MODEL_PATH = 'models/ai_pilot_trained_final.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PilotNetwork()
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Trained model loaded from: {MODEL_PATH}")
print(f"   Trained for: {checkpoint.get('total_timesteps', 'unknown')} timesteps")
print(f"   Final reward: {checkpoint.get('best_reward', 'unknown')}")
```

---

## 10. Troubleshooting

### Error: CUDA out of memory

**Symptom:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate X MB
```

**Fix:** Reduce the number of parallel environments and/or the batch size.

```python
# Option 1: Pass num_envs flag directly
cmd = [
    sys.executable, 'train.py',
    '--session', '1',
    '--num_envs', '4',           # Default is often 8 or 16 — reduce it
    '--batch_size', '512',       # Default is often 2048 — reduce it
    '--checkpoint_dir', checkpoint_dir,
    '--max_hours', '4.5',
    '--device', 'cuda'
]
```

```python
# Option 2: Free CUDA cache before training
import torch
torch.cuda.empty_cache()
print(f"VRAM free: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
```

Try reducing `num_envs` in this order until it fits: 16 → 8 → 4 → 2.

---

### Error: ModuleNotFoundError (no module named 'environments' or 'models')

**Symptom:**
```
ModuleNotFoundError: No module named 'environments'
ModuleNotFoundError: No module named 'models'
```

**Fix:** The `sys.path` is not set. Run this cell before training:

```python
import sys, os

project_root = '/content/AI-airplane-pilot'

if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.chdir(project_root)

# Verify
print("First 5 entries on sys.path:")
for p in sys.path[:5]:
    print(f"  {p}")
print(f"Working directory: {os.getcwd()}")
```

---

### Error: Session Timeout Mid-Training

**Symptom:** Training output freezes. Status icon goes grey. No error message — it just stops printing.

**Fix:**
1. Click **Reconnect** (top-right)
2. Wait 30–60 seconds for the VM to spin up
3. Re-run all setup cells in order (Drive mount → extract project → install deps → sys.path → GPU check)
4. Re-run the training cell — it resumes automatically from the last checkpoint

You lose only the training since the last checkpoint (usually less than 50 episodes).

---

### Error: RuntimeError: CUDA error: device-side assert triggered

**Symptom:**
```
RuntimeError: CUDA error: device-side assert triggered
```

**Fix:** Set `CUDA_LAUNCH_BLOCKING=1` before re-running to get a readable error message:

```python
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Now re-run your training — the real error (usually an index out of bounds)
# will appear in the output instead of the generic CUDA assert.
```

This usually means there is a mismatch between your action space and what the model outputs. Fix that underlying index issue, then remove `CUDA_LAUNCH_BLOCKING` and re-run on GPU.

---

### Error: drive.mount() Fails or Hangs

**Symptom:** The mount cell never finishes, or shows an OAuth authentication error.

**Fix:**

```python
# Force a fresh remount
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

If that does not work: go to **Runtime > Disconnect and delete runtime**, reconnect fresh, and try mounting again.

---

### Error: FileNotFoundError for a checkpoint that should exist

**Symptom:**
```
FileNotFoundError: /content/drive/MyDrive/AI_Airplane_Pilot/checkpoints/session1_final.pth
```

**Fix:** Check what is actually on Drive:

```python
import os

checkpoint_dir = '/content/drive/MyDrive/AI_Airplane_Pilot/checkpoints'

if os.path.exists(checkpoint_dir):
    files = os.listdir(checkpoint_dir)
    print(f"Found {len(files)} files in checkpoint dir:")
    for f in sorted(files):
        print(f"  {f}")
else:
    print("Checkpoint directory does not exist yet.")
    print("This is a fresh start — no checkpoints to resume from.")
```

---

### Error: Runtime Disconnects After Only 1–2 Hours

**Possible causes:**
- Browser tab was backgrounded or minimized — Colab detected inactivity
- Too many other Colab GPU notebooks open simultaneously
- Colab is throttling your free-tier access for overuse

**Fix:**
- Keep the Colab tab **active and visible** in the foreground
- Close all other open Colab GPU notebooks
- If consistently disconnecting early, wait 2–4 hours before trying again (Colab resets usage tracking on a rolling window)

---

## 11. Pro Tips to Maximize Free T4 Time

### Tip 1 — Keep the Browser Tab Active and Visible

Colab monitors browser activity. If you minimize the window or leave the tab in the background for too long, it treats you as idle and will disconnect you.

**Best practice:** Keep the Colab tab open and visible. You can have other things open in other windows, just don't close or minimize the entire browser.

---

### Tip 2 — Use the --max_hours 4.5 Flag Every Time

Always include `--max_hours 4.5` in your training command. This causes the training script to save a final checkpoint and exit cleanly at 4.5 hours — just before Colab's forced 5-hour hard disconnect.

```python
'--max_hours', '4.5'
# Script saves checkpoint and exits at 4.5h.
# Colab's hard limit hits at ~5h.
# This gives you a clean, safe exit with your progress saved.
```

Without this flag, Colab may kill the session violently mid-write, potentially corrupting the checkpoint file.

---

### Tip 3 — Start Training Immediately After Connecting

Colab's idle timer and session clock start the moment you connect to a runtime, not when you start training. Every minute you spend in setup is billable GPU time.

**Recommended workflow:**
1. Connect to Colab
2. Immediately run all setup cells (Mount → Extract → Install → Path → GPU check)
3. Launch training within 5 minutes of connecting

Keep your setup cells small and fast — pre-install frequently needed packages to Drive if possible.

---

### Tip 4 — Run Only One GPU Notebook at a Time

Each simultaneously open GPU notebook competes for your free-tier GPU quota. Opening 3 GPU notebooks means your available compute time is fragmented.

**Keep only 1 GPU notebook open at a time.**

---

### Tip 5 — Save Your Notebook to Drive Immediately

Do not let your filled-in notebook get lost when the session resets.

```
File > Save a copy in Drive
```

This saves the entire notebook (all your filled-in cells, session history, etc.) to your Google Drive. Next time you open Colab, you can open this saved notebook and all your cells are already there — no retyping.

---

### Tip 6 — Consider Colab Pro for Longer Runs (Optional)

If you need more reliable and continuous training time, Colab Pro (~$10/month) gives you:

| Feature | Free | Colab Pro |
|---|---|---|
| Max session length | ~5 hours | ~24 hours |
| GPU access | Shared, throttled | Priority access |
| System RAM | ~12.7 GB | Up to 51 GB |
| GPU options | T4 only | T4, V100, A100 |
| Idle timeout | ~90 min | Longer grace periods |

For the full 5-session training plan in this project, the free T4 is sufficient if you follow the session structure. Colab Pro becomes worthwhile if you want to run uninterrupted overnight.

---

### Tip 7 — Use the Keep-Alive Script for Unattended Training

If you must leave your machine while a training session runs, this script prevents Colab from treating you as idle:

```python
# Run this cell AFTER starting training (in a separate cell)
# It simulates button clicks every 60 seconds to prevent idle disconnect.
#
# NOTE: Only use this for legitimate training jobs you cannot babysit.
# Do not use it to hoard idle GPU time with no active workload.

import IPython
IPython.display.display(IPython.display.Javascript("""
function keepAlive() {
    console.log("Keep-alive ping: " + new Date().toISOString());
    var connectBtn = document.querySelector("#top-toolbar > colab-connect-button");
    if (connectBtn && connectBtn.shadowRoot) {
        var btn = connectBtn.shadowRoot.querySelector("#connect");
        if (btn) btn.click();
    }
}
const aliveInterval = setInterval(keepAlive, 60000);
console.log("Keep-alive script started. Runs every 60 seconds.");
"""))
print("Keep-alive script running.")
print("Re-run this cell after each reconnect if needed.")
```

> **CAUTION:** Google's Terms of Service prohibit circumventing Colab's idle disconnect mechanism to hoard GPU resources. This script is intended only for legitimate training runs where you are actively using the GPU and cannot monitor the screen. Do not run this on an idle session with no training job active.

---

## Quick-Reference Checklist

Use this checklist at the start of every Colab session. Print it out or keep it open next to your Colab tab.

```
EVERY SESSION CHECKLIST
========================
[ ] 1. Go to: colab.research.google.com
[ ] 2. Open your saved notebook from Google Drive
[ ] 3. Runtime > Change runtime type > T4 GPU > Save
[ ] 4. Run Cell: Mount Google Drive (authorize if prompted)
[ ] 5. Run Cell: Extract zip / git clone project to /content/
[ ] 6. Run Cell: pip install -r requirements.txt
[ ] 7. Run Cell: Set sys.path and os.chdir
[ ] 8. Run Cell: Verify GPU (must show CUDA Available: True)
[ ] 9. Update SESSION_NUMBER in the master training cell
[  ] 10. Run the training cell
[ ] 11. Watch first 10 episodes of output — confirm FPS > 1000, checkpoints saving
[ ] 12. Leave browser tab active and visible
[ ] 13. Check back every hour to confirm training is still running
```

---

*Last updated: June 2026 | AI Airplane Pilot Project*
