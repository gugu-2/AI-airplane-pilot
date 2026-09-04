import json, os

cells = []

# ── Markdown: Title ─────────────────────────────────────────────────────────
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Aegis OS - Session 5 Training\n",
        "## Vehicle Profiles: Drone vs Fixed-Wing Airplane\n",
        "\n",
        "**Run each block ONE AT A TIME, top to bottom.**\n",
        "\n",
        "### Before you start:\n",
        "1. **Runtime > Change Runtime Type > T4 GPU** (must be GPU, not CPU)\n",
        "2. Upload `session5_colab_package.zip` using the file browser (folder icon on the left sidebar)\n",
        "3. Upload `latest.pth` to Google Drive at: `My Drive/aegis_checkpoints/latest/latest.pth`\n"
    ]
})

# ── Block 1: Unzip ───────────────────────────────────────────────────────────
cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Block 1 - Unzip Your Package\n"]})
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [
        "import zipfile, os\n",
        "\n",
        "with zipfile.ZipFile('session5_colab_package.zip', 'r') as zf:\n",
        "    zf.extractall('.')\n",
        "\n",
        "for f in ['src/train_rl_session5.py', 'src/cognitive/rl_models.py']:\n",
        "    status = 'OK' if os.path.exists(f) else 'MISSING'\n",
        "    print(status + ': ' + f)\n",
        "\n",
        "print('Block 1 complete!')\n"
    ]
})

# ── Block 2: Mount Drive ─────────────────────────────────────────────────────
cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Block 2 - Mount Google Drive\n"]})
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n",
        "print('Google Drive mounted!')\n"
    ]
})

# ── Block 3: Verify Checkpoint ───────────────────────────────────────────────
cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Block 3 - Verify Session 4 Brain\n"]})
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [
        "import torch, time, os\n",
        "\n",
        "path = '/content/drive/MyDrive/aegis_checkpoints/latest/latest.pth'\n",
        "\n",
        "if not os.path.exists(path):\n",
        "    print('NOT FOUND: ' + path)\n",
        "    print('Please upload latest.pth to Google Drive at:')\n",
        "    print('  My Drive / aegis_checkpoints / latest / latest.pth')\n",
        "else:\n",
        "    c = torch.load(path, map_location='cpu', weights_only=False)\n",
        "    print('Checkpoint found!')\n",
        "    print('  Session:     ' + str(c.get('session')))\n",
        "    print('  Update:      ' + str(c.get('update')) + ' / 500')\n",
        "    print('  Total Steps: ' + format(c.get('total_steps', 0), ','))\n",
        "    print('  Best Reward: ' + str(round(c.get('best_reward', 0), 4)))\n",
        "    dim = c['actor']['physics_net.0.weight'].shape[1]\n",
        "    print('  Physics Dim: ' + str(dim) + '  (9 = Session 4 brain, ready for surgery!)')\n",
        "    saved = time.strftime('%Y-%m-%d %H:%M', time.localtime(c.get('timestamp', 0)))\n",
        "    print('  Saved:       ' + saved)\n",
        "    print()\n",
        "    print('Block 3 complete. Brain verified and ready!')\n"
    ]
})

# ── Block 4: Check GPU ───────────────────────────────────────────────────────
cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Block 4 - Check GPU\n"]})
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [
        "import torch\n",
        "\n",
        "if torch.cuda.is_available():\n",
        "    name = torch.cuda.get_device_name(0)\n",
        "    vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)\n",
        "    print('GPU Active: ' + name)\n",
        "    print('VRAM: ' + str(vram) + ' GB')\n",
        "    print('You are good to start training!')\n",
        "else:\n",
        "    print('NO GPU FOUND!')\n",
        "    print('Go to: Runtime > Change Runtime Type > T4 GPU, then reconnect.')\n"
    ]
})

# ── Block 5: Train ───────────────────────────────────────────────────────────
cells.append({"cell_type": "markdown", "metadata": {}, "source": [
    "## Block 5 - START SESSION 2 TRAINING\n",
    "\n",
    "This runs for up to **4.5 hours** and auto-saves every ~12 minutes to your Drive.\n",
    "\n",
    "You will see this line confirming the brain upgrade worked:\n",
    "```\n",
    "[Surgery] Grafted physics_net.0.weight: expanded (64, 9) -> (64, 10)\n",
    "```\n",
    "Then training updates will print every 5 updates like:\n",
    "```\n",
    "  Update |       Steps | Avg Reward | Goals/Env\n",
    "     500 | 524,800,000 |      -0.10 |      0.00\n",
    "```\n"
]})
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [
        "!python src/train_rl_session5.py \\\n",
        "    --session 2 \\\n",
        "    --checkpoint_dir /content/drive/MyDrive/aegis_checkpoints/ \\\n",
        "    --max_hours 4.5\n"
    ]
})

# ── Block 6: Verify Results ──────────────────────────────────────────────────
cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## Block 6 - Save and Check Session 5 Final Brain\n"]})
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [
        "import torch, time, os, shutil\n",
        "\n",
        "latest_path = '/content/drive/MyDrive/aegis_checkpoints/latest/latest.pth'\n",
        "final_path = '/content/drive/MyDrive/aegis_checkpoints/session5_final.pth'\n",
        "\n",
        "# Copy the latest brain to a new distinct file for Session 5\n",
        "if os.path.exists(latest_path):\n",
        "    shutil.copy(latest_path, final_path)\n",
        "    print(f'✅ Successfully saved Session 5 brain as: {final_path}\\n')\n",
        "else:\n",
        "    print('❌ ERROR: Could not find latest.pth to copy!')\n",
        "\n",
        "c = torch.load(final_path, map_location='cpu', weights_only=False)\n",
        "\n",
        "print('=== SESSION 2 FINAL RESULTS ===')\n",
        "print('Session:     ' + str(c.get('session')))\n",
        "print('Update:      ' + str(c.get('update')) + ' / 3000')\n",
        "print('Total Steps: ' + format(c.get('total_steps', 0), ','))\n",
        "print('Best Reward: ' + str(round(c.get('best_reward', 0), 4)))\n",
        "dim = c['actor']['physics_net.0.weight'].shape[1]\n",
        "print('Physics Dim: ' + str(dim) + '  (10 = Session 5 brain!)')\n",
        "saved = time.strftime('%Y-%m-%d %H:%M', time.localtime(c.get('timestamp', 0)))\n",
        "print('Saved:       ' + saved)\n",
        "print()\n",
        "print('NEXT STEP: Download session5_final.pth from Google Drive')\n",
        "print('           and save it to your computer! (Do NOT overwrite Session 4)')\n"
    ]
})

# ── Write notebook ───────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU"
    },
    "cells": cells
}

out = 'c:/Users/ceohr/Downloads/AI-airplane-pilot/Session5_Training.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

size = round(os.path.getsize(out) / 1024, 1)
print(f'Notebook created: {out}  ({size} KB)')
