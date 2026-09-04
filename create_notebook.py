import json

notebook = {
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Aegis Autonomy - Colab Training Notebook\n",
        "1. Click the **Folder icon** on the left panel.\n",
        "2. Upload `colab_training_package.zip`.\n",
        "3. Run the cells below sequentially."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "!unzip -q -o colab_training_package.zip"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "import os\n",
        "os.environ['TORCH_CUDNN_BENCHMARK'] = 'True'\n",
        "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'\n",
        "!python src/train_rl.py"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": None,
      "metadata": {},
      "outputs": [],
      "source": [
        "from google.colab import files\n",
        "import os\n",
        "model_path = 'models/rl_waypoint_planner.pt'\n",
        "if os.path.exists(model_path):\n",
        "    files.download(model_path)\n",
        "else:\n",
        "    print('Model not found. Did the training finish?')"
      ]
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "gpuType": "T4"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}

with open("Aegis_Colab_Training.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)
print("Notebook generated successfully.")
