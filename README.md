**Project Overview**
- **Description**: This repository implements a simple Multiple Instance Learning (MIL) pipeline to detect Acute Lymphoblastic Leukemia (ALL) from bags of microscopy images. The model uses an attention-based MIL network and includes utilities for creating bags, training, testing, visualizing attention heatmaps, and a Streamlit app for quick inference.

**Quick Start**
- **Activate virtualenv (PowerShell)**:
```
& .\.venv\Scripts\Activate.ps1
```
- **Install dependencies (example)**:
```
pip install -r requirements.txt
```
- **Create bags from `data/`**:
```
python create_bags.py
```
- **Train model**:
```
python train_mil.py
```
- **Test model**:
```
python test_mil.py
```
- **Show attention heatmap (example)**:
```
python heatmap_mil.py
```
- **Run demo app (Streamlit)**:
```
streamlit run app.py
```

**Files & Purpose**
- `train_mil.py`: Dataset, AttentionMIL model, and training loop—saves model to `mil_all_classifier.pth`.
- `test_mil.py`: Loads the saved model and runs inference on test bags, printing bag-level predictions.
- `create_bags.py`: Splits images in `data/ALL` and `data/NORMAL` into fixed-size bag folders under `data_bags/`.
- `heatmap_mil.py`: Builds a patch-based bag from a high-resolution image and overlays attention heatmaps on the original image.
- `app.py`: Streamlit UI to upload images, run the model, and display predictions + attention overlays.
- `mil_all_classifier.pth`: Example/produced model weights in repo root.

**Dataset / Folder Structure**
- `data/ALL/` — PNG images labeled ALL
- `data/NORMAL/` — PNG images labeled Normal
- `data_bags/ALL/bag_1/` — example bag folder created by `create_bags.py`

**Notes on implementation**
- The dataset `MILDataset` creates random bags of fixed `bag_size` by sampling images from the `ALL` and `NORMAL` folders.
- The `AttentionMIL` model in `train_mil.py` uses a simple CNN backbone and an attention module to aggregate instance features into a bag representation.
- `train_mil.py` uses default values (10 epochs, bag_size=20 inside the dataset) and saves model weights to `mil_all_classifier.pth`.

**Tips & Troubleshooting**
- If you need a specific PyTorch build (CUDA vs CPU), install PyTorch using the instructions at https://pytorch.org/ before installing the rest of requirements.
- If `streamlit run app.py` fails, ensure `streamlit` is installed and that `mil_all_classifier.pth` exists.
- If training is slow, set `device` to CPU or CUDA depending on availability; modify `train_mil.py` to change `batch_size` or `epochs`.

**Next steps (optional)**
- Add a `requirements.txt` (provided) and pin versions for reproducibility.
- Add a `README` section describing hyperparameters and evaluation metrics.
- Provide a small sample dataset or a script to download a test dataset.

**Contact / Author**
- For questions or changes, tell me what you want next (detailed HOWTO, a `requirements.txt` with pinned versions, or Dockerfile).