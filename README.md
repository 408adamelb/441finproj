# CNN vs MLP for Image Classification

Comparing Convolutional Neural Networks and Multi-Layer Perceptrons on scraped image data.

## Classes
- Traffic Signs: `stop_sign`, `yield_sign`, `speed_limit_sign`
- Fruits: `apple`, `banana`, `orange`

## Results
| Model | Validation Accuracy |
|-------|---------------------|
| CNN   | 74.6%               |
| MLP   | 62.2%               |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Scrape images
```bash
export SERPAPI_KEY="your_key"
python scraping/scrape_images.py --max-images-per-class 350
```

### 2. Preprocess
```bash
python preprocessing/resize_normalize.py
python preprocessing/create_splits.py
```

### 3. Train
```bash
python models/train.py --model cnn --epochs 50
python models/train.py --model mlp --epochs 50
```

### 4. Evaluate
```bash
python evaluation/evaluate.py --checkpoint checkpoints/best_cnn.pt --model cnn
python evaluation/stats.py
```

### 5. Visualize
Open `notebooks/exploration.ipynb`

## Project Structure
```
├── models/           # MLP and CNN architectures + training
├── preprocessing/    # Image resizing, splits, augmentation
├── scraping/         # SerpAPI image scraper
├── evaluation/       # Test evaluation and statistics
├── notebooks/        # Visualization notebook
├── experiments/      # Experiment logs
├── data/             # Images (not tracked in git)
└── checkpoints/      # Model weights (not tracked in git)
```
