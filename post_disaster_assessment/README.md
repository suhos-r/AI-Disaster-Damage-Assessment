# AI-Based Post-Disaster Damage Assessment

## Description
This project provides an AI system that analyzes satellite images captured before and after natural disasters to automatically detect and classify damage to buildings and infrastructure. It uses a custom PyTorch U-Net model for image segmentation and offers an easy-to-use Streamlit web interface.

## Installation
1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate virtual environment
4. Install dependencies: `pip install -r requirements.txt`

## Usage (Streamlit App)
Run the application locally:
```bash
streamlit run post_disaster_assessment/app.py
```
This will open a web browser where you can upload pre-disaster and post-disaster satellite images to see the damage assessment results.

If you accidentally run `python post_disaster_assessment/app.py`, the script will now relaunch itself through Streamlit instead of staying in bare mode.

## Model Training
To train the model on your own dataset (format expected in dataset.py), write a training script utilizing the `train_one_epoch` and `validate` functions from `src/train.py`.

## Running Tests
Execute the unit tests:
```bash
python test_pipeline.py
```
