# Image Classification AI/ML Project

Welcome to your Image Classification project! This repository is set up with a clean, beginner-friendly structure using **PyTorch**. 

## Folder Structure

```text
Image_classification/
│
├── data/                   # Put your raw image data here
│   ├── train/              # Training images
│   │   ├── class_a/        # E.g., folder full of "cat" images
│   │   └── class_b/        # E.g., folder full of "dog" images
│   └── val/                # Validation images (Optional)
│       ├── class_a/
│       └── class_b/
│
├── models/                 # Trained model weights (.pth files) will be saved here automatically
│
├── src/                    # Source code directory
│   ├── dataset.py          # Code for reading and transforming images
│   ├── model.py            # Neural Network Architecture (SimpleCNN)
│   └── train.py            # The main training script
│
├── requirements.txt        # Python dependencies
└── README.md               # You are here
```

## How to use this project

### 1. Install Dependencies
Make sure you have Python installed. Open your terminal in this directory and run:
```bash
pip install -r requirements.txt
```

### 2. Prepare your Data
You need to organize your images into folders based on their categories. 
1. Create a `data` folder in the root directory.
2. Inside `data`, create a `train` folder.
3. Inside `train`, create one folder for **each category** you want to classify, and place the corresponding images inside. 

**Example:** If you are classifying Cats vs Dogs:
* `data/train/cats/` (put all cat images here)
* `data/train/dogs/` (put all dog images here)

### 3. Train the Model
Once your images are in place, you can train your Artificial Intelligence by running:
```bash
python src/train.py
```

The script will automatically detect the classes based on your folder names, train a Convolutional Neural Network (CNN) for 10 epochs, and save the learned weights into a `models/` folder.

### Labeling concept

This project uses folder names as image labels. Each folder inside `data/train/` becomes one class label:

```text
data/train/cats/  -> Label 0: cats
data/train/dogs/  -> Label 1: dogs
```

During prediction, the Flask + React UI shows the uploaded file name, predicted label id, predicted label name, confidence percentage, and probability for every label.

### Object detection with boxes

The Flask + React UI also includes object detection using YOLO. This is different from classification:

* Classification predicts one image-level label, such as `cats` or `dogs`.
* Object detection finds objects inside the image and draws bounding boxes around them.

Run the Flask app and click **Detect Objects** after uploading an image:

```bash
python flask_app.py --host 0.0.0.0 --port 8000
```

The first detection run downloads the pretrained YOLO model file `yolov8n.pt`. This pretrained model can detect common COCO objects such as cars, buses, trucks, people, and traffic lights. For custom objects such as specific road signs, train a custom YOLO detection dataset with bounding box labels.

## Run on Lightning AI Studio

No special deployment file is required for training this project on Lightning AI Studio. Upload or clone this project into a Studio, then run:

```bash
pip install -r requirements.txt
python src/train.py
```

Make sure your dataset is available inside:

```text
data/train/<class_name>/
data/val/<class_name>/
```

The class folder names in `data/val` should match the names in `data/train`. For example, if training uses `cats` and `dogs`, validation should also use `cats` and `dogs`.

If you push this project to GitHub, note that `data/` is ignored by `.gitignore`, so upload the dataset separately in Lightning AI Studio or remove `data/` from `.gitignore` only if you really want the images committed.

### See prediction output

After training finishes, run the Streamlit app:

```bash
streamlit run app.py
```

Upload an image in the app to see the predicted class, confidence score, and class probability chart.

### Run Flask + React UI

After training finishes, you can also run the Flask app with a React upload UI:

```bash
python flask_app.py --host 0.0.0.0 --port 8000
```

Open the URL shown by Lightning AI. Upload a cat or dog image and the app will show the predicted class, confidence percentage, and probability for each class.
You can also click **Detect Objects** to draw bounding boxes on the uploaded image.

The Flask API endpoint is:

```text
POST /predict
```

Send the image file using the form field name `image`.
