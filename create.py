# To create the object_categories.csv

import pandas as pd
import os

# Define the categories (need to update manually)
categories = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush", "tree", "building", "road", "sky", "grass", "cloud", "window",
    "door", "fence", "street sign", "lamp", "table", "box", "bag", "shoe", "hat"
]

# Save to CSV
csv_path = "object_categories.csv"
try:
    df = pd.DataFrame(categories, columns=["category"])
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Successfully created {csv_path} with {len(categories)} categories.")
except Exception as e:
    print(f"Error creating CSV: {e}")
    exit()

# Verify the file was created
if os.path.exists(csv_path):
    print(f"Confirmed {csv_path} exists.")
    # Read and print first few rows to verify content
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        print("First 5 categories in CSV:")
        print(df.head())
    except Exception as e:
        print(f"Error reading created CSV: {e}")
else:
    print(f"Error: {csv_path} was not created.")
    exit()
