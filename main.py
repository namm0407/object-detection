import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import pandas as pd  # For reading CSV

# Load the pre-trained Grounding DINO model and processor
model_id = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

# Load an image
try:
    image = Image.open("animals.png").convert("RGB")
except FileNotFoundError:
    print("Error: image not found. Please provide a valid image file.")
    exit()

# Load object categories from a CSV file
try:
    categories_df = pd.read_csv("object_categories.csv")
    categories = categories_df["category"].tolist()
    text_prompt = " . ".join(categories)
except FileNotFoundError:
    print("Error: 'object_categories.csv' not found. Using fallback categories.")
    categories = ["person", "dog", "cat", "car", "truck", "bicycle"]
    text_prompt = " . ".join(categories)
except KeyError:
    print("Error: CSV file must contain a 'category' column.")
    exit()

# Limit the number of categories
max_categories = 100
if len(categories) > max_categories:
    print(f"Warning: Truncating to {max_categories} categories to avoid token limit.")
    categories = categories[:max_categories]
    text_prompt = " . ".join(categories)

print(f"Loaded {len(categories)} categories for detection.")

# Preprocess the inputs
inputs = processor(images=image, text=text_prompt, return_tensors="pt")

# Run inference
with torch.no_grad():
    outputs = model(**inputs)

# Process the outputs
results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.4,
    target_sizes=[image.size[::-1]]  # (height, width)
)[0]

# Visualize the results
print("Detected objects:")
for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
    print(f"- {label}: Confidence = {score:.2f}, Box = {box}")

image_np = np.array(image)
image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
image_height, image_width = image_cv.shape[:2]

for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
    if score > 0.4:
        box = [int(b) for b in box]
        x1, y1, x2, y2 = box

        # Calculate box dimensions
        box_width = x2 - x1
        box_height = y2 - y1

        # Define text and font properties
        text = f"{label} {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        min_font_scale = 0.3  # Minimum font size for readability
        max_font_scale = 1.0  # Maximum font size
        font_thickness = 1    # Initial thickness, adjusted later

        # Find appropriate font scale to fit text within half the box width
        target_width = box_width / 2  # Text width should not exceed half the box width
        font_scale = max_font_scale
        while font_scale >= min_font_scale:
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            if text_size[0] <= target_width:
                break
            font_scale -= 0.05  # Decrease font scale incrementally
        font_scale = max(font_scale, min_font_scale)  # Ensure minimum font scale
        font_thickness = max(1, int(font_scale * 2))  # Adjust thickness

        # Recalculate text size with final font scale
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

        # Draw bounding box
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Place text inside the box, near the top
        text_x = x1 + 5
        text_y = y1 + text_size[1] + 5  # 5 pixels from top of box

        # Ensure text stays within image bounds
        text_y = max(text_y, text_size[1] + 5)
        text_y = min(text_y, image_height - 5)

        # Add a semi-transparent black rectangle as text background
        bg_x1 = text_x - 2
        bg_y1 = text_y - text_size[1] - 2
        bg_x2 = text_x + text_size[0] + 2
        bg_y2 = text_y + 2
        overlay = image_cv.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, image_cv, 1 - alpha, 0, image_cv)

        # Draw text
        cv2.putText(
            image_cv,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 255, 0),
            font_thickness,
        )

# Display the image
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
