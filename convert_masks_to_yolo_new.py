import os
import json
from tqdm import tqdm

# Adjust these if needed
ANNOTATIONS_DIR = 'annotations'
TRAIN_IMAGES_DIR = 'dataset/images/train'
VAL_IMAGES_DIR = 'dataset/images/val'
TRAIN_LABELS_DIR = 'dataset/labels/train'
VAL_LABELS_DIR = 'dataset/labels/val'

os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)
os.makedirs(VAL_LABELS_DIR, exist_ok=True)

def normalize_points(points, img_width, img_height):
    return [f"{x/img_width:.6f} {y/img_height:.6f}" for x, y in points]

def convert_json_to_txt(json_path, target_txt_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    image_height = data['imageHeight']
    image_width = data['imageWidth']
    label_lines = []

    for shape in data['shapes']:
        if shape['shape_type'] != 'polygon':
            continue  # Skip non-polygon annotations

        points = normalize_points(shape['points'], image_width, image_height)
        line = "0 " + " ".join(points)  # class ID is 0 for "walkable_path"
        label_lines.append(line)

    # Write label file
    with open(target_txt_path, 'w') as f:
        f.write("\n".join(label_lines))

def process_all_json():
    for json_file in tqdm(os.listdir(ANNOTATIONS_DIR), desc="Converting annotations"):
        if not json_file.endswith('.json'):
            continue

        json_path = os.path.join(ANNOTATIONS_DIR, json_file)
        base_name = os.path.splitext(json_file)[0]
        image_file = base_name + '.jpg'

        if os.path.exists(os.path.join(TRAIN_IMAGES_DIR, image_file)):
            target_txt_path = os.path.join(TRAIN_LABELS_DIR, base_name + '.txt')
        elif os.path.exists(os.path.join(VAL_IMAGES_DIR, image_file)):
            target_txt_path = os.path.join(VAL_LABELS_DIR, base_name + '.txt')
        else:
            print(f" Warning: Image file for {json_file} not found in train or val.")
            continue

        convert_json_to_txt(json_path, target_txt_path)

if __name__ == '__main__':
    process_all_json()
    print("✅ Conversion complete.")
