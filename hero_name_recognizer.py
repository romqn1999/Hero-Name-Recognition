import sys

# Add yolov5 directory to Python path
sys.path.append('yolov5/')

import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import cv2
import argparse
from utils.general import non_max_suppression
from models.experimental import attempt_load

# Function to crop left side of an image
def crop_left_side(image_path):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    cropped_img = img[:, :w//2, :]  # Crop the left half
    return cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

# Function to get YOLOv5 detections
def get_detections(img, model, imgsz=320):
    original_shape = img.shape[1:3]  # H, W of the original image
    img = torch.from_numpy(img).to('cpu').float() / 255.0  # Convert to torch tensor and normalize
    
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    img = torch.nn.functional.interpolate(img, size=imgsz)  # Resize
    
    # Calculate scaling factors
    scale_x = original_shape[1] / imgsz
    scale_y = original_shape[0] / imgsz

    detections = model(img)[0]  # Model forward pass
    detections = non_max_suppression(detections)[0].cpu().numpy()
    
    # Adjust bounding box coordinates using scaling factors for all detections
    adjusted_detections = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
        adjusted_detections.append([x1, y1, x2, y2, conf, cls])

    return adjusted_detections

# Function to process images, detect bounding boxes, crop, classify, and return class name
def predict_from_bboxes(image_path, yolov5_model, resnet_model):
    # Crop the left side of the image
    cropped_img_rgb = crop_left_side(image_path)
    
    # Convert to (C, H, W) format for YOLOv5 detection
    cropped_img_rgb = cropped_img_rgb.transpose(2, 0, 1)
    
    # Get detections from YOLOv5 model
    detections = get_detections(cropped_img_rgb, yolov5_model)
    
    # If no detections, return None
    if len(detections) == 0:
        return None

    # Sort detections by score and select the one with the highest score
    top_detection = sorted(detections, key=lambda x: x[4], reverse=True)[0]

    x1, y1, x2, y2 = map(int, top_detection[:4])
    cropped_region = Image.fromarray(cropped_img_rgb[:, y1:y2, x1:x2].transpose(1, 2, 0))

    debug_save_path = 'debug'
    cropped_filename = "cropped_" + os.path.basename(image_path)
    cropped_region.save(os.path.join(debug_save_path, cropped_filename))

    # Transform and classify using ResNet18
    tensor = transform(cropped_region).unsqueeze(0)
    with torch.no_grad():
        output = resnet_model(tensor)
        _, predicted_idx = output.max(1)
        prediction = idx_to_class[predicted_idx.item()]

    return prediction

def process_folder(test_images_path, yolov5_model, resnet_model, output_file="output_results.txt"):
    image_files = [f for f in os.listdir(test_images_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    with open(output_file, 'w') as out_file:
        for image_file in image_files:
            image_path = os.path.join(test_images_path, image_file)
            prediction = predict_from_bboxes(image_path, yolov5_model, resnet_model)
            if prediction:
                out_file.write(f"{image_file}\t{prediction}\n")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Process images for object detection and classification.')
    parser.add_argument('--images_path', type=str, required=True, help='Path to the test_images/ folder.')
    parser.add_argument('--output_path', type=str, default="output_results.txt", help='Path to save the output results file.')
    args = parser.parse_args()

    # Load YOLOv5 models
    yolov5_model = attempt_load('models/champions_detection/best.pt')
    yolov5_model.eval()

    # Class mapping
    with open('models/champions_classification/class_names.txt', 'r') as file:
        class_names = file.read().splitlines()
    idx_to_class = {idx: cls for idx, cls in enumerate(class_names)}

    # Instantiate the ResNet model
    resnet_model = torchvision.models.resnet18(pretrained=False, num_classes=len(class_names))
    # Load the saved weights into the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_weights = torch.load('models/champions_classification/best.pth', map_location=device)
    resnet_model.load_state_dict(model_weights)

    # Data transformation for ResNet18 model
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Process the images
    process_folder(args.images_path, yolov5_model, resnet_model, args.output_path)
    print(f"Results saved to {args.output_path}")
