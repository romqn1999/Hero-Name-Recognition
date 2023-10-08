# Hero Name Recognition

This project uses a combination of YOLOv5 for object detection and ResNet for classification to recognize hero names from given images.

## Setup

### Prerequisites:
- Python 3.x
- pip

### Environment Setup:
1. Clone the repository.
```bash
git clone https://github.com/romqn1999/Hero-Name-Recognition.git
cd Hero-Name-Recognition
```

2. Set up the YOLOv5 environment.
```bash
cd yolov5
pip install -r requirements.txt
cd ..
```

## Running the Script

After setting up the environment, you can run the main recognition script as follows:

```bash
python hero_name_recognizer.py --images_path /path/to/your/images/ --output_path /path/to/output/file.txt
```

Replace `/path/to/your/images/` with the path to the folder containing your test images and `/path/to/output/file.txt` with the path where you want to save the output file. If `--output_path` is not specified, the script will save the output to a default location.

## Output

The script will process each image in the specified folder and generate an output file containing the recognized hero names at the location specified by `--output_path`.
