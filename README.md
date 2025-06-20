# YOLO-Based Dense Captioning in Real-Time

This project performs real-time video analysis by combining object detection using YOLOv8, image captioning using GIT, and caption enhancement through a local LLaMA 3 language model.

The result is a more descriptive and natural understanding of the scene captured from a live video stream.

---

## Features

- Real-time video capture from webcam
- Object detection using YOLOv8 (via Ultralytics)
- Caption generation using GIT (`microsoft/git-base`)
- Caption enhancement using LLaMA 3 (via Ollama)
- Visual display of bounding boxes and enhanced captions on video frames

---

## Project Structure

```
yolo-dense-captioning/
├── main.py                   # Main script: handles video capture, detection, captioning, and enhancement
├── detector.py              # YOLOv8 detector class
├── caption_model.py         # GIT-based caption generator
├── caption_enhancer.py      # LLaMA 3 caption enhancer using Ollama's local API
├── utils/
│   ├── preprocessor.py      # Image/frame preprocessing helpers
│   └── visualizer.py        # Functions to draw bounding boxes and captions on frames
├── assets/                  # Optional: sample frames or icons
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/yolo-dense-captioning.git
cd yolo-dense-captioning
```

2. Set up a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Install and run Ollama for LLaMA 3:

- Download Ollama: https://ollama.com
- Pull the model:

```bash
ollama pull llama3
```

---

## Running the Project

To start real-time captioning:

```bash
python main.py
```

The webcam will open, and you'll see live object detection with an enhanced caption displayed at the top of the screen. Press `q` to exit.

---

## Notes

- The LLaMA 3 model runs locally via Ollama. Ensure Ollama is properly installed and accessible from your system PATH.
- Caption generation is based on `microsoft/git-base` and may vary depending on image clarity.
- For faster inference, you can use a quantized model like `llama3:8b-q4_K_M`.

---

## License

This project is for educational and research purposes. For commercial use, please check the licenses of Meta’s LLaMA 3, Microsoft's GIT, and Ultralytics YOLOv8.