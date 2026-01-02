from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import time

class CaptionModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained("microsoft/git-base")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/git-base").to(self.device)
        self.model.eval()

        print("[INFO] Warming up GIT-Base caption model...")
        _ = self.caption(Image.new("RGB", (224, 224)))
        print("[INFO] Warm-up complete ")

    def caption(self, image_pil):
        start = time.time()
        image_pil = image_pil.resize((224, 224))

        inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=50)

        caption = self.processor.batch_decode(output, skip_special_tokens=True)[0].strip()
        end = time.time()

        print(f"true caption time {end - start}")
        print(caption)
        return caption
