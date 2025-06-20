from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import time

class CaptionModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        self.model.eval()

        print("[INFO] Warming up BLIP-1 caption model...")
        _ = self.caption(Image.new("RGB", (224, 224)))
        print("[INFO] Warm-up complete ✅")

    def caption(self, image_pil):

        start = time.time()
        inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=30)
            
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        end = time.time()

        print(f"true caption time {end - start}")
        print(caption)
        return caption
