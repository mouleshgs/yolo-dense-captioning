# caption_enhancer.py

import subprocess
import json

import shutil
print("Ollama path:", shutil.which("ollama"))


class CaptionEnhancer:
    def __init__(self, model_name="llama3"):
        self.model = model_name

    def enhance(self, base_caption, detected_objects):
        object_str = ", ".join(detected_objects)

        prompt = f"""
You are a image captioner . Enhance the following caption based on the list of detected objects.

Base caption:
"{base_caption}"

Detected objects:
{object_str}

Make the caption more descriptive and natural. Reply with only the new enhanced caption. Don't add extra details.
"""

        # Run Ollama chat command
        result = subprocess.run(
            ["ollama", "run", self.model],
            input=prompt.encode(),
            capture_output=True
        )

        enhanced_caption = result.stdout.decode().strip().replace('"', '')
        return enhanced_caption
