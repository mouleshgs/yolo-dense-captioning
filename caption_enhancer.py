# caption_enhancer.py

import subprocess
import json

# import shutil
# print("Ollama path:", shutil.which("ollama"))


class CaptionEnhancer:
    def __init__(self, model_name="llama3.2:latest"):
        self.model = model_name

    def enhance(self, base_caption, detected_objects):
        object_str = ", ".join(detected_objects)
        prompt = f"""
You are an expert image captioner. Enhance the following caption by incorporating the detected objects naturally and making it more vivid and descriptive.

Base caption:
"{base_caption}"

Detected objects:
{object_str}

Generate a single enhanced caption that is natural, fluent, and descriptive. Include the objects appropriately. Reply with only the enhanced caption, without extra explanations, instructions, or formatting.
"""

        print("ollama running!!")
        # Run Ollama chat command
        result = subprocess.run(
            ["ollama", "run", self.model],
            input=prompt.encode(),
            capture_output=True
        )

        enhanced_caption = result.stdout.decode().strip().replace('"', '')
        return enhanced_caption
