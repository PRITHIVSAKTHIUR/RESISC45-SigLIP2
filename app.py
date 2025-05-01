import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/RESISC45-SigLIP2"  # Update to your actual Hugging Face model path
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Label map
id2label = {
    "0": "airplane", "1": "airport", "2": "baseball diamond", "3": "basketball court", "4": "beach",
    "5": "bridge", "6": "chaparral", "7": "church", "8": "circular farmland", "9": "cloud",
    "10": "commercial area", "11": "dense residential", "12": "desert", "13": "forest", "14": "freeway",
    "15": "golf course", "16": "ground track field", "17": "harbor", "18": "industrial area", "19": "intersection",
    "20": "island", "21": "lake", "22": "meadow", "23": "medium residential", "24": "mobile home park",
    "25": "mountain", "26": "overpass", "27": "palace", "28": "parking lot", "29": "railway",
    "30": "railway station", "31": "rectangular farmland", "32": "river", "33": "roundabout", "34": "runway",
    "35": "sea ice", "36": "ship", "37": "snowberg", "38": "sparse residential", "39": "stadium",
    "40": "storage tank", "41": "tennis court", "42": "terrace", "43": "thermal power station", "44": "wetland"
}

def classify_resisc_image(image):
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze().tolist()
    
    threshold = 0.5
    predictions = {
        id2label[str(i)]: round(probs[i], 3)
        for i in range(len(probs)) if probs[i] >= threshold
    }

    return predictions or {"None Detected": 0.0}

# Gradio Interface
iface = gr.Interface(
    fn=classify_resisc_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Predicted Scene Categories"),
    title="RESISC45-SigLIP2",
    description="Upload a satellite image to detect multiple land use and land cover categories (e.g., airport, forest, mountain)."
)

if __name__ == "__main__":
    iface.launch()
