
![1.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/_VFyr_efAG3NA1_GlHa87.png)

# **RESISC45-SigLIP2**

> **RESISC45-SigLIP2** is a vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for **multi-label** image classification. It is specifically trained to recognize and tag multiple land use and land cover scene categories from the **RESISC45** dataset using the **SiglipForImageClassification** architecture.

```py
Classification Report:
                       precision    recall  f1-score   support

             airplane     0.9830    0.9900    0.9865       700
              airport     0.9461    0.9529    0.9495       700
     baseball diamond     0.9802    0.9886    0.9844       700
     basketball court     0.9516    0.9271    0.9392       700
                beach     0.9914    0.9900    0.9907       700
               bridge     0.9730    0.9771    0.9751       700
            chaparral     0.9957    0.9986    0.9971       700
               church     0.7949    0.8971    0.8430       700
    circular farmland     0.9914    0.9914    0.9914       700
                cloud     0.9957    0.9871    0.9914       700
      commercial area     0.9231    0.8229    0.8701       700
    dense residential     0.9355    0.8914    0.9129       700
               desert     0.9821    0.9414    0.9613       700
               forest     0.9652    0.9514    0.9583       700
              freeway     0.9344    0.9571    0.9457       700
          golf course     0.9759    0.9843    0.9801       700
   ground track field     0.9623    0.9857    0.9739       700
               harbor     0.9885    0.9843    0.9864       700
      industrial area     0.9505    0.9043    0.9268       700
         intersection     0.9855    0.9686    0.9769       700
               island     0.9871    0.9829    0.9850       700
                 lake     0.9440    0.9629    0.9533       700
               meadow     0.9564    0.9400    0.9481       700
   medium residential     0.8602    0.9314    0.8944       700
     mobile home park     0.9610    0.9500    0.9555       700
             mountain     0.9388    0.9429    0.9408       700
             overpass     0.9614    0.9614    0.9614       700
               palace     0.8455    0.8286    0.8369       700
          parking lot     0.9899    0.9757    0.9827       700
              railway     0.9407    0.9071    0.9236       700
      railway station     0.9104    0.9143    0.9123       700
 rectangular farmland     0.9572    0.9271    0.9419       700
                river     0.9281    0.9586    0.9431       700
           roundabout     0.9914    0.9871    0.9893       700
               runway     0.9669    0.9586    0.9627       700
              sea ice     0.9957    0.9943    0.9950       700
                 ship     0.9558    0.9886    0.9719       700
             snowberg     0.9886    0.9900    0.9893       700
   sparse residential     0.9238    0.9700    0.9463       700
              stadium     0.9716    0.9757    0.9736       700
         storage tank     0.9787    0.9829    0.9808       700
         tennis court     0.9326    0.9486    0.9405       700
              terrace     0.9372    0.9586    0.9477       700
thermal power station     0.9482    0.9671    0.9576       700
              wetland     0.9444    0.8986    0.9209       700

             accuracy                         0.9532     31500
            macro avg     0.9538    0.9532    0.9532     31500
         weighted avg     0.9538    0.9532    0.9532     31500
```

---

## **Label Space: 45 Scene Categories**

The model predicts the presence of one or more of the following **45 scene categories**:

```
Class 0: "airplane"
Class 1: "airport"
Class 2: "baseball diamond"
Class 3: "basketball court"
Class 4: "beach"
Class 5: "bridge"
Class 6: "chaparral"
Class 7: "church"
Class 8: "circular farmland"
Class 9: "cloud"
Class 10: "commercial area"
Class 11: "dense residential"
Class 12: "desert"
Class 13: "forest"
Class 14: "freeway"
Class 15: "golf course"
Class 16: "ground track field"
Class 17: "harbor"
Class 18: "industrial area"
Class 19: "intersection"
Class 20: "island"
Class 21: "lake"
Class 22: "meadow"
Class 23: "medium residential"
Class 24: "mobile home park"
Class 25: "mountain"
Class 26: "overpass"
Class 27: "palace"
Class 28: "parking lot"
Class 29: "railway"
Class 30: "railway station"
Class 31: "rectangular farmland"
Class 32: "river"
Class 33: "roundabout"
Class 34: "runway"
Class 35: "sea ice"
Class 36: "ship"
Class 37: "snowberg"
Class 38: "sparse residential"
Class 39: "stadium"
Class 40: "storage tank"
Class 41: "tennis court"
Class 42: "terrace"
Class 43: "thermal power station"
Class 44: "wetland"
```

---

## **Install dependencies**

```bash
pip install -q transformers torch pillow gradio
```

---

## **Inference Code**

```python
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
```

---

## **Intended Use**

The **RESISC45-SigLIP2** model is ideal for multi-label classification tasks involving remote sensing imagery. Use cases include:

- **Remote Sensing Analysis** – Label elements in aerial/satellite images.
- **Urban Planning** – Identify urban structures and landscape features.
- **Geospatial Intelligence** – Aid in automated image interpretation pipelines.
- **Environmental Monitoring** – Track natural landforms and changes.
