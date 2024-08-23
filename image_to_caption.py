
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

def generate_captioning(image_path, question):
    raw_image = Image.open(image_path).convert('RGB')

 
    inputs = processor(raw_image, question, return_tensors="tf")

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True).strip())