import gradio as gr
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import torch
import numpy as np
import cv2
from PIL import Image

torch.hub.download_url_to_file('http://images.cocodataset.org/val2017/000000039769.jpg', 'cats.jpg')

feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

def write_depth(depth, bits):
  depth_min = depth.min()
  depth_max = depth.max()

  max_val = (2 ** (8 * bits)) - 1

  if depth_max - depth_min > np.finfo("float").eps:
      out = max_val * (depth - depth_min) / (depth_max - depth_min)
  else:
      out = np.zeros(depth.shape, dtype=depth.dtype)

  cv2.imwrite("result.png", out.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
  
  return

def process_image(image):
    # prepare image for the model
    encoding = feature_extractor(image, return_tensors="pt")
    
    # forward pass
    with torch.no_grad():
       outputs = model(**encoding)
    
    predicted_depth = outputs.predicted_depth
    
    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(1),
                        size=image.size[::-1],
                        mode="bicubic",
                        align_corners=False,
                 )
    prediction = prediction.squeeze().cpu().numpy()
    
    # write predicted depth to file
    write_depth(prediction, bits=2)
    
    result = Image.open("result.png")
    
    return result
    
title = "Interactive demo: DPT"
description = "Demo for Intel's DPT, a Dense Prediction Transformer for state-of-the-art dense prediction tasks such as semantic segmentation and depth estimation."
examples =[['cats.jpg']]

css = ".output-image, .input-image {height: 40rem !important; width: 100% !important;}"
#css = "@media screen and (max-width: 600px) { .output_image, .input_image {height:20rem !important; width: 100% !important;} }"
# css = ".output_image, .input_image {height: 600px !important}"

iface = gr.Interface(fn=process_image, 
                     inputs=gr.inputs.Image(type="pil"), 
                     outputs=gr.outputs.Image(type="pil", label="predicted depth"),
                     title=title,
                     description=description,
                     examples=examples,
                     css=css,
                     enable_queue=True)
iface.launch(debug=True)