import gradio as gr
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import open3d as o3d

torch.hub.download_url_to_file('http://images.cocodataset.org/val2017/000000039769.jpg', 'cats.jpg')

feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

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
                 ).squeeze()
    output = prediction.cpu().numpy()
    depth_image = (output * 255 / np.max(output)).astype('uint8')
    # create_obj(formatted, "test.obj")
    create_obj_2(np.array(image), depth_image)
    # img = Image.fromarray(formatted)
    return "output.gltf"
    
    # return result

    # gradio.inputs.Image3D(self, label=None, optional=False)

def create_obj_2(rgb_image, depth_image):
    depth_o3d = o3d.geometry.Image(depth_image)
    image_o3d = o3d.geometry.Image(rgb_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d)
    w = int(depth_image.shape[0])
    h = int(depth_image.shape[1])

    FOV = np.pi/4
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    camera_intrinsic.set_intrinsics(w, h, w*0.5, h*0.5, w*0.5, h*0.5 )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,camera_intrinsic)
    print('normals')
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
    pcd.estimate_normals()
    # pcd.orient_normals_consistent_tangent_plane(100)
    print('run Poisson surface reconstruction')
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    print(mesh)
    o3d.io.write_triangle_mesh("output.gltf",mesh,write_triangle_uvs=True)
    return "output.gltf"
    
title = "Interactive demo: DPT + 3D"
description = "Demo for Intel's DPT, a Dense Prediction Transformer for state-of-the-art dense prediction tasks such as semantic segmentation and depth estimation."
examples =[['cats.jpg']]

iface = gr.Interface(fn=process_image, 
                     inputs=gr.inputs.Image(type="pil"), 
                     outputs=gr.outputs.Image3D(label="predicted depth", clear_color=[1.0,1.0,1.0,1.0]),
                     title=title,
                     description=description,
                     examples=examples,
                     allow_flagging="never",
                     enable_queue=True)
iface.launch(debug=True)