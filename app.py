import os
from pathlib import Path

import gradio as gr
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from transformers import DPTForDepthEstimation, DPTImageProcessor

# Initialize the image processor and depth estimation model
image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large", ignore_mismatched_sizes=True)

def process_image(image_path, resized_width=800, z_scale=208):
    """
    Processes the input image to generate a depth map and a 3D mesh reconstruction.

    Args:
        image_path (str): The file path to the input image.

    Returns:
        list: A list containing the depth image, 3D mesh reconstruction, and GLTF file path.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise ValueError("Image file not found")

    # Load and resize the image
    image_raw = Image.open(image_path).convert("RGB")
    print(f"Original size: {image_raw.size}")
    resized_height = int(resized_width * image_raw.size[1] / image_raw.size[0])
    image = image_raw.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    print(f"Resized size: {image.size}")

    # Prepare image for the model
    encoding = image_processor(image, return_tensors="pt")

    # Perform depth estimation
    with torch.no_grad():
        outputs = depth_model(**encoding)
        predicted_depth = outputs.predicted_depth

    # Interpolate depth to match the image size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(image.height, image.width),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    # Normalize the depth image to 8-bit
    if torch.cuda.is_available():
        prediction = prediction.numpy()
    else:
        prediction = prediction.cpu().numpy()
    depth_min, depth_max = prediction.min(), prediction.max()
    depth_image = ((prediction - depth_min) / (depth_max - depth_min) * 255).astype("uint8")

    try:
        gltf_path = create_3d_obj(np.array(image), prediction, image_path, depth=10, z_scale=z_scale)
    except Exception:
        gltf_path = create_3d_obj(np.array(image), prediction, image_path, depth=8, z_scale=z_scale)

    img = Image.fromarray(depth_image)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return [img, gltf_path, gltf_path]

def create_3d_obj(rgb_image, raw_depth, image_path, depth=10, z_scale=200):
    """
    Creates a 3D object from RGB and depth images.

    Args:
        rgb_image (np.ndarray): The RGB image as a NumPy array.
        raw_depth (np.ndarray): The raw depth data.
        image_path (Path): The path to the original image.
        depth (int, optional): Depth parameter for Poisson reconstruction. Defaults to 10.
        z_scale (float, optional): Scaling factor for the Z-axis. Defaults to 200.

    Returns:
        str: The file path to the saved GLTF model.
    """
    # Normalize the depth image
    depth_image = ((raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min()) * 255).astype("uint8")
    depth_o3d = o3d.geometry.Image(depth_image)
    image_o3d = o3d.geometry.Image(rgb_image)

    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        image_o3d, depth_o3d, convert_rgb_to_intensity=False
    )

    height, width = depth_image.shape

    # Define camera intrinsics
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        fx=z_scale,
        fy=z_scale,
        cx=width / 2.0,
        cy=height / 2.0,
    )

    # Generate point cloud from RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

    # Scale the Z dimension
    points = np.asarray(pcd.points)
    depth_scaled = ((raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min())) * (z_scale*100)
    z_values = depth_scaled.flatten()[:len(points)]
    points[:, 2] *= z_values
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate and orient normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=60)
    )
    pcd.orient_normals_towards_camera_location(camera_location=np.array([0.0, 0.0, 1.5 ]))

    # Apply transformations
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
    pcd.transform([[-1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    # Perform Poisson surface reconstruction
    print(f"Running Poisson surface reconstruction with depth {depth}")
    mesh_raw, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=0, scale=1.1, linear_fit=True
    )
    print(f"Raw mesh vertices: {len(mesh_raw.vertices)}, triangles: {len(mesh_raw.triangles)}")

    # Simplify the mesh using vertex clustering
    voxel_size = max(mesh_raw.get_max_bound() - mesh_raw.get_min_bound()) / (max(width, height) * 0.8)
    mesh = mesh_raw.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average,
    )
    print(f"Simplified mesh vertices: {len(mesh.vertices)}, triangles: {len(mesh.triangles)}")

    # Crop the mesh to the bounding box of the point cloud
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh_crop = mesh.crop(bbox)

    # Save the mesh as a GLTF file
    temp_dir = Path.cwd() / "models"
    temp_dir.mkdir(exist_ok=True)
    gltf_path = str(temp_dir / f"{image_path.stem}.gltf")
    o3d.io.write_triangle_mesh(gltf_path, mesh_crop, write_triangle_uvs=True)
    return gltf_path


# Define Gradio interface components
title = "Zero-Shot Depth Estimation with DPT + 3D Point Cloud"
description = (
    "This demo by <a href='https://huggingface.co/Surn' target='_blank'>Charles Fettinger</a> is an update to the original "
    "<a href='https://huggingface.co/spaces/nielsr/dpt-depth-estimation' target='_blank'>DPT Demo</a>. "
    "It uses the DPT model to predict the depth of an image and then uses 3D Point Cloud to create a 3D object."
)
# Create Gradio sliders for resized_width and z_scale
resized_width_slider = gr.Slider(
    minimum=256,
    maximum=1760,
    step=16,
    value=800,
    label="Resized Width",
    info="Resize the image based upon width, preserving the aspect ratio"
)

z_scale_slider = gr.Slider(
    minimum=0.2,
    maximum=3.0,
    step=0.01,
    value=0.5,
    label="Z-Scale",
    info="Scale the amount of 3D model depth, short or tall (can distort)."
)
examples = [["examples/" + img] for img in os.listdir("examples/")]

process_image.zerogpu = True
#gr.set_static_paths(paths=["models/","examples/"])
iface = gr.Interface(
    fn=process_image,
        inputs=[
        gr.Image(type="filepath", label="Input Image"),
        resized_width_slider,
        z_scale_slider
    ],
    outputs=[
        gr.Image(label="Predicted Depth", type="pil"),
        gr.Model3D(label="3D Mesh Reconstruction", clear_color=[1.0, 1.0, 1.0, 1.0]),
        gr.File(label="3D GLTF"),
    ],
    title=title,
    description=description,
    examples=examples,
    examples_per_page=15,
    flagging_mode=None,
    allow_flagging="never",
    cache_examples=False,
    delete_cache=(86400,86400),
    theme="Surn/Beeuty",
    show_progress = 'full'
)

if __name__ == "__main__":
    iface.launch(debug=True, show_api=False, favicon_path="./favicon.ico", allowed_paths=["models/","examples/"])
