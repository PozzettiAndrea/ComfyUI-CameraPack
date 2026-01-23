"""
ComfyUI-CameraPack: Convert camera parameters between formats.

Converts DepthAnythingV3 extrinsics/intrinsics to ComfyUI's Load3DCamera format.
"""

import torch
import numpy as np


class DA3ToLoad3DCamera:
    """
    Convert DepthAnythingV3 camera parameters to ComfyUI Load3DCamera format.

    DA3 outputs:
    - extrinsics: [B, 4, 4] pose matrix (rotation + translation)
    - intrinsics: [B, 3, 3] matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

    Load3DCamera expects:
    - position: {x, y, z} camera position in world space
    - target: {x, y, z} look-at point
    - zoom: int (derived from focal length)
    - cameraType: "perspective" or "orthographic"
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "extrinsics": ("EXTRINSICS",),
                "intrinsics": ("INTRINSICS",),
            },
            "optional": {
                "target_distance": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 100.0,
                    "step": 0.1,
                    "tooltip": "Distance from camera to target point along view direction"
                }),
                "zoom_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Scale factor for zoom (multiply focal length)"
                }),
            }
        }

    RETURN_TYPES = ("LOAD3D_CAMERA",)
    RETURN_NAMES = ("camera_info",)
    FUNCTION = "convert"
    CATEGORY = "3d/camera"

    def convert(self, extrinsics, intrinsics, target_distance=1.0, zoom_scale=1.0):
        """
        Convert DA3 camera parameters to Load3DCamera format.

        The extrinsics matrix is a 4x4 world-to-camera transform:
        [R | t]
        [0 | 1]

        Camera position in world space: pos = -R^T @ t
        Camera forward direction: forward = R^T @ [0, 0, 1]
        """
        # Handle batch dimension - take first item
        if isinstance(extrinsics, torch.Tensor):
            ext = extrinsics[0].cpu().numpy() if extrinsics.dim() > 2 else extrinsics.cpu().numpy()
        else:
            ext = np.array(extrinsics)

        if isinstance(intrinsics, torch.Tensor):
            intr = intrinsics[0].cpu().numpy() if intrinsics.dim() > 2 else intrinsics.cpu().numpy()
        else:
            intr = np.array(intrinsics)

        # Flatten and handle different extrinsics formats
        ext = ext.flatten()
        if ext.size == 16:
            ext = ext.reshape(4, 4)
        elif ext.size == 12:
            # 3x4 matrix - add [0, 0, 0, 1] row to make 4x4
            ext = ext.reshape(3, 4)
            ext = np.vstack([ext, [0, 0, 0, 1]])
        else:
            raise ValueError(f"Unexpected extrinsics size: {ext.size}, expected 12 or 16")

        # Ensure intrinsics is 3x3
        intr = intr.flatten()
        if intr.size == 9:
            intr = intr.reshape(3, 3)
        else:
            raise ValueError(f"Unexpected intrinsics size: {intr.size}, expected 9")

        # Extract rotation and translation from extrinsics
        R = ext[:3, :3]  # 3x3 rotation matrix
        t = ext[:3, 3]   # 3x1 translation vector

        # Camera position in world space: pos = -R^T @ t
        R_T = R.T
        camera_pos = -R_T @ t

        # Camera forward direction (Z-axis in camera space transformed to world)
        # In OpenGL/standard convention, camera looks down -Z, so forward is [0, 0, -1]
        # In some conventions it's [0, 0, 1]. We'll use [0, 0, 1] and let user adjust.
        forward = R_T @ np.array([0, 0, 1])

        # Target point is camera position + forward * distance
        target_pos = camera_pos + forward * target_distance

        # Extract focal length from intrinsics
        fx = float(intr[0, 0])
        fy = float(intr[1, 1])
        focal_length = (fx + fy) / 2.0

        # Compute zoom - this is somewhat arbitrary, scale to reasonable range
        zoom = int(focal_length * zoom_scale)
        zoom = max(1, min(zoom, 1000))  # Clamp to reasonable range

        # Build camera_info dict
        camera_info = {
            "position": {
                "x": float(camera_pos[0]),
                "y": float(camera_pos[1]),
                "z": float(camera_pos[2]),
            },
            "target": {
                "x": float(target_pos[0]),
                "y": float(target_pos[1]),
                "z": float(target_pos[2]),
            },
            "zoom": zoom,
            "cameraType": "perspective",
        }

        return (camera_info,)


class CreateLoad3DCamera:
    """
    Manually create a Load3DCamera from position, target, and zoom values.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pos_x": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "pos_y": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "pos_z": ("FLOAT", {"default": 5.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "target_x": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "target_y": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "target_z": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "zoom": ("INT", {"default": 50, "min": 1, "max": 1000, "step": 1}),
                "camera_type": (["perspective", "orthographic"], {"default": "perspective"}),
            }
        }

    RETURN_TYPES = ("LOAD3D_CAMERA",)
    RETURN_NAMES = ("camera_info",)
    FUNCTION = "create"
    CATEGORY = "3d/camera"

    def create(self, pos_x, pos_y, pos_z, target_x, target_y, target_z, zoom, camera_type):
        camera_info = {
            "position": {"x": pos_x, "y": pos_y, "z": pos_z},
            "target": {"x": target_x, "y": target_y, "z": target_z},
            "zoom": zoom,
            "cameraType": camera_type,
        }
        return (camera_info,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "DA3ToLoad3DCamera": DA3ToLoad3DCamera,
    "CreateLoad3DCamera": CreateLoad3DCamera,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DA3ToLoad3DCamera": "DA3 to Load3D Camera",
    "CreateLoad3DCamera": "Create Load3D Camera",
}
