"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Module.
Provides visual explanations for Burn Injuries Diagnostic predictions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Computes 2D normalized Grad-CAM heatmap for the specified or highest-scoring class.
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        gradients = self.gradients[0].cpu().numpy()     # [C, H, W]
        activations = self.activations[0].cpu().numpy() # [C, H, W]

        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))       # [C]

        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU to retain only features that have positive impact on the class
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(cam)

        return cam

    @staticmethod
    def overlay_heatmap(
        pil_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET
    ) -> Image.Image:
        """
        Resizes heatmap to image dimensions, applies colormap, and superimposes over image.
        """
        orig_img = np.array(pil_image.convert("RGB"))
        h, w = orig_img.shape[:2]

        # Resize heatmap to original image dimensions
        resized_cam = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Scale to 0-255 uint8
        heatmap_uint8 = np.uint8(255 * resized_cam)
        
        # Apply colormap
        colored_cam = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_cam = cv2.cvtColor(colored_cam, cv2.COLOR_BGR2RGB)

        # Blend
        blended = np.float32(colored_cam) * alpha + np.float32(orig_img) * (1 - alpha)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        return Image.fromarray(blended), Image.fromarray(colored_cam)
