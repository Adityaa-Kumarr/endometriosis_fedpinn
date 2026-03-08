"""
Image encoder for uterus / ultrasound / pelvic imaging.
Produces 128-d embeddings to feed the model's ultrasound stream (us_dim=128).
Uses a pretrained CNN backbone so the system can "understand" images, not just OCR text.
"""
import torch
import torch.nn as nn

# ImageNet normalization for pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_SIZE = 224
OUTPUT_DIM = 128


def _get_encoder(backbone="resnet18", output_dim=OUTPUT_DIM):
    """Build encoder: pretrained backbone + projection to output_dim."""
    try:
        from torchvision import models
    except ImportError:
        return None
    if backbone == "resnet18":
        try:
            from torchvision.models import ResNet18_Weights
            m = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except (ImportError, AttributeError):
            m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, output_dim)
    else:
        return None
    m.eval()
    return m


def image_to_tensor(pil_image, size=DEFAULT_SIZE, device=None):
    """Convert PIL Image to normalized batch tensor [1, 3, H, W]."""
    import numpy as np
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    img = pil_image.resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0
    for i in range(3):
        arr[..., i] = (arr[..., i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]
    # HWC -> CHW, add batch
    arr = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    if device is not None:
        arr = arr.to(device)
    return arr


def encode_image(pil_image, encoder=None, device=None, output_dim=OUTPUT_DIM):
    """
    Encode a single PIL Image to a 128-d vector for the ultrasound stream.
    Returns numpy array of shape (1, 128) or None if encoder unavailable.
    """
    if encoder is None:
        encoder = _get_encoder(output_dim=output_dim)
    if encoder is None:
        return None
    if device is None:
        device = next(encoder.parameters()).device
    encoder = encoder.to(device)
    encoder.eval()
    x = image_to_tensor(pil_image, size=DEFAULT_SIZE, device=device)
    with torch.no_grad():
        emb = encoder(x)
    return emb.cpu().float().numpy()


def load_image_encoder(device=None):
    """Load and return the image encoder (for caching in app). Returns None if torchvision unavailable."""
    enc = _get_encoder(output_dim=OUTPUT_DIM)
    if enc is not None and device is not None:
        enc = enc.to(device)
    return enc
