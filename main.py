from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Free Sketch Converter")

# CORS setup
origins = [
    "http://localhost:5173",
    "https://your-frontend-domain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def photo_to_sketch(file_bytes, style: str = "pencil", blur_strength: int = 21):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resize if too large
    max_dim = 1024
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))

    # Default: enhanced realistic sketch pipeline
    # Smooth color while preserving edges (reduce noise but keep structure)
    color = img.copy()
    for _ in range(2):
        color = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)

    # Normalize blur_strength to a valid odd kernel size
    k = max(1, int(blur_strength))
    if k % 2 == 0:
        k += 1

    # Convert a working copy
    work = img.copy()

    # Resize if too large (already done above) - keep as-is

    # Pencil style: classic grayscale pencil sketch using color dodge
    if style == "pencil":
        # Slight smoothing before conversion
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        gray_inv = 255 - gray
        blur = cv2.GaussianBlur(gray_inv, (k, k), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)

        # Local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        sketch = clahe.apply(sketch)

        # Optional gentle unsharp to bring out texture
        gauss = cv2.GaussianBlur(sketch, (0, 0), sigmaX=1.5)
        sketch = cv2.addWeighted(sketch, 1.2, gauss, -0.2, 0)

        pil_img = Image.fromarray(sketch)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return buf

    # Charcoal style: heavier strokes, darker contrast, smudged look
    if style == "charcoal":
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        # stronger blur for smudging
        k_char = max(3, k * 2 + 1)
        gray_inv = 255 - gray
        blur = cv2.GaussianBlur(gray_inv, (k_char, k_char), 0)
        base = cv2.divide(gray, 255 - blur, scale=256)

        # Emphasize edges
        edges = cv2.Canny(gray, 30, 120)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Darken regions where edges are present to simulate charcoal strokes
        edges_mask = (edges > 0).astype(np.uint8)
        # Blend base with darker strokes
        dark = (base.astype(np.float32) * 0.6).astype(np.uint8)
        charcoal = np.where(edges_mask[..., None], dark[..., None], base[..., None])
        charcoal = charcoal.squeeze()

        # Add subtle grain for texture
        noise = (np.random.randn(*charcoal.shape) * 8).astype(np.int16)
        charcoal = np.clip(charcoal.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        pil_img = Image.fromarray(charcoal)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return buf

    # Cartoon style: smooth regions + bold edges (grayscale cartoon/sketch)
    if style == "cartoon":
        # Smooth colors then desaturate to grayscale-like sketch
        try:
            smooth = cv2.pyrMeanShiftFiltering(work, sp=21, sr=51)
        except Exception:
            smooth = work.copy()
            for _ in range(2):
                smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=75, sigmaSpace=75)

        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        # Create bold edge map
        blur_small = cv2.GaussianBlur(gray, (k, k), 0)
        edges = cv2.adaptiveThreshold(
            blur_small, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, max(3, k), 2
        )
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Combine gray shading with edges: darken along edges
        edges_inv = 255 - edges
        shade = (gray.astype(np.float32) * (edges_inv.astype(np.float32) / 255.0)).astype(np.uint8)

        pil_img = Image.fromarray(shade)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return buf

    # Fallback to pencil if unknown style
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray_inv = 255 - gray
    blur = cv2.GaussianBlur(gray_inv, (k, k), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    pil_img = Image.fromarray(sketch)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf

@app.post("/sketch")
async def create_sketch(file: UploadFile = File(...), style: str = Form("sketch")):
    file_bytes = await file.read()
    sketch_img = photo_to_sketch(file_bytes, style=style)
    return StreamingResponse(sketch_img, media_type="image/png")
