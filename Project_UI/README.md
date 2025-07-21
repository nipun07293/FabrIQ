# Fashion Discovery & AI Style Transfer Web App

A modern web application for exploring a large fashion dataset, powered by AI-based recommendations and neural style transfer.  
Users can search, filter, and discover clothing items, find visually similar products using text or image queries, and apply artistic styles to their own clothing images.

---

## Features

- **Fashion Product Browser:**  
  Browse, search, and filter a large collection of clothing images with rich metadata (category, brand, color, pattern, material, etc.).

- **AI-Powered Recommendations:**  
  - Find similar products using text queries (e.g., "blue floral dress").
  - Upload an image to find visually similar items.
  - Click "Find Similar" on any product to get recommendations.

- **Neural Style Transfer:**  
  - Upload a clothing image and a style image.
  - The app uses deep learning (VGG19, YOLOv8 segmentation) to blend the style onto the clothing, preserving the clothing's shape.
  - Results are shown in the browser and can be downloaded.

- **Modern, Responsive UI:**  
  - Built with Flask, HTML/CSS, and JavaScript.
  - Clean, user-friendly interface for both product discovery and style transfer.

---

## Project Structure

```
.
├── app.py                # Main Flask app
├── recommender.py        # AI recommendation engine (OpenCLIP, FAISS, graph-based)
├── style_transfer.py     # Neural style transfer (VGG19, YOLOv8 segmentation)
├── data.csv              # Product metadata
├── image/                # Product images
├── model_data/           # Precomputed model files (FAISS, OpenCLIP, VGG, etc.)
├── results/              # Style transfer output images
├── uploads/              # Temporary upload storage
├── static/               # JS and CSS files
├── templates/            # HTML templates
```

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Project_UI
```

### 2. Install Python dependencies

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required packages:

```bash
pip install flask flask-cors pandas numpy torch torchvision ultralytics faiss-cpu open_clip_torch pillow opencv-python
```

> **Note:**  
> - For GPU acceleration, install the appropriate versions of `torch` and `torchvision` for your CUDA version.
> - If you encounter issues with `faiss-cpu`, try `faiss-gpu` if you have a compatible GPU.

### 3. Download/prepare model files

- Place the following files in the `model_data/` directory:
  - `faiss_openclip_cloth.idx`
  - `faiss_openclip_person.idx`
  - `openclip_caption_paths.pkl`
  - `graphs/unified_image_graph.pkl`
  - `vgg19-dcbb9e9d.pth` (VGG19 weights)
- Place `yolov8l-seg.pt` in the project root.

> **Note:**  
> These files are large and may need to be generated or downloaded separately.

### 4. Prepare data

- Place your product images in the `image/` directory.
- Ensure `data.csv` contains metadata for your products (see sample format in the file).

---

## Usage

### Start the Flask server

```bash
python app.py
```

- The app will be available at local host. 
- On first use, AI models will be loaded into memory (may take some time).

### Main Pages

- **Product Store:**  
  - Search, filter, and browse products.
  - Use the search bar or upload an image to find similar styles.
  - Click "Find Similar" on any product for recommendations.

- **AI Style Transfer:**  
  - Click "Try AI Style Transfer ✨" in the header.
  - Upload a clothing image and a style image.
  - Click "Start Styling" to generate a stylized result.

---

## File/Folder Descriptions

- `app.py` — Flask backend, API endpoints, and app configuration.
- `recommender.py` — Handles AI-based recommendations using OpenCLIP, FAISS, and a graph-based approach.
- `style_transfer.py` — Performs neural style transfer using VGG19 and YOLOv8 segmentation.
- `templates/` — HTML templates for the main UI and style transfer page.
- `static/js/` — JavaScript for frontend interactivity.
- `model_data/` — Precomputed model files and indices (see above).
- `image/` — Product images.
- `results/` — Output images from style transfer.
- `uploads/` — Temporary storage for uploaded images.

---

## Requirements

- Python 3.8+
- See "Installation" for required Python packages.

---


## License

MIT License (or specify your license here) 