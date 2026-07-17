# Window Vision

A static, browser-based window detector powered by a custom YOLO model exported to ONNX. It accepts
images and videos, supports device camera capture, and can run continuous detection on video or a live
camera feed. All inference happens in the browser; media is never uploaded to a server.

## Run locally

The site must be served over HTTP because browsers do not allow ONNX model loading from a `file://` URL.

```bash
python -m http.server 8000
```

Open <http://localhost:8000>. Camera access works on `localhost`; a deployed camera experience requires
HTTPS, which the hosting services below provide automatically.

## Free deployment

This repository is a static site with no build step. Deploy the repository root to any static host:

- **GitHub Pages:** Repository Settings → Pages → deploy from the `main` branch and `/ (root)` folder.
- **Cloudflare Pages:** Connect the repository, leave the build command empty, and use `/` as the output directory.
- **Netlify:** Import the repository and deploy without a build command.

The ONNX model is about 36 MB, below GitHub's 100 MB per-file limit. ONNX Runtime Web is pinned to
version 1.27.0 and loaded from jsDelivr. Visitors therefore need network access the first time the runtime
and model are loaded; browser caching makes later visits faster.

## Files

- `best.pt` — original Ultralytics model weights
- `best.onnx` — exported browser model, input `1×3×640×640`, output `1×300×6`
- `index.html` — accessible application structure
- `styles.css` — responsive presentation
- `app.js` — media capture, preprocessing, ONNX inference, and bounding-box rendering

## Re-export the model

With Ultralytics installed:

```bash
yolo export model=best.pt format=onnx imgsz=640 opset=12 simplify=False dynamic=False
```

The JavaScript postprocessor expects the end-to-end model output `[batch, 300, 6]`, where each row is
`[x1, y1, x2, y2, confidence, class]`. If the architecture or export format changes, update
`parseDetections()` in `app.js`.

## Privacy and browser support

- Media remains in browser memory and is not transmitted by this app.
- Inference uses ONNX Runtime Web's WebAssembly execution provider for broad browser support.
- A current Chrome, Edge, Firefox, or Safari release is recommended.
- Camera access depends on browser permission and requires HTTPS outside `localhost`.

The exported model includes Ultralytics AGPL-3.0 metadata. Review the Ultralytics licensing terms before
public or commercial deployment.
