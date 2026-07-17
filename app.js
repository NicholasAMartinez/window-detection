"use strict";

const MODEL_URL = "best.onnx";
const MODEL_SIZE = 640;
const CLASS_NAMES = ["window"];
const BOX_COLOR = "#b8ff50";
const ORT_CDN = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";

const elements = {
  viewer: document.querySelector("#viewer"),
  canvas: document.querySelector("#outputCanvas"),
  video: document.querySelector("#sourceVideo"),
  image: document.querySelector("#sourceImage"),
  empty: document.querySelector("#emptyState"),
  loading: document.querySelector("#loadingOverlay"),
  loadingTitle: document.querySelector("#loadingTitle"),
  loadingDetail: document.querySelector("#loadingDetail"),
  drop: document.querySelector("#dropOverlay"),
  input: document.querySelector("#mediaInput"),
  camera: document.querySelector("#cameraButton"),
  cameraEmpty: document.querySelector("#cameraEmptyButton"),
  capture: document.querySelector("#captureButton"),
  play: document.querySelector("#playButton"),
  playLabel: document.querySelector("#playLabel"),
  playIcon: document.querySelector("#playIcon"),
  sourceLabel: document.querySelector("#sourceLabel"),
  confidence: document.querySelector("#confidence"),
  confidenceValue: document.querySelector("#confidenceValue"),
  liveToggle: document.querySelector("#liveToggle"),
  badge: document.querySelector("#modelBadge"),
  count: document.querySelector("#detectionCount"),
  time: document.querySelector("#inferenceTime"),
  rate: document.querySelector("#processingRate"),
  status: document.querySelector("#statusMessage"),
};

const displayContext = elements.canvas.getContext("2d");
const modelCanvas = document.createElement("canvas");
modelCanvas.width = MODEL_SIZE;
modelCanvas.height = MODEL_SIZE;
const modelContext = modelCanvas.getContext("2d", { willReadFrequently: true });

let session = null;
let currentMode = "empty";
let mediaUrl = null;
let cameraStream = null;
let animationId = null;
let inferenceRunning = false;
let lastInferenceAt = 0;
let lastDetections = [];
let lastTransform = null;

function setStatus(message, type = "normal") {
  elements.status.textContent = message;
  elements.status.style.borderLeftColor = type === "error" ? "#d85244" : "";
}

function showLoading(title, detail) {
  elements.loadingTitle.textContent = title;
  elements.loadingDetail.textContent = detail;
  elements.loading.hidden = false;
}

function hideLoading() {
  elements.loading.hidden = true;
}

async function loadModel() {
  try {
    ort.env.wasm.wasmPaths = ORT_CDN;
    ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 1);
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    elements.badge.className = "model-badge";
    elements.badge.innerHTML = "<i></i> Ready";
    setStatus("Ready. Choose media or start the camera to detect windows.");
    if (currentMode === "image") await runImageDetection();
    if (currentMode === "video" || currentMode === "camera") startRenderLoop();
  } catch (error) {
    console.error("Model failed to load", error);
    elements.badge.className = "model-badge error";
    elements.badge.innerHTML = "<i></i> Error";
    setStatus("The model could not load. Serve this folder over HTTP and check your connection.", "error");
  } finally {
    hideLoading();
  }
}

function releaseCurrentSource() {
  cancelAnimationFrame(animationId);
  animationId = null;
  inferenceRunning = false;
  lastDetections = [];
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  elements.video.pause();
  elements.video.srcObject = null;
  elements.video.removeAttribute("src");
  elements.video.load();
  if (mediaUrl) URL.revokeObjectURL(mediaUrl);
  mediaUrl = null;
}

function prepareViewer(mode, label) {
  currentMode = mode;
  elements.viewer.classList.remove("is-empty");
  elements.empty.hidden = true;
  elements.canvas.style.display = "block";
  elements.sourceLabel.textContent = label;
  elements.capture.hidden = mode !== "camera";
  elements.play.hidden = mode !== "video";
}

function setCanvasSize(width, height) {
  if (elements.canvas.width !== width || elements.canvas.height !== height) {
    elements.canvas.width = width;
    elements.canvas.height = height;
  }
}

function drawFrame(source) {
  const width = source.videoWidth || source.naturalWidth;
  const height = source.videoHeight || source.naturalHeight;
  if (!width || !height) return;
  setCanvasSize(width, height);
  displayContext.drawImage(source, 0, 0, width, height);
  drawDetections(lastDetections);
}

function drawDetections(detections) {
  const scale = Math.max(1, elements.canvas.width / 900);
  displayContext.lineWidth = 3 * scale;
  displayContext.font = `600 ${Math.max(13, 15 * scale)}px "DM Sans", sans-serif`;
  displayContext.textBaseline = "top";

  for (const detection of detections) {
    const { x1, y1, x2, y2, score, classId } = detection;
    const label = `${CLASS_NAMES[classId] || "object"} ${Math.round(score * 100)}%`;
    const textWidth = displayContext.measureText(label).width;
    const labelHeight = 24 * scale;
    const labelY = Math.max(0, y1 - labelHeight);

    displayContext.strokeStyle = BOX_COLOR;
    displayContext.strokeRect(x1, y1, x2 - x1, y2 - y1);
    displayContext.fillStyle = BOX_COLOR;
    displayContext.fillRect(x1, labelY, textWidth + 13 * scale, labelHeight);
    displayContext.fillStyle = "#10211a";
    displayContext.fillText(label, x1 + 6 * scale, labelY + 3 * scale);
  }
}

function makeTensor(source) {
  const width = source.videoWidth || source.naturalWidth;
  const height = source.videoHeight || source.naturalHeight;
  const scale = Math.min(MODEL_SIZE / width, MODEL_SIZE / height);
  const drawWidth = Math.round(width * scale);
  const drawHeight = Math.round(height * scale);
  const padX = Math.floor((MODEL_SIZE - drawWidth) / 2);
  const padY = Math.floor((MODEL_SIZE - drawHeight) / 2);

  modelContext.fillStyle = "rgb(114, 114, 114)";
  modelContext.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  modelContext.drawImage(source, padX, padY, drawWidth, drawHeight);
  const pixels = modelContext.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
  const plane = MODEL_SIZE * MODEL_SIZE;
  const data = new Float32Array(plane * 3);
  for (let pixel = 0, rgba = 0; pixel < plane; pixel += 1, rgba += 4) {
    data[pixel] = pixels[rgba] / 255;
    data[plane + pixel] = pixels[rgba + 1] / 255;
    data[plane * 2 + pixel] = pixels[rgba + 2] / 255;
  }
  lastTransform = { scale, padX, padY, width, height };
  return new ort.Tensor("float32", data, [1, 3, MODEL_SIZE, MODEL_SIZE]);
}

function parseDetections(output) {
  const threshold = Number(elements.confidence.value) / 100;
  const values = output.data;
  const rows = output.dims.at(-2);
  const columns = output.dims.at(-1);
  const { scale, padX, padY, width, height } = lastTransform;
  const detections = [];

  if (columns !== 6) throw new Error(`Unexpected model output: ${output.dims.join("×")}`);
  for (let row = 0; row < rows; row += 1) {
    const offset = row * columns;
    const score = values[offset + 4];
    if (score < threshold) continue;
    const classId = Math.round(values[offset + 5]);
    const x1 = Math.max(0, Math.min(width, (values[offset] - padX) / scale));
    const y1 = Math.max(0, Math.min(height, (values[offset + 1] - padY) / scale));
    const x2 = Math.max(0, Math.min(width, (values[offset + 2] - padX) / scale));
    const y2 = Math.max(0, Math.min(height, (values[offset + 3] - padY) / scale));
    if (x2 > x1 && y2 > y1) detections.push({ x1, y1, x2, y2, score, classId });
  }
  return detections;
}

async function detect(source) {
  if (!session || inferenceRunning) return;
  inferenceRunning = true;
  const started = performance.now();
  try {
    const tensor = makeTensor(source);
    const result = await session.run({ [session.inputNames[0]]: tensor });
    lastDetections = parseDetections(result[session.outputNames[0]]);
    const elapsed = performance.now() - started;
    elements.count.textContent = String(lastDetections.length);
    elements.time.textContent = `${Math.round(elapsed)} ms`;
    elements.rate.textContent = `${(1000 / elapsed).toFixed(1)} FPS`;
  } catch (error) {
    console.error("Inference failed", error);
    setStatus(`Detection failed: ${error.message}`, "error");
    cancelAnimationFrame(animationId);
  } finally {
    inferenceRunning = false;
  }
}

async function runImageDetection() {
  drawFrame(elements.image);
  if (!session) {
    setStatus("Image ready. Detection will begin when the model finishes loading.");
    return;
  }
  showLoading("Inspecting image", "Running the window detector…");
  await detect(elements.image);
  drawFrame(elements.image);
  hideLoading();
  setStatus(`Detection complete. Found ${lastDetections.length} window${lastDetections.length === 1 ? "" : "s"}.`);
}

function startRenderLoop() {
  cancelAnimationFrame(animationId);
  const tick = async (timestamp) => {
    if (currentMode !== "video" && currentMode !== "camera") return;
    if (elements.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      drawFrame(elements.video);
      const shouldInfer = session && elements.liveToggle.checked && !elements.video.paused;
      if (shouldInfer && !inferenceRunning && timestamp - lastInferenceAt > 80) {
        lastInferenceAt = timestamp;
        await detect(elements.video);
      }
    }
    animationId = requestAnimationFrame(tick);
  };
  animationId = requestAnimationFrame(tick);
}

async function handleFile(file) {
  if (!file) return;
  const isImage = file.type.startsWith("image/");
  const isVideo = file.type.startsWith("video/");
  if (!isImage && !isVideo) {
    setStatus("Please choose an image or video file.", "error");
    return;
  }

  releaseCurrentSource();
  mediaUrl = URL.createObjectURL(file);
  prepareViewer(isImage ? "image" : "video", file.name);
  setStatus(`Loading ${isImage ? "image" : "video"}…`);

  if (isImage) {
    elements.image.onload = runImageDetection;
    elements.image.onerror = () => setStatus("This image could not be opened.", "error");
    elements.image.src = mediaUrl;
  } else {
    elements.video.src = mediaUrl;
    elements.video.loop = true;
    elements.video.onloadeddata = async () => {
      setCanvasSize(elements.video.videoWidth, elements.video.videoHeight);
      drawFrame(elements.video);
      try {
        await elements.video.play();
        updatePlayButton();
        startRenderLoop();
        setStatus("Video is playing with live detection.");
      } catch {
        updatePlayButton();
        startRenderLoop();
        setStatus("Video ready. Press Play to begin live detection.");
      }
    };
    elements.video.onerror = () => setStatus("This browser cannot decode that video. Try MP4 or WEBM.", "error");
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("Camera access is not supported in this browser.", "error");
    return;
  }
  releaseCurrentSource();
  prepareViewer("camera", "Live camera");
  showLoading("Starting camera", "Waiting for camera permission…");
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    elements.video.srcObject = cameraStream;
    await elements.video.play();
    setCanvasSize(elements.video.videoWidth, elements.video.videoHeight);
    startRenderLoop();
    setStatus("Camera is live. Tap Photo to freeze and inspect a frame.");
  } catch (error) {
    console.error("Camera failed", error);
    resetViewer();
    setStatus(error.name === "NotAllowedError" ? "Camera permission was not granted." : "The camera could not be started.", "error");
  } finally {
    hideLoading();
  }
}

async function capturePhoto() {
  if (currentMode !== "camera" || !elements.video.videoWidth) return;
  const snapshot = document.createElement("canvas");
  snapshot.width = elements.video.videoWidth;
  snapshot.height = elements.video.videoHeight;
  snapshot.getContext("2d").drawImage(elements.video, 0, 0);
  releaseCurrentSource();
  prepareViewer("image", "Camera photo");
  elements.image.onload = runImageDetection;
  elements.image.src = snapshot.toDataURL("image/jpeg", 0.92);
}

function resetViewer() {
  releaseCurrentSource();
  currentMode = "empty";
  elements.viewer.classList.add("is-empty");
  elements.empty.hidden = false;
  elements.canvas.style.display = "none";
  elements.capture.hidden = true;
  elements.play.hidden = true;
  elements.sourceLabel.textContent = "No media selected";
}

function updatePlayButton() {
  const paused = elements.video.paused;
  elements.playLabel.textContent = paused ? "Play" : "Pause";
  elements.playIcon.innerHTML = paused
    ? '<path d="m8 5 11 7-11 7V5Z" />'
    : '<path d="M8 5v14M16 5v14" />';
}

elements.input.addEventListener("change", () => {
  handleFile(elements.input.files[0]);
  elements.input.value = "";
});
elements.camera.addEventListener("click", startCamera);
elements.cameraEmpty.addEventListener("click", startCamera);
elements.capture.addEventListener("click", capturePhoto);
elements.play.addEventListener("click", async () => {
  if (elements.video.paused) await elements.video.play();
  else elements.video.pause();
  updatePlayButton();
});
elements.video.addEventListener("play", updatePlayButton);
elements.video.addEventListener("pause", updatePlayButton);
elements.confidence.addEventListener("input", () => {
  elements.confidenceValue.value = `${elements.confidence.value}%`;
  if (currentMode === "image") runImageDetection();
});
elements.liveToggle.addEventListener("change", () => {
  if (elements.liveToggle.checked) setStatus("Live detection is on.");
  else setStatus("Live detection is paused; the video will keep playing.");
});

for (const eventName of ["dragenter", "dragover"]) {
  elements.viewer.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.drop.hidden = false;
  });
}
for (const eventName of ["dragleave", "drop"]) {
  elements.viewer.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.drop.hidden = true;
  });
}
elements.viewer.addEventListener("drop", (event) => handleFile(event.dataTransfer.files[0]));
window.addEventListener("beforeunload", releaseCurrentSource);

showLoading("Loading model", "Downloading the ONNX window detector…");
loadModel();
