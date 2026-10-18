import * as ort from 'onnxruntime-web/webgpu';
import wasmUrl from '../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm?url';
import './style.css';

const SIZE = 320;
const assetUrl = (path) => `${import.meta.env.BASE_URL}${path.replace(/^\/+/, '')}`;
// The official TEED pipeline reads OpenCV BGR pixels and subtracts the BIPED
// training mean. Canvas gives us RGB, so preprocessing reverses R and B below.
const BGR_MEAN = [103.939, 116.779, 123.68];
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];
const DETECTORS = {
  teed: { label: 'TEED', url: 'teed-320.onnx', preprocessing: 'teed', normalizeMask: true, threshold: 0.35 },
  'teed-person-v4': { label: 'TEED Person V4 Synthetic', url: 'teed-person-v4-320.onnx', preprocessing: 'teed', normalizeMask: false, threshold: 0.46 },
  'teed-person-v3': { label: 'TEED Person V3 Silhouette', url: 'teed-person-v3-320.onnx', preprocessing: 'teed', normalizeMask: false, threshold: 0.56 },
  'teed-person-v2': { label: 'TEED Person V2', url: 'teed-person-v2-320.onnx', preprocessing: 'teed', normalizeMask: true, threshold: 0.18 },
  'teed-person': { label: 'TEED Person V1', url: 'teed-person-320.onnx', preprocessing: 'teed', normalizeMask: true, threshold: 0.18 },
  pidinet: { label: 'PiDiNet Small', url: 'pidinet-small-320.onnx', preprocessing: 'imagenet', normalizeMask: false, threshold: 0.35 },
  canny: { label: 'Canny', url: null, preprocessing: 'canny', normalizeMask: false, threshold: 0.08 }
};
const SEGMENTERS = {
  'mediapipe-multiclass': {
    label: 'MediaPipe multiclass fast 256 × 256', backend: 'MediaPipe multiclass'
  },
  mediapipe: { label: 'MediaPipe landscape', backend: 'MediaPipe' },
  'rvm-fp32': {
    label: 'RVM MobileNetV3 FP32', backend: 'RVM FP32 (WASM)',
    url: 'rvm-mobilenetv3-fp32.onnx', precision: 'float32'
  },
  'rvm-fp16': {
    label: 'RVM MobileNetV3 FP16', backend: 'RVM FP16 (WASM)',
    url: 'rvm-mobilenetv3-fp16.onnx', precision: 'float16'
  }
};

const el = Object.fromEntries([
  'video', 'output', 'modelInput', 'maskCanvas', 'dropZone', 'dropPrompt', 'emptyState',
  'fileInput', 'fileButton', 'cameraButton', 'sourceButton', 'detector', 'effect', 'threshold',
  'thresholdValue', 'strength', 'strengthValue', 'color', 'status', 'statusDot',
  'metrics', 'backend', 'personMask', 'personMaskCanvas', 'frameCanvas', 'personLayer',
  'personThreshold', 'personThresholdValue', 'segmenter'
].map((id) => [id, document.getElementById(id)]));

const outputCtx = el.output.getContext('2d');
const inputCtx = el.modelInput.getContext('2d', { willReadFrequently: true });
const maskCtx = el.maskCanvas.getContext('2d');
const personMaskCtx = el.personMaskCanvas.getContext('2d', { willReadFrequently: true });
const frameCtx = el.frameCanvas.getContext('2d');
const personLayerCtx = el.personLayer.getContext('2d');
let session;
let selfieSegmenter;
let multiclassSegmenter;
let multiclassBackgroundIndex = 0;
let multiclassGpuCanvas;
let rvmSession;
let rvmStates;
let rvmPrecision = 'float32';
let personMaskReady = false;
let latestPersonMask;
let currentDetector = 'teed';
let detectorReady = false;
const sessionCache = new Map();
const rvmSessionCache = new Map();
let stream;
let objectUrl;
let processing = false;
let active = false;
let lastRun = 0;
let lastFrameTime = performance.now();
let smoothedMs = 0;
let smoothedFps = 0;
let lastVideoTime = 0;
const float32Scratch = new Float32Array(1);
const uint32Scratch = new Uint32Array(float32Scratch.buffer);

function getPersonGate(confidence) {
  // Preserve the previous 0.15–0.50 feather at the default center of 0.33.
  const center = Number(el.personThreshold.value);
  const feather = 0.175;
  return Math.max(0, Math.min(1, (confidence - (center - feather)) / (feather * 2)));
}

function updateBackend(label = DETECTORS[currentDetector].label) {
  const runtime = DETECTORS[currentDetector].url
    ? ('gpu' in navigator ? 'WebGPU (WASM fallback)' : 'WASM')
    : 'JavaScript';
  const person = el.personMask.checked ? ` + ${SEGMENTERS[el.segmenter.value].backend} person mask` : '';
  el.backend.textContent = `${runtime} · ${label} ${SIZE} × ${SIZE}${person}`;
}

function applyPersonMask(mask) {
  const alphaMask = personMaskCtx.createImageData(SIZE, SIZE);
  for (let i = 0, p = 0; i < alphaMask.data.length; i += 4, p += 1) {
    const gate = getPersonGate(mask[p]);
    alphaMask.data[i] = 255;
    alphaMask.data[i + 1] = 255;
    alphaMask.data[i + 2] = 255;
    alphaMask.data[i + 3] = Math.round(gate * 255);
  }
  personMaskCtx.putImageData(alphaMask, 0, 0);
  latestPersonMask = mask;
}

function resizePersonMask(source, sourceWidth, sourceHeight) {
  if (sourceWidth === SIZE && sourceHeight === SIZE) return Float32Array.from(source);
  const output = new Float32Array(SIZE * SIZE);
  const xScale = sourceWidth / SIZE;
  const yScale = sourceHeight / SIZE;
  for (let y = 0; y < SIZE; y += 1) {
    const sourceY = Math.max(0, Math.min(sourceHeight - 1, (y + 0.5) * yScale - 0.5));
    const y0 = Math.floor(sourceY);
    const y1 = Math.min(sourceHeight - 1, y0 + 1);
    const yWeight = sourceY - y0;
    for (let x = 0; x < SIZE; x += 1) {
      const sourceX = Math.max(0, Math.min(sourceWidth - 1, (x + 0.5) * xScale - 0.5));
      const x0 = Math.floor(sourceX);
      const x1 = Math.min(sourceWidth - 1, x0 + 1);
      const xWeight = sourceX - x0;
      const top = source[y0 * sourceWidth + x0] * (1 - xWeight)
        + source[y0 * sourceWidth + x1] * xWeight;
      const bottom = source[y1 * sourceWidth + x0] * (1 - xWeight)
        + source[y1 * sourceWidth + x1] * xWeight;
      output[y * SIZE + x] = top * (1 - yWeight) + bottom * yWeight;
    }
  }
  return output;
}

function runMulticlassSegmentation() {
  multiclassSegmenter.segmentForVideo(el.modelInput, performance.now(), (result) => {
    const masks = result.confidenceMasks;
    if (!masks?.length || !masks[multiclassBackgroundIndex]) {
      throw new Error('MediaPipe multiclass returned no background confidence mask');
    }
    const backgroundMask = masks[multiclassBackgroundIndex];
    const background = backgroundMask.getAsFloat32Array();
    const person = new Float32Array(background.length);
    for (let i = 0; i < background.length; i += 1) person[i] = 1 - background[i];
    applyPersonMask(resizePersonMask(person, backgroundMask.width, backgroundMask.height));
  });
}

function resetRvmState() {
  rvmStates?.forEach((state) => state.dispose?.());
  const data = rvmPrecision === 'float16' ? new Uint16Array(1) : new Float32Array(1);
  rvmStates = Array.from({ length: 4 }, () => (
    new ort.Tensor(rvmPrecision, data.slice(), [1, 1, 1, 1])
  ));
}

function float32ToFloat16(value) {
  float32Scratch[0] = value;
  const bits = uint32Scratch[0];
  const sign = (bits >>> 16) & 0x8000;
  const exponent = ((bits >>> 23) & 0xff) - 127 + 15;
  const mantissa = bits & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    return sign | ((mantissa | 0x800000) >>> (14 - exponent));
  }
  if (exponent >= 31) return sign | 0x7c00;
  return sign | (exponent << 10) | (mantissa >>> 13);
}

function float16ToFloat32(value) {
  const sign = value & 0x8000 ? -1 : 1;
  const exponent = (value >>> 10) & 0x1f;
  const mantissa = value & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 31) return mantissa ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

async function loadPersonMask() {
  personMaskReady = false;
  if (el.segmenter.value === 'mediapipe-multiclass') {
    if (!multiclassSegmenter) {
      const { FilesetResolver, ImageSegmenter } = await import('@mediapipe/tasks-vision');
      // Tasks Vision injects its WASM loader as a classic script. The module
      // loader uses import.meta and therefore cannot run in that context.
      const vision = await FilesetResolver.forVisionTasks(assetUrl('mediapipe/tasks-vision'));
      const options = {
        baseOptions: {
          modelAssetPath: assetUrl('mediapipe/selfie_multiclass_256x256.tflite')
        },
        runningMode: 'VIDEO',
        outputConfidenceMasks: true,
        outputCategoryMask: false
      };
      try {
        multiclassGpuCanvas = document.createElement('canvas');
        multiclassSegmenter = await ImageSegmenter.createFromOptions(vision, {
          ...options,
          baseOptions: { ...options.baseOptions, delegate: 'GPU' },
          canvas: multiclassGpuCanvas
        });
        SEGMENTERS['mediapipe-multiclass'].backend = 'MediaPipe multiclass GPU';
      } catch (error) {
        console.warn('MediaPipe multiclass GPU initialization failed; using CPU.', error);
        multiclassGpuCanvas = undefined;
        multiclassSegmenter = await ImageSegmenter.createFromOptions(vision, {
          ...options,
          baseOptions: { ...options.baseOptions, delegate: 'CPU' }
        });
        SEGMENTERS['mediapipe-multiclass'].backend = 'MediaPipe multiclass CPU';
      }
      const labels = multiclassSegmenter.getLabels();
      const backgroundIndex = labels.findIndex((label) => label.toLowerCase() === 'background');
      multiclassBackgroundIndex = backgroundIndex >= 0 ? backgroundIndex : 0;
      console.info('MediaPipe multiclass labels:', labels);
    }
  } else if (el.segmenter.value === 'mediapipe') {
    if (!selfieSegmenter) {
      const { SelfieSegmentation } = await import('@mediapipe/selfie_segmentation');
      selfieSegmenter = new SelfieSegmentation({ locateFile: (file) => assetUrl(`mediapipe/${file}`) });
      selfieSegmenter.setOptions({ modelSelection: 1, selfieMode: false });
      selfieSegmenter.onResults((results) => {
        personMaskCtx.clearRect(0, 0, SIZE, SIZE);
        personMaskCtx.drawImage(results.segmentationMask, 0, 0, SIZE, SIZE);
        const rgba = personMaskCtx.getImageData(0, 0, SIZE, SIZE).data;
        const mask = new Float32Array(SIZE * SIZE);
        for (let i = 0, p = 0; i < rgba.length; i += 4, p += 1) mask[p] = rgba[i] / 255;
        applyPersonMask(mask);
      });
      await selfieSegmenter.initialize();
    }
  } else {
    const config = SEGMENTERS[el.segmenter.value];
    rvmPrecision = config.precision;
    if (!rvmSessionCache.has(el.segmenter.value)) {
      rvmSessionCache.set(el.segmenter.value, await ort.InferenceSession.create(assetUrl(config.url), {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      }));
    }
    rvmSession = rvmSessionCache.get(el.segmenter.value);
    resetRvmState();
  }
  personMaskReady = true;
}

async function runRvm(rgba) {
  const plane = SIZE * SIZE;
  const rgb = rvmPrecision === 'float16'
    ? new Uint16Array(plane * 3)
    : new Float32Array(plane * 3);
  for (let i = 0, p = 0; i < rgba.length; i += 4, p += 1) {
    const r = rgba[i] / 255;
    const g = rgba[i + 1] / 255;
    const b = rgba[i + 2] / 255;
    rgb[p] = rvmPrecision === 'float16' ? float32ToFloat16(r) : r;
    rgb[plane + p] = rvmPrecision === 'float16' ? float32ToFloat16(g) : g;
    rgb[plane * 2 + p] = rvmPrecision === 'float16' ? float32ToFloat16(b) : b;
  }
  const src = new ort.Tensor(rvmPrecision, rgb, [1, 3, SIZE, SIZE]);
  const ratio = new ort.Tensor('float32', new Float32Array([1]), [1]);
  const outputs = await rvmSession.run({
    src,
    r1i: rvmStates[0],
    r2i: rvmStates[1],
    r3i: rvmStates[2],
    r4i: rvmStates[3],
    downsample_ratio: ratio
  });
  const previousStates = rvmStates;
  rvmStates = [outputs.r1o, outputs.r2o, outputs.r3o, outputs.r4o];
  previousStates.forEach((state) => state.dispose?.());
  const rawAlpha = outputs.pha.data;
  const alpha = rvmPrecision === 'float16' && rawAlpha instanceof Uint16Array
    ? Float32Array.from(rawAlpha, float16ToFloat32)
    : Float32Array.from(rawAlpha);
  applyPersonMask(alpha);
  outputs.fgr.dispose?.();
  outputs.pha.dispose?.();
  src.dispose?.();
  ratio.dispose?.();
}

function setStatus(message, type = '') {
  el.status.textContent = message;
  el.status.parentElement.className = `status ${type}`;
}

function setControlsEnabled(enabled) {
  el.fileButton.disabled = !enabled;
  el.cameraButton.disabled = !enabled;
}

async function loadDetector(detectorId) {
  const detector = DETECTORS[detectorId];
  detectorReady = false;
  setControlsEnabled(false);
  el.detector.disabled = true;
  setStatus(`Loading ${detector.label}…`);
  // Vite fingerprints assets, so give ORT the resolved URL instead of letting it
  // guess a filename next to the JavaScript bundle.
  ort.env.wasm.wasmPaths = { wasm: wasmUrl };
  ort.env.wasm.numThreads = crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 1, 4) : 1;
  const supportsWebGpu = 'gpu' in navigator;
  const providers = supportsWebGpu ? ['webgpu', 'wasm'] : ['wasm'];
  try {
    if (detector.url) {
      if (!sessionCache.has(detectorId)) {
        sessionCache.set(detectorId, await ort.InferenceSession.create(assetUrl(detector.url), {
          executionProviders: providers,
          graphOptimizationLevel: 'all'
        }));
      }
      session = sessionCache.get(detectorId);
    } else {
      session = undefined;
    }
    currentDetector = detectorId;
    updateBackend(detector.label);
    detectorReady = true;
    el.threshold.value = detector.threshold;
    el.thresholdValue.value = detector.threshold.toFixed(2);
    smoothedMs = 0;
    smoothedFps = 0;
    setStatus(active ? 'Processing live' : 'Detector ready', 'ready');
    setControlsEnabled(true);
  } catch (error) {
    console.error(error);
    setStatus(`${detector.label} failed to load`, 'error');
    el.backend.textContent = 'See browser console for details';
  } finally {
    el.detector.disabled = false;
  }
}

function stopCurrentSource() {
  active = false;
  lastVideoTime = 0;
  if (rvmSession) resetRvmState();
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
    stream = undefined;
  }
  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
    objectUrl = undefined;
  }
  el.video.pause();
  el.video.srcObject = null;
  el.video.removeAttribute('src');
  el.dropZone.classList.remove('playing');
  el.emptyState.hidden = false;
}

async function startVideo(file) {
  stopCurrentSource();
  objectUrl = URL.createObjectURL(file);
  el.video.src = objectUrl;
  el.video.loop = true;
  await el.video.play();
  beginProcessing();
}

async function startCamera() {
  if (!detectorReady) return;
  stopCurrentSource();
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false
    });
    el.video.srcObject = stream;
    await el.video.play();
    beginProcessing();
  } catch (error) {
    console.error(error);
    setStatus(error.name === 'NotAllowedError' ? 'Camera permission denied' : 'Camera unavailable', 'error');
  }
}

function beginProcessing() {
  const width = el.video.videoWidth || 1280;
  const height = el.video.videoHeight || 720;
  el.output.width = width;
  el.output.height = height;
  el.frameCanvas.width = width;
  el.frameCanvas.height = height;
  el.personLayer.width = width;
  el.personLayer.height = height;
  el.emptyState.hidden = true;
  el.dropZone.classList.add('playing');
  fitOutputToStage();
  active = true;
  setStatus('Processing live', 'ready');
  scheduleFrame();
}

function fitOutputToStage() {
  if (!el.output.width || !el.output.height) return;
  const availableWidth = el.dropZone.clientWidth;
  const availableHeight = el.dropZone.clientHeight;
  const scale = Math.min(
    availableWidth / el.output.width,
    availableHeight / el.output.height
  );
  el.output.style.width = `${Math.floor(el.output.width * scale)}px`;
  el.output.style.height = `${Math.floor(el.output.height * scale)}px`;
}

function scheduleFrame() {
  if (!active) return;
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    el.video.requestVideoFrameCallback(processFrame);
  } else {
    requestAnimationFrame(processFrame);
  }
}

function readFrame() {
  // Freeze one full-resolution frame. Both models and the final composite use
  // this snapshot, so an advancing video cannot drift underneath the masks.
  frameCtx.drawImage(el.video, 0, 0, el.frameCanvas.width, el.frameCanvas.height);
  inputCtx.drawImage(el.frameCanvas, 0, 0, SIZE, SIZE);
  return inputCtx.getImageData(0, 0, SIZE, SIZE).data;
}

function toTensor(data, preprocessing) {
  const plane = SIZE * SIZE;
  const input = new Float32Array(plane * 3);
  for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
    if (preprocessing === 'teed') {
      input[p] = data[i + 2] - BGR_MEAN[0];
      input[plane + p] = data[i + 1] - BGR_MEAN[1];
      input[plane * 2 + p] = data[i] - BGR_MEAN[2];
    } else {
      input[p] = (data[i] / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
      input[plane + p] = (data[i + 1] / 255 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
      input[plane * 2 + p] = (data[i + 2] / 255 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }
  }
  return new ort.Tensor('float32', input, [1, 3, SIZE, SIZE]);
}

function detectCanny(rgba) {
  const count = SIZE * SIZE;
  const gray = new Float32Array(count);
  const horizontal = new Float32Array(count);
  const blur = new Float32Array(count);
  const magnitude = new Float32Array(count);
  const direction = new Uint8Array(count);
  const suppressed = new Float32Array(count);

  for (let i = 0, p = 0; i < rgba.length; i += 4, p += 1) {
    gray[p] = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
  }
  for (let y = 0; y < SIZE; y += 1) {
    const row = y * SIZE;
    for (let x = 1; x < SIZE - 1; x += 1) {
      const i = row + x;
      horizontal[i] = (gray[i - 1] + 2 * gray[i] + gray[i + 1]) * 0.25;
    }
  }
  for (let y = 1; y < SIZE - 1; y += 1) {
    for (let x = 1; x < SIZE - 1; x += 1) {
      const i = y * SIZE + x;
      blur[i] = (horizontal[i - SIZE] + 2 * horizontal[i] + horizontal[i + SIZE]) * 0.25;
    }
  }

  let maxMagnitude = 0;
  for (let y = 1; y < SIZE - 1; y += 1) {
    for (let x = 1; x < SIZE - 1; x += 1) {
      const i = y * SIZE + x;
      const gx = -blur[i - SIZE - 1] + blur[i - SIZE + 1]
        - 2 * blur[i - 1] + 2 * blur[i + 1]
        - blur[i + SIZE - 1] + blur[i + SIZE + 1];
      const gy = -blur[i - SIZE - 1] - 2 * blur[i - SIZE] - blur[i - SIZE + 1]
        + blur[i + SIZE - 1] + 2 * blur[i + SIZE] + blur[i + SIZE + 1];
      const mag = Math.hypot(gx, gy);
      magnitude[i] = mag;
      maxMagnitude = Math.max(maxMagnitude, mag);
      let angle = Math.atan2(gy, gx) * 180 / Math.PI;
      if (angle < 0) angle += 180;
      direction[i] = angle < 22.5 || angle >= 157.5 ? 0 : angle < 67.5 ? 1 : angle < 112.5 ? 2 : 3;
    }
  }

  for (let y = 1; y < SIZE - 1; y += 1) {
    for (let x = 1; x < SIZE - 1; x += 1) {
      const i = y * SIZE + x;
      const d = direction[i];
      const a = d === 0 ? i - 1 : d === 1 ? i - SIZE + 1 : d === 2 ? i - SIZE : i - SIZE - 1;
      const b = d === 0 ? i + 1 : d === 1 ? i + SIZE - 1 : d === 2 ? i + SIZE : i + SIZE + 1;
      if (magnitude[i] >= magnitude[a] && magnitude[i] >= magnitude[b]) suppressed[i] = magnitude[i];
    }
  }

  const low = maxMagnitude * 0.08;
  const high = maxMagnitude * 0.2;
  const state = new Uint8Array(count);
  const stack = [];
  for (let i = 0; i < count; i += 1) {
    if (suppressed[i] >= high) {
      state[i] = 2;
      stack.push(i);
    } else if (suppressed[i] >= low) {
      state[i] = 1;
    }
  }
  while (stack.length) {
    const i = stack.pop();
    const y = Math.floor(i / SIZE);
    const x = i - y * SIZE;
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        if (!dx && !dy) continue;
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 1 || nx >= SIZE - 1 || ny < 1 || ny >= SIZE - 1) continue;
        const neighbor = ny * SIZE + nx;
        if (state[neighbor] === 1) {
          state[neighbor] = 2;
          stack.push(neighbor);
        }
      }
    }
  }
  const edges = new Float32Array(count);
  const scale = 1 / Math.max(maxMagnitude, 1e-6);
  for (let i = 0; i < count; i += 1) {
    if (state[i] === 2) edges[i] = suppressed[i] * scale;
  }
  return edges;
}

function hexToRgb(hex) {
  const n = Number.parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function render(mask, normalizeMask, personMask = null) {
  const threshold = Number(el.threshold.value);
  const strength = Number(el.strength.value);
  const color = hexToRgb(el.color.value);
  const effect = el.effect.value;
  const isLineMask = effect === 'mask' || effect === 'mask-filled';
  const lineMaskColor = effect === 'mask-filled' ? [3, 5, 7] : [255, 255, 255];
  const pixels = maskCtx.createImageData(SIZE, SIZE);

  // TEED's official inference code min/max normalizes every fused edge map.
  // Keep that behavior so the UI threshold remains useful across scenes.
  let maskMin = 0;
  let maskMax = 1;
  if (normalizeMask) {
    maskMin = Infinity;
    maskMax = -Infinity;
    for (let i = 0; i < mask.length; i += 1) {
      maskMin = Math.min(maskMin, mask[i]);
      maskMax = Math.max(maskMax, mask[i]);
    }
  }
  const maskRange = Math.max(maskMax - maskMin, 1e-6);

  for (let i = 0, p = 0; i < mask.length; i += 1, p += 4) {
    const normalized = (mask[i] - maskMin) / maskRange;
    const confidence = Math.max(0, Math.min(1, (normalized - threshold) / Math.max(0.001, 1 - threshold)));
    // Keep boundary pixels, where selfie segmentation is naturally soft, while
    // suppressing edges confidently classified as background.
    const personGate = personMask ? getPersonGate(personMask[i]) : 1;
    const alpha = Math.round(255 * confidence * strength * personGate);
    if (isLineMask) {
      pixels.data[p] = lineMaskColor[0];
      pixels.data[p + 1] = lineMaskColor[1];
      pixels.data[p + 2] = lineMaskColor[2];
      pixels.data[p + 3] = Math.max(alpha, 10);
    } else if (effect === 'ink') {
      pixels.data[p] = 3;
      pixels.data[p + 1] = 5;
      pixels.data[p + 2] = 7;
      pixels.data[p + 3] = alpha;
    } else {
      pixels.data[p] = color[0];
      pixels.data[p + 1] = color[1];
      pixels.data[p + 2] = color[2];
      pixels.data[p + 3] = alpha;
    }
  }

  maskCtx.putImageData(pixels, 0, 0);
  outputCtx.clearRect(0, 0, el.output.width, el.output.height);
  if (isLineMask) {
    outputCtx.fillStyle = '#050607';
    outputCtx.fillRect(0, 0, el.output.width, el.output.height);
    if (effect === 'mask-filled' && personMask) {
      outputCtx.drawImage(
        el.personMaskCanvas,
        0,
        0,
        el.output.width,
        el.output.height
      );
    }
  } else if (effect === 'person-debug') {
    outputCtx.fillStyle = '#00b140';
    outputCtx.fillRect(0, 0, el.output.width, el.output.height);
    if (personMask) {
      personLayerCtx.clearRect(0, 0, el.personLayer.width, el.personLayer.height);
      personLayerCtx.drawImage(el.frameCanvas, 0, 0);
      personLayerCtx.globalCompositeOperation = 'destination-in';
      personLayerCtx.drawImage(el.personMaskCanvas, 0, 0, el.personLayer.width, el.personLayer.height);
      personLayerCtx.globalCompositeOperation = 'source-over';
      outputCtx.drawImage(el.personLayer, 0, 0);
    }
  } else {
    outputCtx.drawImage(el.frameCanvas, 0, 0, el.output.width, el.output.height);
  }

  outputCtx.save();
  if (effect === 'neon') {
    outputCtx.globalCompositeOperation = 'screen';
    outputCtx.filter = 'blur(7px)';
    outputCtx.globalAlpha = 0.65;
    outputCtx.drawImage(el.maskCanvas, 0, 0, el.output.width, el.output.height);
    outputCtx.filter = 'none';
    outputCtx.globalAlpha = 1;
  }
  outputCtx.drawImage(el.maskCanvas, 0, 0, el.output.width, el.output.height);
  outputCtx.restore();
}

async function processFrame(now) {
  scheduleFrame();
  if (!active || !detectorReady || processing || el.video.readyState < 2) return;
  const targetInterval = 1000 / 30;
  if (now - lastRun < targetInterval) return;
  lastRun = now;
  processing = true;
  const started = performance.now();
  let tensor;
  let result;
  try {
    const detector = DETECTORS[currentDetector];
    const pixels = readFrame();
    const usePersonMask = el.personMask.checked && personMaskReady;
    if (usePersonMask) {
      latestPersonMask = undefined;
      if (el.segmenter.value === 'mediapipe-multiclass') {
        runMulticlassSegmentation();
      } else if (el.segmenter.value === 'mediapipe') {
        await selfieSegmenter.send({ image: el.modelInput });
      } else {
        // RVM carries information between adjacent frames. A loop or backward
        // seek is a scene discontinuity, so do not feed it state from the end.
        if (el.video.currentTime + 0.05 < lastVideoTime) resetRvmState();
        await runRvm(pixels);
      }
    }
    lastVideoTime = el.video.currentTime;
    const framePersonMask = usePersonMask ? latestPersonMask : null;
    if (detector.preprocessing === 'canny') {
      render(detectCanny(pixels), detector.normalizeMask, framePersonMask);
    } else {
      const frameSession = session;
      tensor = toTensor(pixels, detector.preprocessing);
      const feeds = { [frameSession.inputNames[0]]: tensor };
      const outputs = await frameSession.run(feeds);
      result = outputs[frameSession.outputNames[0]];
      if (DETECTORS[currentDetector] !== detector) return;
      render(result.data, detector.normalizeMask, framePersonMask);
    }
    const elapsed = performance.now() - started;
    const frameDelta = performance.now() - lastFrameTime;
    lastFrameTime = performance.now();
    smoothedMs = smoothedMs ? smoothedMs * .85 + elapsed * .15 : elapsed;
    const fps = 1000 / Math.max(frameDelta, 1);
    smoothedFps = smoothedFps ? smoothedFps * .85 + fps * .15 : fps;
    el.metrics.textContent = `${smoothedMs.toFixed(0)} ms · ${smoothedFps.toFixed(1)} fps`;
  } catch (error) {
    console.error(error);
    active = false;
    setStatus('Inference stopped', 'error');
  } finally {
    tensor?.dispose?.();
    result?.dispose?.();
    processing = false;
  }
}

function acceptFiles(files) {
  const file = [...files].find((candidate) => candidate.type.startsWith('video/'));
  if (file && detectorReady) startVideo(file).catch(console.error);
}

el.fileButton.addEventListener('click', () => el.fileInput.click());
el.fileInput.addEventListener('change', () => acceptFiles(el.fileInput.files));
el.cameraButton.addEventListener('click', startCamera);
el.detector.addEventListener('change', () => loadDetector(el.detector.value));
el.segmenter.addEventListener('change', async () => {
  latestPersonMask = undefined;
  personMaskReady = false;
  if (!el.personMask.checked) {
    updateBackend();
    return;
  }
  el.segmenter.disabled = true;
  setStatus(`Loading ${SEGMENTERS[el.segmenter.value].label}…`);
  try {
    await loadPersonMask();
    updateBackend();
    setStatus(active ? 'Processing live' : 'Detector ready', 'ready');
  } catch (error) {
    console.error(error);
    setStatus('Person model failed to load', 'error');
  } finally {
    el.segmenter.disabled = false;
  }
});
el.effect.addEventListener('change', () => {
  const needsPersonMask = el.effect.value === 'person-debug' || el.effect.value === 'mask-filled';
  if (needsPersonMask && !el.personMask.checked) {
    el.personMask.checked = true;
    el.personMask.dispatchEvent(new Event('change'));
  }
});
el.personMask.addEventListener('change', async () => {
  if (!el.personMask.checked) {
    latestPersonMask = undefined;
    el.personThreshold.disabled = true;
    updateBackend();
    return;
  }
  el.personThreshold.disabled = false;
  el.personMask.disabled = true;
  setStatus(`Loading ${SEGMENTERS[el.segmenter.value].label}…`);
  try {
    await loadPersonMask();
    updateBackend();
    setStatus(active ? 'Processing live' : 'Detector ready', 'ready');
  } catch (error) {
    console.error(error);
    personMaskReady = false;
    selfieSegmenter = undefined;
    multiclassSegmenter?.close();
    multiclassSegmenter = undefined;
    multiclassGpuCanvas = undefined;
    el.personMask.checked = false;
    el.personThreshold.disabled = true;
    updateBackend();
    setStatus('Person mask failed to load', 'error');
  } finally {
    el.personMask.disabled = false;
  }
});
el.sourceButton.addEventListener('click', () => {
  stopCurrentSource();
  el.fileInput.value = '';
  setStatus('Detector ready', 'ready');
});

for (const eventName of ['dragenter', 'dragover']) {
  el.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    el.dropZone.classList.add('dragging');
  });
}
for (const eventName of ['dragleave', 'drop']) {
  el.dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    el.dropZone.classList.remove('dragging');
  });
}
el.dropZone.addEventListener('drop', (event) => acceptFiles(event.dataTransfer.files));

for (const id of ['threshold', 'strength', 'personThreshold']) {
  el[id].addEventListener('input', () => { el[`${id}Value`].value = el[id].value; });
}

window.addEventListener('beforeunload', stopCurrentSource);
window.addEventListener('resize', fitOutputToStage);
loadDetector(el.detector.value);
