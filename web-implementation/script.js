import { ImageSegmenter, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const status = document.getElementById("status");
const labelsDiv = document.getElementById("labels");

let imageSegmenter;
let webcamRunning = false;

// Couleurs pour chaque classe
const legendColors = [
    [0, 0, 0, 0],         // 0: Background (Transparent)
    [255, 0, 0, 150],     // 1: Hair (Red)
    [0, 255, 0, 150],     // 2: Body skin (Green)
    [0, 0, 255, 150],     // 3: Face skin (Blue)
    [255, 255, 0, 150],   // 4: Clothes (Yellow)
    [255, 0, 255, 150]    // 5: Others (Purple)
];

const labels = ["Fond", "Cheveux", "Peau Corps", "Peau Visage", "Vêtements", "Accessoires"];

async function initialize() {
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        
        imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            outputCategoryMask: true,
            outputConfidenceMasks: false
        });

        status.innerText = "Modèle chargé. Démarrage de la caméra...";
        startWebcam();
        displayLabels();
    } catch (error) {
        status.innerText = "Erreur: " + error.message;
        console.error(error);
    }
}

function displayLabels() {
    labelsDiv.innerHTML = labels.map((label, i) => {
        const color = legendColors[i];
        const rgba = `rgba(${color[0]}, ${color[1]}, ${color[2]}, 1)`;
        return `<span class="label-pill" style="border-left: 10px solid ${rgba}">${label}</span>`;
    }).join("");
}

async function startWebcam() {
    const constraints = { video: true };
    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
        webcamRunning = true;
        status.style.display = "none";
    } catch (err) {
        status.innerText = "Erreur caméra: " + err.message;
    }
}

let lastVideoTime = -1;
async function predictWebcam() {
    if (video.currentTime === lastVideoTime) {
        if (webcamRunning) {
            window.requestAnimationFrame(predictWebcam);
        }
        return;
    }
    lastVideoTime = video.currentTime;

    if (imageSegmenter) {
        const startTimeMs = performance.now();
        const segmentationResult = imageSegmenter.segmentForVideo(video, startTimeMs);
        
        drawSegmentation(segmentationResult);
    }

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
}

function drawSegmentation(result) {
    const mask = result.categoryMask.getAsUint8Array();
    const width = result.categoryMask.width;
    const height = result.categoryMask.height;

    // Redimensionner le canvas si nécessaire
    if (canvasElement.width !== video.videoWidth || canvasElement.height !== video.videoHeight) {
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
    }

    // Créer une image de données pour le masque
    const imageData = canvasCtx.createImageData(width, height);
    const data = imageData.data;

    for (let i = 0; i < mask.length; i++) {
        const category = mask[i];
        const color = legendColors[category] || [0, 0, 0, 0];
        
        const rIdx = i * 4;
        data[rIdx] = color[0];
        data[rIdx + 1] = color[1];
        data[rIdx + 2] = color[2];
        data[rIdx + 3] = color[3];
    }

    // Dessiner le masque sur un canvas temporaire pour le redimensionner
    const offscreenCanvas = new OffscreenCanvas(width, height);
    const offscreenCtx = offscreenCanvas.getContext("2d");
    offscreenCtx.putImageData(imageData, 0, 0);

    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(offscreenCanvas, 0, 0, canvasElement.width, canvasElement.height);
}

initialize();
