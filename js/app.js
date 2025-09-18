// Importamos las herramientas necesarias de MediaPipe desde su CDN
import {
  FaceLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

// Referencias a elementos del DOM
const video = document.getElementById('video');
const canvasOutput = document.getElementById('canvasOutput');
const canvasCtx = canvasOutput.getContext('2d');
const status = document.getElementById('status');
const eyeCounterEl = document.getElementById('eyeCounter');
const mouthCounterEl = document.getElementById('mouthCounter');
const eyebrowCounterEl = document.getElementById('eyebrowCounter');

// --- PARÁMETROS DE CALIBRACIÓN ---
const EYE_AR_THRESH = 0.23; 
const EYE_AR_CONSEC_FRAMES = 2; 
const MOUTH_AR_THRESH = 0.5;

// CAMBIO ÚNICO: Umbral de ceja más sensible, pero aún robusto.
const EYEBROW_RAISE_THRESH = 0.09; // Antes era 0.12

const EYEBROW_RAISE_CONSEC_FRAMES = 1;

// --- VARIABLES DE ESTADO ---
let eyeBlinks = 0;
let mouthOpenings = 0;
let eyebrowRaises = 0;
let blinkCounter = 0;
let isMouthOpen = false;
let isEyebrowRaised = false;
let eyebrowRaiseCounter = 0;
let neutralEyebrowRatio = null; 

let faceLandmarker;
let drawingUtils;
let lastVideoTime = -1;

// --- FUNCIONES DE CÁLCULO GEOMÉTRICO ---
const euclideanDist = (p1, p2) => Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));

function getEAR(eyeLandmarks) {
    const p1 = eyeLandmarks[0], p2 = eyeLandmarks[1], p3 = eyeLandmarks[2];
    const p4 = eyeLandmarks[3], p5 = eyeLandmarks[4], p6 = eyeLandmarks[5];
    const dV1 = euclideanDist(p2, p6);
    const dV2 = euclideanDist(p3, p5);
    const dH = euclideanDist(p1, p4);
    return (dV1 + dV2) / (2.0 * dH);
}

function getMAR(mouthLandmarks) {
    const p1 = mouthLandmarks[0], p2 = mouthLandmarks[1], p3 = mouthLandmarks[2], p4 = mouthLandmarks[3];
    const dV = euclideanDist(p3, p4); 
    const dH = euclideanDist(p1, p2); 
    return dV / dH;
}

// --- LÓGICA PRINCIPAL DE LA APLICACIÓN ---
async function main() {
    const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: "GPU",
        },
        runningMode: "VIDEO",
        numFaces: 1,
    });
    drawingUtils = new DrawingUtils(canvasCtx);
    status.textContent = 'SYSTEM READY.';
    startCamera();
}

function startCamera() {
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false })
        .then(stream => {
            video.srcObject = stream;
            video.addEventListener('loadeddata', predictWebcam);
        })
        .catch(err => {
            console.error("Error al acceder a la cámara: ", err);
            status.textContent = 'CAMERA ACCESS DENIED.';
        });
}

async function predictWebcam() {
    canvasOutput.width = video.videoWidth;
    canvasOutput.height = video.videoHeight;
    
    if (video.readyState < 2) {
        window.requestAnimationFrame(predictWebcam);
        return;
    }

    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        const results = faceLandmarker.detectForVideo(video, performance.now());
        
        if (results.faceLandmarks && results.faceLandmarks.length > 0) {
            if (status.textContent !== "TRACKING...") status.textContent = "TRACKING...";
            const landmarks = results.faceLandmarks[0];
            
            const rightEyeIndices = [33, 160, 158, 133, 153, 144];
            const leftEyeIndices = [362, 385, 387, 263, 373, 380];
            const mouthIndices = [61, 291, 13, 14];
            const leftEyebrowIndex = 105;
            const noseBridgeIndex = 6; 
            const noseBottomIndex = 2;
            const chinIndex = 152;

            const rightEye = rightEyeIndices.map(i => landmarks[i]);
            const leftEye = leftEyeIndices.map(i => landmarks[i]);
            const mouth = mouthIndices.map(i => landmarks[i]);
            
            const ear = (getEAR(leftEye) + getEAR(rightEye)) / 2.0;
            const mar = getMAR(mouth);

            if (ear < EYE_AR_THRESH) {
                blinkCounter++;
            } else {
                if (blinkCounter >= EYE_AR_CONSEC_FRAMES) {
                    eyeBlinks++;
                    eyeCounterEl.textContent = eyeBlinks;
                }
                blinkCounter = 0;
            }

            if (mar > MOUTH_AR_THRESH) {
                if (!isMouthOpen) {
                    mouthOpenings++;
                    mouthCounterEl.textContent = mouthOpenings;
                    isMouthOpen = true;
                }
            } else {
                isMouthOpen = false;
            }

            const eyebrowPoint = landmarks[leftEyebrowIndex];
            const noseBridgePoint = landmarks[noseBridgeIndex];
            const nosePoint = landmarks[noseBottomIndex];
            const chinPoint = landmarks[chinIndex];

            const faceHeight = euclideanDist(nosePoint, chinPoint);
            const eyebrowDist = euclideanDist(eyebrowPoint, noseBridgePoint);
            const currentRatio = eyebrowDist / faceHeight;

            if (neutralEyebrowRatio === null) {
                neutralEyebrowRatio = currentRatio;
            } else {
                neutralEyebrowRatio = (neutralEyebrowRatio * 0.95) + (currentRatio * 0.05);
            }
            
            if (currentRatio > neutralEyebrowRatio * (1 + EYEBROW_RAISE_THRESH)) {
                eyebrowRaiseCounter++;
                if (eyebrowRaiseCounter >= EYEBROW_RAISE_CONSEC_FRAMES && !isEyebrowRaised) {
                    eyebrowRaises++;
                    eyebrowCounterEl.textContent = eyebrowRaises;
                    isEyebrowRaised = true;
                }
            } else {
                eyebrowRaiseCounter = 0;
                isEyebrowRaised = false;
            }

            canvasCtx.clearRect(0, 0, canvasOutput.width, canvasOutput.height);
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: '#FF4D4D' });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: '#FF4D4D' });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: '#FFFFFF' });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: '#FFFFFF' });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: '#FFFFFF' });
        }
    }
    
    window.requestAnimationFrame(predictWebcam);
}

main();