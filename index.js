// VERSION MODIFICANDOSE


import { env, AutoProcessor, AutoModel, RawImage } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';
const model_id = 'onnx-community/yolov10s';
const processor = await AutoProcessor.from_pretrained(model_id);
const model = await AutoModel.from_pretrained(model_id);



const THRESHOLD = 0.25;
const PEOPLE = 0;
let lastDetectionTime = 0;


const WHISPER_SAMPLING_RATE = 16000;
const MAX_AUDIO_LENGTH = 3.5; // seconds

const MAX_SAMPLES = WHISPER_SAMPLING_RATE * MAX_AUDIO_LENGTH;

let language = 'en';
const timeDataQueue = [];

let audioContext;
let stream;

const contentSummary = document.querySelector('.resum');
const transcriptionContainer = document.querySelector('.trans');
const videoElement = document.querySelector('.video video');


(async function app() {
    

// Read image and run processor
    if (navigator.mediaDevices.getUserMedia) {
        console.log('Initializing...');
        
    }
}());





//Function to start recording in real time our own voice (not from a video)
async function startRecordingRT() {
    try {
        console.log('Starting recording...');
        stream = await getAudioStream();
        audioContext = new AudioContext({
            latencyHint: "playback",
            sampleRate: WHISPER_SAMPLING_RATE
        });
        const streamSource = audioContext.createMediaStreamSource(stream);
        
        await audioContext.audioWorklet.addModule("recorder.worklet.js");
        const recorder = new AudioWorkletNode(audioContext, "recorder.worklet");
        streamSource.connect(recorder).connect(audioContext.destination);

        recorder.port.onmessage = async (e) => {
            const inputBuffer = Array.from(e.data);
            if (inputBuffer[0] === 0) return;

            timeDataQueue.push(...inputBuffer);

            if (timeDataQueue.length >= MAX_SAMPLES) {
                const audioData = new Float32Array(timeDataQueue.splice(0, MAX_SAMPLES));
                worker.postMessage({ type: 'generate', data: { audio: audioData, language } });
            }
        };
    } catch (e) {
        console.error('Error starting recording:', e);
    }
}

//Function to start recording the audio of a video (actual implementation)
async function startRecordingVideo(videoElement) {
    try {
        console.log('Starting recording...');
        
        // Crear un contexto de audio
        audioContext = new AudioContext({
            latencyHint: "playback",
            sampleRate: WHISPER_SAMPLING_RATE
        });

        // Crear una fuente de audio desde el elemento de video
        const streamSource = audioContext.createMediaElementSource(videoElement);

        // Cargar y conectar el módulo del worklet
        await audioContext.audioWorklet.addModule("recorder.worklet.js");
        const recorder = new AudioWorkletNode(audioContext, "recorder.worklet");
        streamSource.connect(audioContext.destination);
        streamSource.connect(recorder);
        // Conectar el flujo de audio al recorder y al destino de audio
        //streamSource.connect(recorder).connect(audioContext.destination);

        // Procesar los datos del audio
        recorder.port.onmessage = async (e) => {
            const inputBuffer = Array.from(e.data);
            if (inputBuffer[0] === 0) return;

            timeDataQueue.push(...inputBuffer);

            if (timeDataQueue.length >= MAX_SAMPLES) {
                const audioData = new Float32Array(timeDataQueue.splice(0, MAX_SAMPLES));
                worker.postMessage({ type: 'generate', data: { audio: audioData, language } });
            }
        };

        console.log('Recording started.');
    } catch (e) {
        console.error('Error starting recording:', e);
    }
}


async function stopRecording() {
    console.log('Stopping recording...');
    if (!audioContext || !stream) return;
    audioContext.close();
    timeDataQueue.length = 0;
    if (stream.getTracks().length > 0) {
        stream.getTracks()[0].stop();
    }
}

//Function to get the audio stream
async function getAudioStream(audioTrackConstraints) {
    let options = audioTrackConstraints || {};
    try {
        return await navigator.mediaDevices.getUserMedia({
            video: false,
            audio: {
                sampleRate: options.sampleRate || WHISPER_SAMPLING_RATE,
                sampleSize: options.sampleSize || 16,
                channelCount: options.channelCount || 1
            }
        });
    } catch (e) {
        console.error(e);
        return null;
    }
}

const worker = new Worker('whisper.worker.js', { type: "module" });

// Different messages that the worker can receive and how to handle them
worker.onmessage = function (e) {
    switch (e.data.status) {
        case 'loading':
            console.log('Loading status:', e.data.data);
            break;

        case 'initiate':
            break;

        case 'progress':
            break;

        case 'done':
            break;

        case 'ready':
            // Start to get the audio of the video if the worker is ready
            console.log('Worker is ready for processing.');
            detectObjectsInVideo();
            startRecordingVideo(videoElement);
            
            break;

        case 'start':
            break;

        case 'update':
            console.log(e.data.output);
            break;

        case 'complete':
            // Start to transcribe the audio when the worker is done
            console.log(e.data.output);
            updateTranscription(e.data.output);
            break;
        default:
            console.error('Unknown status:', status);
    }
};

worker.onerror = function (error) {
    console.error('Worker error:', error.message);
};

worker.postMessage({ type: 'load' });

// Function to update the transcription in the HTML
function updateTranscription(output) {
    const transcriptionText = output[0] || "";
    const transcriptionParagraph = transcriptionContainer.querySelector('p');

    if (transcriptionParagraph) {
        transcriptionParagraph.textContent += transcriptionText + ' ';
    } else {
        const newParagraph = document.createElement('p');
        newParagraph.textContent = transcriptionText + ' ';
        transcriptionContainer.appendChild(newParagraph);
    }
}

function generateSummary(){ 
    // FUNCTION WHERE A SUMMARY IS GENERATED BECAUSE A PERSON ENTERED THE ROOM
    const summaryParagraph = contentSummary.querySelectorAll('p')[1];
    summaryParagraph.textContent = 'IT WORKS!!!!!!!!!!!!!!!!!';

 }
let previousDetectionsCount = 0;

async function detectObjectsInVideo() {
    if (!videoElement) {
        console.error('Video element not found.');
        return;
    }
    // Reproducir video y capturar los frames
    videoElement.addEventListener('play', async function () {
        while (!videoElement.paused && !videoElement.ended) {
            const frameData = await captureFrame(videoElement);

            // Evitar procesar demasiado rápido los fotogramas (limitamos la frecuencia de detección)
            const currentTime = performance.now();
            if (currentTime - lastDetectionTime < 100) { // Espera de 100ms entre cada detección
                await new Promise(resolve => requestAnimationFrame(resolve));
                continue;
            }
            lastDetectionTime = currentTime;

            // Detectar objetos en el frame

            const detections = await detect(frameData);
            compareDetectionCounts(previousDetectionsCount, detections.length);
            previousDetectionsCount = detections.length;

            //renderDetections(detections);

            await new Promise(resolve => requestAnimationFrame(resolve)); // Espera al siguiente frame
        }
    });
}

function compareDetectionCounts(previousCount, currentCount) {
    if (currentCount > previousCount) {
        console.log(`El número de detecciones ha aumentado: ${previousCount} → ${currentCount}`);
        generateSummary();
    } else if (currentCount < previousCount) {
        console.log(`El número de detecciones ha disminuido: ${previousCount} → ${currentCount}`);
    } else {
        console.log(`El número de detecciones se mantiene igual: ${currentCount}`);
    }
}

// Función para capturar un frame del video con el cual se va a hacer la detección
async function captureFrame(videoElement) {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg');
}

// Función para realizar la detección con el modelo YOLOv10s
async function detect(frameData) {
    const image = await RawImage.fromURL(frameData);
    const inputs = await processor(image);
    const { output0 } = await model({ images: inputs.pixel_values });

    const detections = output0.tolist()[0].map(x => ({
        xmin: x[0], ymin: x[1], xmax: x[2], ymax: x[3], score: x[4], id: x[5]
    }));

    // Filtrar detecciones con un umbral de confianza
    return detections.filter(d => d.score >= THRESHOLD && d.id === 0); // Que supere un umbral además de que SOLO sea detección de persona
}

// Función para renderizar las detecciones en el video

// Iniciar la detección cuando el video comience a reproducirse
