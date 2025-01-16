import { env,pipeline, AutoProcessor, AutoModel, RawImage } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1'; // importaciones para la ejecución de yolo
import { summarizeTextWithHF } from './summarizer.js';


//Carga de yolo10s
const model_id = 'onnx-community/yolov10s';
const processor = await AutoProcessor.from_pretrained(model_id);
const model = await AutoModel.from_pretrained(model_id);


const THRESHOLD = 0.25; // Umbral de confianza para las detecciones
const PEOPLE = 0; // Cantidad de gente detectada en el video
let lastDetectionTime = 0;


const WHISPER_SAMPLING_RATE = 16000; //sampling rate para la transcripción de video
const MAX_AUDIO_LENGTH = 3.5; // seconds

const MAX_SAMPLES = WHISPER_SAMPLING_RATE * MAX_AUDIO_LENGTH; // max_samples para que haga la transcripción cada max_audio_legth segundos

let language = 'en'; // Lenguaje de la transcripción en inglés pq va mejor
const timeDataQueue = [];

let audioContext;
let stream;
let textBuffer = ''; // Texto acumulado durante 30 segundos para la generación de palabras clave 
let keywords = [];
let ideas = [];

const contentSummary = document.querySelector('.summary'); // Obtener el contenedor del resumen (derecha infererior)
const transcriptionContainer = document.querySelector('.transcription'); // Obtener el contenedor de la transcripción (izquierda)
const keyWordsItems = document.querySelector('.keywords ul'); // Obtener los elementos de la lista de palabras clave (derecha superior) teniendo en cuenta que se trata de una lista donde cada elemento es el ul/li del fromaro que se le ha puesto en el HTML (actualmente tan solo pone "keywords" en el segundo elemento)
const n_ideas = 1;
const videoElement = document.querySelector('.video video'); // Obtener el elemento de video centro


const qwenWorker = new Worker('qwen.worker.js', { type: "module" });
qwenWorker.postMessage({ type: 'load' });
qwenWorker.onerror = function (error) {
    console.error('qwenWorker error:', error.message);
};

qwenWorker.onmessage = async (e) => {
    switch(e.data.type) {
        
    case 'result_keywords':
        keywords = e.data.result_keywords.split(",");
        generateIdeas(keywords);
        console.log("Extracted Keywords:", keywords);

        // Iterar sobre la lista de palabras clave y crear un elemento de lista para cada una
        keywords.forEach((keyword) => {
            const listItem = document.createElement('li'); // Crear elemento de lista
            listItem.textContent = keyword;               // Definir palabra clave
            keyWordsItems.appendChild(listItem);      // Añadir a la lista
        });
        
        break;
        
    case 'result_ideas':
        ideas.push(e.data.result_ideas);
        if (ideas.length==n_ideas){
            displayIdeasWithDragAndDrop(ideas);
        }
        ideas = [];
        break;

    case 'ready':
        console.log('Qwen2.5 worker READY');
        break;
    }
}



const worker = new Worker('whisper.worker.js', { type: "module" });

// Mensajes que puede recibir el worker
worker.onmessage = function (e) {
    switch (e.data.status) {
        case 'loading':
            break;

        case 'initiate':
            break;

        case 'progress':
            break;

        case 'done':
            break;

        case 'ready':
            // Mensaje de que el worker WHISPER esta listo para usarse
            console.log('Whisper Worker READY');
            detectObjectsInVideo();
            startRecordingVideo(videoElement);
            break;

        case 'start':
            break;

        case 'update':
            console.log(e.data.output);
            break;

        case 'complete':
            // Comenzar a transcribir el audio una vez el worker WHISPER esté listo
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




(async function app() {
// Inicialización
    if (navigator.mediaDevices.getUserMedia) {
        console.log('Initializing...');
        
    }
}());


// Start a timer to generate keywords every 30 seconds
setInterval(() => {
    if (textBuffer.trim().length > 0) {
        console.log('Generating keywords for:', textBuffer);
        extractKeywords(textBuffer);
        textBuffer = ''; // Clear the buffer after processing
    }
}, 30000); // 30 seconds (30*1000 miliseconds)


// Función para iniciar la grabación de audio en tiempo real, es decir, con nuestra propia voz (actualmente en desuso posible implementación futura)
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

async function stopRecording() {
    console.log('Stopping recording...');
    if (!audioContext || !stream) return;
    audioContext.close();
    timeDataQueue.length = 0;
    if (stream.getTracks().length > 0) {
        stream.getTracks()[0].stop();
    }
}


// Función para iniciar la grabación del audio del video reproducido en el HTML, es decir, con el audio del video (actualmente en uso)
async function startRecordingVideo(videoElement) {
    try {
        
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
        streamSource.connect(recorder).connect(audioContext.destination);

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

        // Para la transcripción cuando el vídeo termina
        videoElement.addEventListener('ended', async () => {
            console.log('Video playback finished. Stopping transcription.');
            setTimeout(() => {
                worker.terminate(); 
            }, 30 * 1000); // Para el worker a los 30 segundos
        });
    } catch (e) {
        console.error('Error starting recording:', e);
    }
}




// Función para obtener el audio del video para poder procesarlo
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


// FFun
async function updateTranscription(output) {
    const transcriptionText = output[0] || "";
    const transcriptionParagraph = transcriptionContainer.querySelector('p');
    if (transcriptionParagraph) {
        transcriptionParagraph.textContent += transcriptionText + ' ';

    } else {
        const newParagraph = document.createElement('p');
        newParagraph.textContent = transcriptionText + ' ';
        transcriptionContainer.appendChild(newParagraph);
    }

    textBuffer += transcriptionText + ' '; // Añadir nuevo texto al buffer
}

function getFullTranscriptionText() {
    let fullText = '';
    const paragraphs = transcriptionContainer.querySelectorAll('p');
    paragraphs.forEach(p => {
      fullText += p.textContent + ' ';
    });
    return fullText.trim();
}

// Función para extraer palabras clave del prompt generado a partir de la transcripción
async function extractKeywords(transcription) {
    let content = 
        `I have the following text: ${transcription}.
        Based on the information above, extract THREE keywords that best describe the topic of the text.
        Make sure the output is composed by ONLY THREE keywords SEPARATED BY COMMAS (,)`;
        
    prompt = [
        {"role":"system","content": "You are a keyword expert generator, whose objective is to generate keywords from a given text."},
        {"role": "user", "content": content},
    ]
    // Generar palabras clave
    qwenWorker.postMessage({ type: 'generate_keywords', prompt});
}

// Función para ideas a partir de las keywords extraídas
async function generateIdeas(keywords) {
    console.log('Generating ideas for:', keywords);

    let content = 
        `I have the following keywords: ${keywords}.
        Based on the information above, generate a list of ONE innovative and precise idea that best describes the topic of the keywords.
        Make sure the output is composed ONLY by a SINGLE idea.

        Example 1:
        keywords: ["Project Updates", "Timeline Adjustment", "Customer Feedback Analysis"], 
        idea: "Use the extra week to improve the presentation design and content."
        
        Example 2:
        keywords: ["Project Updates", "Timeline Adjustment", "Customer Feedback Analysis"], 
        idea: "Assign a team member to focus specifically on the customer feedback analysis section."
        
        Example 3:
        keywords: ["Project Updates", "Timeline Adjustment", "Customer Feedback Analysis"], 
        idea: "Schedule a review meeting mid-week to ensure the team stays on track."`;

        
    prompt = [
        {"role":"system","content": "You are a creative generator, whose objective is to generate ideas from a given list of keywords."},
        {"role": "user", "content": content},
    ]
    // Generar palabras clave
    qwenWorker.postMessage({ type: 'generate_ideas', prompt});
}

function displayIdeasWithDragAndDrop(ideas) {
    const ideasBoard = document.getElementById("ideas-board");

    ideas.forEach((idea) => {
        // Crear tarjeta para cada idea
        const card = document.createElement("div");
        card.classList.add("card");
        card.textContent = idea;
        card.setAttribute("draggable", "true");

        // Drag-and-drop listeners
        card.addEventListener("dragstart", (e) => {
            e.dataTransfer.setData("text/plain", idea);
            card.classList.add("dragging");
            setTimeout(() => card.classList.add("hidden"), 0);
        });

        card.addEventListener("dragend", () => {
            card.classList.remove("dragging", "hidden");
        });

        ideasBoard.appendChild(card);
    });

    // Drag/arrastrar sobre la zona
    ideasBoard.addEventListener("dragover", (e) => {
        e.preventDefault();
        const draggingCard = document.querySelector(".dragging");
        const afterElement = getDragAfterElement(ideasBoard, e.clientY);

        if (afterElement == null) {
            ideasBoard.appendChild(draggingCard);
        } else {
            ideasBoard.insertBefore(draggingCard, afterElement);
        }
    });

    // Drop/dejar la tarjeta
    ideasBoard.addEventListener("drop", (e) => {
        e.preventDefault();
        const draggedContent = e.dataTransfer.getData("text/plain");
        console.log(`Dropped: ${draggedContent}`);
    });
    
    // Elementos de ideas que se pueden arrastrar
    function getDragAfterElement(container, y) {
        const draggableElements = [
            ...container.querySelectorAll(".card:not(.dragging)"),
        ];

        return draggableElements.reduce(
            (closest, child) => {
                const box = child.getBoundingClientRect();
                const offset = y - box.top - box.height / 2;
                if (offset < 0 && offset > closest.offset) {
                    return { offset, element: child };
                } else {
                    return closest;
                }
            },
            { offset: Number.NEGATIVE_INFINITY }
        ).element;
    }
}

// Generar resumen de la transcripción
async function generateSummary() {
    const fullText = getFullTranscriptionText();
    const summary = await summarizeTextWithHF(fullText);
  
    const summaryParagraph = contentSummary.querySelector('p');
    summaryParagraph.textContent = summary || "No se pudo generar un resumen.";

    console.log(`Summary: ${summaryParagraph.textContent}`);
}

let previousDetectionsCount = 0; // Para quedarse con la detección anterior y compararla con la actual (para ver si ha aumentado o disminuido)
// Función de detección de objetos en el video (en este caso, personas)
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
            compareDetectionCounts(previousDetectionsCount, detections.length); // Comparar el número de detecciones
            previousDetectionsCount = detections.length;

            //renderDetections(detections);

            await new Promise(resolve => requestAnimationFrame(resolve)); // Espera al siguiente frame
        }
    });
}

function compareDetectionCounts(previousCount, currentCount) {
    if (previousCount !== 0) {
        if (currentCount > previousCount) {
            console.log(`El número de detecciones ha aumentado: ${previousCount} → ${currentCount}`);
            generateSummary();
        } else if (currentCount < previousCount) {
            console.log(`El número de detecciones ha disminuido: ${previousCount} → ${currentCount}`);
        } else {
            console.log(`El número de detecciones se mantiene igual: ${currentCount}`);
        }
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

