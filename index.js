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
let keywords;
let ideas;

const contentSummary = document.querySelector('.summary'); // Obtener el contenedor del resumen (derecha infererior)
const transcriptionContainer = document.querySelector('.transcription'); // Obtener el contenedor de la transcripción (izquierda)
const keyWordsItems = document.querySelector('.keywords ul'); // Obtener los elementos de la lista de palabras clave (derecha superior) teniendo en cuenta que se trata de una lista donde cada elemento es el ul/li del fromaro que se le ha puesto en el HTML (actualmente tan solo pone "keywords" en el segundo elemento)
const ideasItems = document.querySelector('.ideas ul');
const videoElement = document.querySelector('.video video'); // Obtener el elemento de video centro


const qwenWorker = new Worker('qwen.worker.js', { type: "module" }); // Path to your worker file
qwenWorker.postMessage({ type: 'load' });
qwenWorker.onerror = function (error) {
    console.error('qwenWorker error:', error.message);
};

qwenWorker.onmessage = async (e) => {
    switch(e.data.type) {
        
    case 'result_keywords': // Cuando se genera tokens (palabras clave en nuestro caso de worker) pues las metemos en keyWordsItems[1] que es el segundo elemento de la lista de palabras clave (de momento)
        console.log('GENERANDO PALABRAS CLAVE');
        // Clear the existing list to replace it with new items
        keyWordsItems.innerHTML = '';

        keywords = e.data.result_keywords;
        console.log("Extracted Keywords:", keywords); // Log the keywords

        // Iterate over the new keywords list and create list items
        keywords.forEach((keyword) => {
            const listItem = document.createElement('li'); // Create a new list item
            listItem.textContent = keyword;               // Set the text content to the keyword
            keyWordsItems.appendChild(listItem);      // Append the item to the list container
        });
        await generateIdeas(keywords);
        break;
        
    case 'result_ideas':
        console.log('GENERANDO IDEAS');
        // Clear the existing list to replace it with new items
        ideasItems.innerHTML = '';

        ideas = e.data.result_ideas;
        console.log("Generated Ideas:", ideas); // Log the ideas

        // Iterate over the new ideas list and create list items
        ideas.forEach((idea) => {
            const listItem = document.createElement('li'); // Create a new list item
            listItem.textContent = idea;               // Set the text content to the idea
            ideasItems.appendChild(listItem);      // Append the item to the list container
        });
        break;

    case 'ready':
        console.log('Qwen2.5 worker READY');
        break;
    }
}



const worker = new Worker('whisper.worker.js', { type: "module" });

// Different messages that the worker can receive and how to handle them
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




(async function app() {
    

// Read image and run processor
    if (navigator.mediaDevices.getUserMedia) {
        console.log('Initializing...');
        
    }
}());





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
        await extractKeywords(transcriptionParagraph.textContent); // Mismo tiempo que transcribe actualiza las palabras clave

    } else {
        const newParagraph = document.createElement('p');
        newParagraph.textContent = transcriptionText + ' ';
        transcriptionContainer.appendChild(newParagraph);
        await extractKeywords(transcriptionParagraph.textContent);
    }
}


// Función para extraer palabras clave del prompt generado a partir de la transcripción
async function extractKeywords(transcription) {
    let content = 
        `I have the following text: ${transcription}.
        Based on the information above, extract the keywords that best describe the topic of the text.
        The output must be a list of keywords. For example: ["School", "Student", "Music"]`;
    prompt = [
        {"role":"system","content": "You are a keyword expert generator, whose objective is to generate keywords from a given text."},
        {"role": "user", "content": content},
    ]
    // Generar palabras clave
    qwenWorker.postMessage({ type: 'generate_keywords', prompt});
}

// Función para ideas a partir de las keywords extraídas
async function generateIdeas(keywords) {

    let content = 
        `I have the following keywords: ${keywords}.
        Based on the information above, generate a list of innovative and precise ideas that best describe the topic of the keywords.
         
        Example:
        keywords: ["Project Updates", "Timeline Adjustment", "Customer Feedback Analysis"], 
        Ideas: ["Use the extra week to improve the presentation design and content.", "Assign a team member to focus specifically on the customer feedback analysis section.", "Schedule a review meeting mid-week to ensure the team stays on track."]

        The output MUST be a list containing ONLY the ideas.`;

    prompt = [
        {"role":"system","content": "You are an creative generator, whose objective is to generate ideas from a given list of keywords."},
        {"role": "user", "content": content},
    ]
    // Generar palabras clave
    qwenWorker.postMessage({ type: 'generate_ideas', prompt});
}

function getFullTranscriptionText() {
    let fullText = '';
    const paragraphs = transcriptionContainer.querySelectorAll('p');
    paragraphs.forEach(p => {
      fullText += p.textContent + ' ';
    });
    return fullText.trim();
}

async function generateSummary() {
    const fullText = getFullTranscriptionText();
    const summary = await summarizeTextWithHF(fullText);
  
    const summaryParagraph = contentSummary.querySelectorAll('p')[1];
    summaryParagraph.textContent = summary || "No se pudo generar un resumen.";

    console.log(`Summary: ${summaryParagraph.textContent}`);
}
// Generar resumen de la transcripción
/*function generateSummary(){ 
    // FUNCTION WHERE A SUMMARY IS GENERATED BECAUSE A PERSON ENTERED THE ROOM
    const summaryParagraph = contentSummary.querySelectorAll('p')[1];
    summaryParagraph.textContent = 'TECHNICALLY, A SUMMARY OF THE TRANSCRIPTION WOULD GO HERE (SHOULD BE GENERATED WHEN THE DETECTION COUNT OF PEOPLE INCREASES)';

 }*/
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

