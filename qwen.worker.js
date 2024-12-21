import { TextStreamer, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.0";

const TASK_NAME = "text-generation";
//const MODEL_NAME = "onnx-community/Qwen2.5-0.5B-Instruct";
const MODEL_NAME = "onnx-community/Qwen2.5-0.5B-Instruct";

let generator = null;
let streamer = null;

self.onmessage = async (e) => {
    switch (e.data.type) {
        case 'load':
            await load();
            break;
        case 'generate':
            await generate(e.data.prompt);
            break;
    }

};

// Función para cargar el modelo
async function load() {
    console.log('QWEN2.5 LOADING...');
    generator = await pipeline(
        TASK_NAME,
        MODEL_NAME,
        { dtype: "fp16", device: "wasm", }
    );

    streamer = new TextStreamer(generator.tokenizer, {
        skip_prompt: true,
        callback_function,
    });

    // WARM-UP: Perform a dummy inference
    await generator("Warm up", {
        max_new_tokens: 1
    });
    self.postMessage({ type: "ready" });
}

// Función para generar una respuesta con el modelo
async function generate(prompt) {
    const output = await generator(prompt, {
        max_new_tokens: 3,
        temperature: 0,
        top_p: 0,
        do_sample: true,
        early_stopping: true,
        streamer
    });

    // Ya se están enviando tokens de forma incremental con el callback_function
}

// Función de callback para enviar cada token generado al hilo principal
function callback_function(token) {
    self.postMessage({ type: "token", token });
}




