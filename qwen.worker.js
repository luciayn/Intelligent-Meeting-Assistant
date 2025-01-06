import { TextStreamer, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.0"; 

const TASK_NAME = "text-generation";
const MODEL_NAME = "onnx-community/Qwen2.5-0.5B-Instruct";

let generator = null;
let streamer = null;

self.onmessage = async (e) => {
    switch (e.data.type) {
        case 'load':
            await load();
            break;
        case 'generate_keywords':
            const result_keywords = await generate_keywords(e.data.prompt);
            self.postMessage({ type: 'result_keywords', result_keywords: result_keywords }); // Send the result back
            break;
        case 'generate_ideas':
            console.log(`check entering IDEAS`);
            const result_ideas = await generate_ideas(e.data.prompt);
            self.postMessage({ type: 'result_ideas', result_ideas: result_ideas }); // Send the result back
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
async function generate_keywords(prompt) {
    const output = await generator(prompt, {
        max_new_tokens: 8,
        temperature: 0.2,
        top_p: 0,
        do_sample: true,
        early_stopping: true
    });
    console.log(`keywords format: ${output[0].generated_text[2]['content']}`)
    return JSON.parse(output[0].generated_text[2]['content'])
    // Ya se están enviando tokens de forma incremental con el callback_function
}

// Función para generar una respuesta con el modelo
async function generate_ideas(prompt) {
    const output = await generator(prompt, {
        max_new_tokens: 50,
        temperature: 0.2,
        top_p: 0,
        do_sample: true,
        early_stopping: true
    });
    console.log(`ideas format: ${output[0].generated_text[2]['content']}`)
    return JSON.parse(output[0].generated_text[2]['content'])
    // Ya se están enviando tokens de forma incremental con el callback_function
}

// Función de callback para enviar cada token generado al hilo principal
function callback_function(token) {
    self.postMessage({ type: "token", token });
}




