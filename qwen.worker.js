import { TextStreamer, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.0"; 

const TASK_NAME = "text-generation";
const MODEL_NAME = "onnx-community/Qwen2.5-0.5B-Instruct";

let generator = null;

self.onmessage = async (e) => {
    switch (e.data.type) {
        case 'load':
            await load();
            break;
        case 'generate_keywords':
            const result_keywords = await generate_keywords(e.data.prompt);
            self.postMessage({ type: 'result_keywords', result_keywords: result_keywords }); // Devolver al main las palabras claves generadas
            break;
        case 'generate_ideas':
            const result_ideas = await generate_ideas(e.data.prompt);
            self.postMessage({ type: 'result_ideas', result_ideas: result_ideas }); // Devolver al main las ideas generadas
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

    // WARM-UP: Perform a dummy inference
    await generator("Warm up", {
        max_new_tokens: 1
    });
    self.postMessage({ type: "ready" });
}

// Función para generar una respuesta con el modelo
async function generate_keywords(prompt) {
    const output = await generator(prompt, {
        max_new_tokens: 20,
        temperature: 0.3,
        top_p: 0.9,
        do_sample: true,
        early_stopping: true
    });
    console.log(`keywords format: ${output[0].generated_text[2]['content']}`)
    return output[0].generated_text[2]['content']
}

// Función para generar una respuesta con el modelo
async function generate_ideas(prompt) {
    const output = await generator(prompt, {
        max_new_tokens: 35,
        temperature: 0.3,
        top_p: 0.9,
        do_sample: true,
        early_stopping: true
    });
    console.log(`ideas format: ${output[0].generated_text[2]['content']}`)
    return output[0].generated_text[2]['content']
}
