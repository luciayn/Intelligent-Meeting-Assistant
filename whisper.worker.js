//Adapted from https://github.com/xenova/whisper-web/tree/main/src


import {
    AutoTokenizer,
    AutoProcessor,
    WhisperForConditionalGeneration,
    TextStreamer,
    full,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.0";


const MAX_NEW_TOKENS = 64;

/**
 * This class uses the Singleton pattern to ensure that only one instance of the model is loaded.
 */
class AutomaticSpeechRecognitionPipeline {
    static model_id = null;
    static tokenizer = null;
    static processor = null;
    static model = null;

    static async getInstance() {
        this.model_id = 'onnx-community/whisper-base';

        this.tokenizer ??= AutoTokenizer.from_pretrained(this.model_id);
        this.processor ??= AutoProcessor.from_pretrained(this.model_id);

        this.model ??= WhisperForConditionalGeneration.from_pretrained(this.model_id, {
            dtype: {
                encoder_model: 'fp32', // 'fp16' works too
                decoder_model_merged: 'q4', // or 'fp32' ('fp16' is broken)
            },
            device: 'webgpu',
        });

        return Promise.all([this.tokenizer, this.processor, this.model]);
    }
}

let processing = false;
async function generate({ audio, language }) {
    if (processing) return;
    processing = true;
    console.log('Generating text...');
    // Tell the main thread we are starting
    self.postMessage({ status: 'start' });
    console.log('Starting processing...');
    // Retrieve the text-generation pipeline.
    const [tokenizer, processor, model] = await AutomaticSpeechRecognitionPipeline.getInstance();

    const callback_function = (output) => {
        self.postMessage({
            status: 'update',
            output
        });
    }
    console.log('Updating streamer...');
    const streamer = new TextStreamer(tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function,
    });

    const inputs = await processor(audio);
    console.log('Generating outputs...',inputs);
    const outputs = await model.generate({
        ...inputs,
        max_new_tokens: MAX_NEW_TOKENS,
        language,
        streamer,
    });

    const outputText = tokenizer.batch_decode(outputs, { skip_special_tokens: true });

    // Send the output back to the main thread
    self.postMessage({
        status: 'complete',
        output: outputText,
    });
    processing = false;
}

async function load() {
    console.log('WHISPER LOADING...');
    self.postMessage({
        status: 'loading',
        data: 'Loading model...'
    });

    // Load the pipeline and save it for future use.
    const [tokenizer, processor, model] = await AutomaticSpeechRecognitionPipeline.getInstance();

    self.postMessage({
        status: 'loading',
        data: 'Compiling shaders and warming up model...'
    });

    // Run model with dummy input to compile shaders
    await model.generate({
        input_features: full([1, 80, 3000], 0.0),
        max_new_tokens: 1,
    });
    self.postMessage({ status: 'ready' });
}
// Listen for messages from the main thread
self.addEventListener('message', async (e) => {
    const { type, data } = e.data;

    switch (type) {
        case 'load':
            load();
            break;

        case 'generate':
            generate(data);
            break;
    }
});
