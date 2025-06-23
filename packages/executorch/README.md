# `@react-native-rag/executorch`

This package provides implementations for the `Embeddings` and `LLM` interfaces from `react-native-rag`, using `react-native-executorch` to run AI models on-device. This enables you to perform inference directly on the user's device, ensuring privacy and offline capabilities.

## Installation

```bash
npm install @react-native-rag/executorch react-native-executorch
```

## Usage

### `ExecuTorchEmbeddings`

This class allows you to use an ExecuTorch-compatible model to generate text embeddings.

```typescript
import { ALL_MINILM_L6_V2, ALL_MINILM_L6_V2_TOKENIZER } from 'react-native-executorch';
import { ExecuTorchEmbeddings } from '@react-native-rag/executorch';

const embeddings = new ExecuTorchEmbeddings({
  modelSource: ALL_MINILM_L6_V2,
  tokenizerSource: ALL_MINILM_L6_V2_TOKENIZER,
});
```

### `ExecuTorchLLM`

This class allows you to use an ExecuTorch-compatible language model for text generation.

```typescript
import {
  LLAMA3_2_1B,
  LLAMA3_2_TOKENIZER,
  LLAMA3_2_TOKENIZER_CONFIG,
} from 'react-native-executorch';
import { ExecuTorchLLM } from '@react-native-rag/executorch';

const llm = new ExecuTorchLLM({
  modelSource: LLAMA3_2_1B,
  tokenizerSource: LLAMA3_2_TOKENIZER,
  tokenizerConfigSource: LLAMA3_2_TOKENIZER_CONFIG,
});
```

### Integration with `react-native-rag`

You can use these classes directly with the `useRAG` hook:

```typescript
import { useRAG } from 'react-native-rag';
import { ExecuTorchLLM, ExecuTorchEmbeddings } from '@react-native-rag/executorch';
import { MemoryVectorStore } from 'react-native-rag';

const App = () => {
  const { isReady, generate } = useRAG({
    llm,
    vectorStore: new MemoryVectorStore({ embeddings }),
  });

  // ... your component logic
};
```

## API Reference

### `ExecuTorchEmbeddings`

Implements the `EmbeddingsInterface` from `react-native-rag`.

#### `constructor(params: ExecuTorchEmbeddingsParams)`

Initializes a new `ExecuTorchEmbeddings` instance.

  * `params.modelSource`: The source for the embedding model (`.pte` file). This can be a local `require()` asset, a remote URL string, or a local file URI string.
  * `params.tokenizerSource`: The source for the model's tokenizer (`.json` file). This can be a local `require()` asset, a remote URL string, or a local file URI string.
  * `params.onDownloadProgress`: (Optional) A callback function `(progress: number) => void` that tracks the download progress of model assets if they are remote.

#### Methods

  * #### `load(): Promise<this>`

    Asynchronously loads the embedding model and tokenizer into memory. This must be called before you can generate embeddings. It ensures that all assets are ready for inference.

  * #### `embed(text: string): Promise<number[]>`

    Generates a vector embedding for a given piece of text.

      * **Returns**: A `Promise` that resolves to an array of numbers, representing the document's vector.

  * #### `delete(): void`

    A placeholder method, as the underlying native module (`TextEmbeddingsModule`) does not currently support manually unloading the model from memory.

### `ExecuTorchLLM`

Implements the `LLMInterface` from `react-native-rag`.

#### `constructor(params: ExecuTorchLLMParams)`

Initializes a new `ExecuTorchLLM` instance.

  * `params.modelSource`: The source for the language model (`.pte` file). This can be a local `require()` asset, a remote URL string, or a local file URI string.
  * `params.tokenizerSource`: The source for the model's tokenizer (`.json` file). This can be a local `require()` asset, a remote URL string, or a local file URI string.
  * `params.tokenizerConfigSource`: The source for the tokenizer's configuration file (`.json` file). This can be a local `require()` asset, a remote URL string, or a local file URI string.
  * `params.onDownloadProgress`: (Optional) A callback `(progress: number) => void` to monitor asset download progress.
  * `params.responseCallback`: (Optional) A callback `(response: string) => void` that is invoked with the full, final generated response string.
  * `params.chatConfig`: (Optional) A configuration object `Partial<ChatConfig>` to customize the model's behavior, such as setting the `temperature`.

#### Methods

  * #### `load(): Promise<this>`

    Asynchronously loads the language model and its associated tokenizer and configuration. This is a required setup step before text generation can occur.

  * #### `generate(messages: Message[], callback: (token: string) => void): Promise<string>`

    Starts generating a response from the LLM based on a history of messages.

      * `messages`: An array of `Message` objects representing the conversation context.
      * `callback`: A function that is called in real-time with each newly generated token, allowing for a streaming response effect.
      * **Returns**: A `Promise` that resolves with the final, complete response string from the model.

  * #### `interrupt(): void`

    Immediately stops any ongoing text generation. This is useful for allowing users to halt a response before it's finished.

  * #### `delete(): void`

    Unloads the model and its assets from memory, freeing up resources. Call this when you are done with the LLM instance to conserve memory.