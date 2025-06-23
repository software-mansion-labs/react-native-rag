# React Native RAG

Small and extensible library for creating RAG workflows.

## Navigation

  - [🚀 Features](#-features)
  - [📥 Installation](#-installation)
  - [📚 Usage](#-usage)
      - [Using the `useRAG` Hook](#1-using-the-userag-hook)
      - [Using the `RAG` Class](#2-using-the-rag-class)
      - [Using RAG Components Separately](#3-using-rag-components-separately)
  - [📖 API Reference](#-api-reference)
      - [Hooks](#hooks)
      - [Classes](#classes)
      - [Interfaces (for Custom Components)](#interfaces-for-custom-components)
      - [Text Splitters](#text-splitters)
      - [Utilities](#utilities)
  - [🧩 Using Custom Models](#-using-custom-models)
  - [📱 Example App](#-example-app)
  - [📦 Packages](#-packages)
  - [🤝 Contributing](#-contributing)
  - [📄 License](#-license)

## 🚀 Features

  * **Modular**: Use only the components you need. Choose from `LLM`, `Embeddings`, `VectorStore`, and `TextSplitter`.
  * **Extensible**: Create your own components by implementing the `LLM`, `Embeddings`, `VectorStore`, and `TextSplitter` interfaces.
  * **Multiple Integration Options**: Whether you prefer a simple hook (`useRAG`), a powerful class (`RAG`), or direct component interaction, the library adapts to your needs.
  * **On-device Inference**: Powered by `@react-native-rag/executorch`, allowing for private and efficient model execution directly on the user's device.
  * **Vector Store Persistence**: Includes support for SQLite with `@react-native-rag/op-sqlite` to save and manage vector stores locally.
  * **Semantic Search Ready**: Easily implement powerful semantic search in your app by using the `VectorStore` and `Embeddings` components directly.

## 📥 Installation

```sh
npm install react-native-rag
```

You will also need an embeddings model and a large language model. We recommend using [`@react-native-rag/executorch`](packages/executorch/README.md) for on-device inference. To use it, install the following packages:

```sh
npm install @react-native-rag/executorch react-native-executorch
```

For persisting vector stores, you can use [`@react-native-rag/op-sqlite`](packages/op-sqlite/README.md):

## 📚 Usage

We offer three ways to integrate RAG, depending on your needs.

### 1. Using the `useRAG` Hook

The easiest way to get started. Good for simple use cases where you want to quickly set up RAG.

```jsx
import React, { useState } from 'react';
import { useRAG } from 'react-native-rag';
import {
  ALL_MINILM_L6_V2,
  ALL_MINILM_L6_V2_TOKENIZER,
  LLAMA3_2_1B_QLORA,
  LLAMA3_2_3B_TOKENIZER,
  LLAMA3_2_TOKENIZER_CONFIG,
} from 'react-native-executorch';
import {
  ExecuTorchEmbeddings,
  ExecuTorchLLM,
} from '@react-native-rag/executorch';
import { MemoryVectorStore } from 'react-native-rag';

const app = () => {
  const [response, setResponse] = useState<string | null>(null);

  const { generate } = useRAG({
    vectorStore: new MemoryVectorStore({
        embeddings: new ExecuTorchEmbeddings({
            modelSource: ALL_MINILM_L6_V2,
            tokenizerSource: ALL_MINILM_L6_V2_TOKENIZER,
        })
    }),
    llm: new ExecuTorchLLM({
      modelSource: LLAMA3_2_1B_QLORA,
      tokenizerSource: LLAMA3_2_3B_TOKENIZER,
      tokenizerConfigSource: LLAMA3_2_TOKENIZER_CONFIG,
      responseCallback: setResponse,
    }),
  });

  // ...
}
```

### 2. Using the `RAG` Class

For more control over components and configuration.

```jsx
import React, { useEffect, useState } from 'react';
import { RAG, MemoryVectorStore } from 'react-native-rag';
import {
  ExecuTorchEmbeddings,
  ExecuTorchLLM,
} from '@react-native-rag/executorch';
import {
  ALL_MINILM_L6_V2,
  ALL_MINILM_L6_V2_TOKENIZER,
  LLAMA3_2_1B_QLORA,
  LLAMA3_2_3B_TOKENIZER,
  LLAMA3_2_TOKENIZER_CONFIG,
} from 'react-native-executorch';

const app = () => {
  const [rag, setRag] = useState<RAG | null>(null);
  const [response, setResponse] = useState<string | null>(null);

  useEffect(() => {
    const initializeRAG = async () => {
      const embeddings = new ExecuTorchEmbeddings({
        modelSource: ALL_MINILM_L6_V2,
        tokenizerSource: ALL_MINILM_L6_V2_TOKENIZER,
      });

      const llm = new ExecuTorchLLM({
        modelSource: LLAMA3_2_1B_QLORA,
        tokenizerSource: LLAMA3_2_3B_TOKENIZER,
        tokenizerConfigSource: LLAMA3_2_TOKENIZER_CONFIG,
        responseCallback: setResponse,
      });

      const vectorStore = new MemoryVectorStore({ embeddings });

      const ragInstance = new RAG({
        llm: llm,
        vectorStore: vectorStore,
      });

      await ragInstance.load();
      setRag(ragInstance);
    };
    initializeRAG();
  }, []);
  // ...
}
```

### 3. Using RAG Components Separately

For advanced use cases requiring fine-grained control.

This is the recommended way you if you want to implement semantic search in your app, use the `VectorStore` and `Embeddings` classes directly.

```jsx
import React, { useEffect, useState } from 'react';
import { MemoryVectorStore } from 'react-native-rag';
import {
  ExecuTorchEmbeddings,
  ExecuTorchLLM,
} from '@react-native-rag/executorch';
import {
  ALL_MINILM_L6_V2,
  ALL_MINILM_L6_V2_TOKENIZER,
  LLAMA3_2_1B_QLORA,
  LLAMA3_2_3B_TOKENIZER,
  LLAMA3_2_TOKENIZER_CONFIG,
} from 'react-native-executorch';

const app = () => {
  const [embeddings, setEmbeddings] = useState<ExecuTorchEmbeddings | null>(null);
  const [llm, setLLM] = useState<ExecuTorchLLM | null>(null);
  const [vectorStore, setVectorStore] = useState<MemoryVectorStore | null>(null);
  const [response, setResponse] = useState<string | null>(null);

  useEffect(() => {
    const initializeRAG = async () => {
      // Instantiate and load the Embeddings Model
      // NOTICE: Calling init on VectorStore will automatically load the embeddings model
      // so loading the embeddings model separately is not necessary in this case.
      const embeddings = await new ExecuTorchEmbeddings({
        modelSource: ALL_MINILM_L6_V2,
        tokenizerSource: ALL_MINILM_L6_V2_TOKENIZER,
      }).load();

      // Instantiate and load the Large Language Model
      const llm = await new ExecuTorchLLM({
        modelSource: LLAMA3_2_1B_QLORA,
        tokenizerSource: LLAMA3_2_3B_TOKENIZER,
        tokenizerConfigSource: LLAMA3_2_TOKENIZER_CONFIG,
        responseCallback: setResponse,
      }).load();

      // Instantiate and initialize the Vector Store
      const vectorStore = await new MemoryVectorStore({ embeddings }).init();

      setEmbeddings(embeddings);
      setLLM(llm);
      setVectorStore(vectorStore);
    };
    initializeRAG();
  }, []);
  // ...
}
```

## 📖 API Reference

### Hooks

#### `useRAG(params: UseRAGParams)`

A React hook for Retrieval Augmented Generation (RAG). Manages the RAG system lifecycle, loading, generation, and document storage.

**Parameters:**

  * `params`: An object containing:
      * `vectorStore`: An instance of a class that implements `VectorStoreInterface`.
      * `llm`: An instance of a class that implements `LLMInterface`.
      * `preventLoad` (optional): A boolean to defer loading the RAG system.

**Returns:** An object with the following properties:

  * `response` (`string`): The current generated text from the LLM.
  * `isReady` (`boolean`): True if the RAG system (Vector Store and LLM) is loaded.
  * `isGenerating` (`boolean`): True if the LLM is currently generating a response.
  * `isStoring` (`boolean`): True if a document operation (add, update, delete) is in progress.
  * `error` (`string | null`): The last error message, if any.
  * `generate`: A function to generate text. See `RAG.generate()` for details.
  * `interrupt`: A function to stop the current generation.
  * `splitAddDocument`: A function to split and add a document. See `RAG.splitAddDocument()` for details.
  * `addDocument`: Adds a document. See `RAG.addDocument()` for details.
  * `updateDocument`: Updates a document. See `RAG.updateDocument()` for details.
  * `deleteDocument`: Deletes a document. See `RAG.deleteDocument()` for details.

### Classes

#### `RAG`

The core class for managing the RAG workflow.

**`constructor(params: RAGParams)`**

  * `params`: An object containing:
      * `vectorStore`: An instance that implements `VectorStoreInterface`.
      * `llm`: An instance that implements `LLMInterface`.

**Methods:**

  * `async load(): Promise<this>`: Initializes the vector store and loads the LLM.
  * `async generate(input: Message[] | string, augmentedGeneration?: boolean, options?: object, callback?: (token: string) => void): Promise<string>`: Generates a response.
      * `input`: A string or an array of `Message` objects.
      * `augmentedGeneration`: If `true` (default), retrieves context from the vector store to augment the prompt.
      * `options`: Includes `k` (number of documents to retrieve), `questionGenerator`, and `promptGenerator`.
      * `callback`: A function that receives tokens as they are generated.
  * `async splitAddDocument(document: string, metadataGenerator?: (chunks: string[]) => Record<string, any>[], textSplitter?: TextSplitterInterface): Promise<string[]>`: Splits a document into chunks and adds them to the vector store.
  * `async addDocument(document: string, metadata?: Record<string, any>): Promise<string>`: Adds a single document to the vector store.
  * `async updateDocument(id: string, document?: string, metadata?: Record<string, any>): Promise<void>`: Updates a document in the vector store.
  * `async deleteDocument(id: string): Promise<void>`: Deletes a document from the vector store.
  * `interrupt(): void`: Interrupts the ongoing LLM generation.

#### `MemoryVectorStore`

An in-memory implementation of the `VectorStoreInterface`. Useful for development and testing without persistent storage or when you don't need to save documents across app restarts.

**`constructor(params: { embeddings: EmbeddingsInterface })`**

  * `params`: Requires an `embeddings` instance to generate vectors for documents.

### Interfaces (for Custom Components)

These interfaces define the contracts for creating your own custom components.

#### `EmbeddingsInterface`

  * `load: () => Promise<this>`: Loads the embedding model.
  * `delete: () => void`: Unloads the model.
  * `embed: (text: string) => Promise<number[]>`: Generates an embedding for a given text.

#### `LLMInterface`

  * `load: () => Promise<this>`: Loads the language model.
  * `interrupt: () => void`: Stops the current text generation.
  * `delete: () => void`: Unloads the model.
  * `generate: (messages: Message[], callback: (token: string) => void) => Promise<string>`: Generates a response from a list of messages, streaming tokens to the callback.

#### `VectorStoreInterface`

  * `init: () => Promise<this>`: Initializes the vector store.
  * `add(document: string, metadata?: Record<string, any>): Promise<string>`: Adds a document.
  * `update(id: string, document?: string, metadata?: Record<string, any>): Promise<void>`: Updates a document.
  * `delete(id: string): Promise<void>`: Deletes a document.
  * `similaritySearch(query: string, k?: number): Promise<{ id: string; content: string; ... }[]>`: Searches for `k` similar documents.

#### `TextSplitterInterface`

  * `splitText: (text: string) => Promise<string[]>`: Splits text into an array of chunks.

### Text Splitters

The library provides wrappers around common `langchain` text splitters. All splitters are initialized with `{ chunkSize: number, chunkOverlap: number }`.

  * `RecursiveCharacterTextSplitter`: Splits text recursively by different characters. (Default in `RAG` class).
  * `CharacterTextSplitter`: Splits text by a fixed character count.
  * `TokenTextSplitter`: Splits text by token count.
  * `MarkdownTextSplitter`: Splits text while preserving Markdown structure.
  * `LatexTextSplitter`: Splits text while preserving LaTeX structure.

### Utilities

  * `uuidv4(): string`: Generates a compliant Version 4 UUID. Not cryptographically secure.
  * `cosine(a: number[], b: number[]): number`: Calculates the cosine similarity between two vectors.
  * `dotProduct(a: number[], b: number[]): number`: Calculates the dot product of two vectors.
  * `magnitude(a: number[]): number`: Calculates the Euclidean magnitude of a vector.

## 🧩 Using Custom Models

Bring your own models by creating classes that implement the `LLM`, `Embeddings`, `VectorStore` and `TextSplitter` interfaces. This allows you to use any model or service that fits your needs.

```typescript
interface EmbeddingsInterface {
  load: () => Promise<this>;
  delete: () => void;
  embed: (text: string) => Promise<number[]>;
}

interface LLMInterface {
  load: () => Promise<this>;
  interrupt: () => void;
  delete: () => void;
  generate: (
    messages: Message[],
    callback: (token: string) => void
  ) => Promise<string>;
}

interface TextSplitterInterface {
  splitText: (text: string) => Promise<string[]>;
}

interface VectorStoreInterface {
  init: () => Promise<this>;
  add(document: string, metadata?: Record<string, any>): Promise<string>;
  update(
    id: string,
    document?: string,
    metadata?: Record<string, any>
  ): Promise<void>;
  delete(id: string): Promise<void>;
  similaritySearch(
    query: string,
    k?: number
  ): Promise<
    {
      id: string;
      content: string;
      metadata?: Record<string, any>;
      similarity: number;
    }[]
  >;
}
```

## 📱 Example App

For a complete example app that demonstrates how to use the library, check out the [example app](example).

## 📦 Packages

  * [`@react-native-rag/executorch`](packages/executorch/README.md): On-device inference with `react-native-executorch`.
  * [`@react-native-rag/op-sqlite`](packages/op-sqlite/README.md): Persisting vector stores using SQLite.

## 🤝 Contributing

Contributions are welcome! See the [contributing guide](CONTRIBUTING.md) to learn about the development workflow.

## 📄 License

MIT