# React Native RAG

![header](https://github.com/user-attachments/assets/dc07a506-0248-4115-9371-e0d10e6c8792)


Private, local RAGs. Supercharge LLMs with your own knowledge base.

## Navigation

  - [:rocket: Features](#rocket-features)
  - [:earth_africa: Real-World Example](#earth_africa-real-world-example)
  - [:package: Installation](#package-installation)
  - [:iphone: Quickstart - Example App](#iphone-quickstart---example-app)
  - [:books: Usage](#books-usage)
      - [Using the `useRAG` Hook](#1-using-the-userag-hook)
      - [Using the `RAG` Class](#2-using-the-rag-class)
      - [Using RAG Components Separately](#3-using-rag-components-separately)
  - [:book: API Reference](#book-api-reference)
      - [Hooks](#hooks)
      - [Classes](#classes)
      - [Interfaces (for Custom Components)](#interfaces-for-custom-components)
      - [Text Splitters](#text-splitters)
      - [Utilities](#utilities)
      - [Types](#types)
  - [:jigsaw: Using Custom Components](#jigsaw-using-custom-components)
  - [:electric_plug: Plugins](#electric_plug-plugins)
  - [:handshake: Contributing](#handshake-contributing)
  - [:page_facing_up: License](#page_facing_up-license)

## :rocket: Features

  * **Modular**: Use only the components you need. Choose from `LLM`, `Embeddings`, `VectorStore`, and `TextSplitter`.
  * **Extensible**: Create your own components by implementing the `LLM`, `Embeddings`, `VectorStore`, and `TextSplitter` interfaces.
  * **Multiple Integration Options**: Whether you prefer a simple hook (`useRAG`), a powerful class (`RAG`), or direct component interaction, the library adapts to your needs.
  * **On-device Inference**: Powered by `@react-native-rag/executorch`, allowing for private and efficient model execution directly on the user's device.
  * **Vector Store Persistence**: Includes support for SQLite with `@react-native-rag/op-sqlite` to save and manage vector stores locally.
  * **Semantic Search Ready**: Easily implement powerful semantic search in your app by using the `VectorStore` and `Embeddings` components directly.

## :earth_africa: Real-World Example

React Native RAG is powering [Private Mind](https://github.com/software-mansion-labs/private-mind), a privacy-first mobile AI app available on [App Store](https://apps.apple.com/gb/app/private-mind/id6746713439) and [Google Play](https://play.google.com/store/apps/details?id=com.swmansion.privatemind).

<img width="2720" height="1085" alt="Private Mind promo" src="https://github.com/user-attachments/assets/2a5ebb32-0146-4b8e-875f-b25bb5cc50e4" />

## :package: Installation

```sh
npm install react-native-rag
```

You will also need an embeddings model and a large language model. We recommend using [`@react-native-rag/executorch`](packages/executorch/README.md) for on-device inference. To use it, install the following packages:

```sh
npm install @react-native-rag/executorch react-native-executorch
```

For persisting vector stores, you can use [`@react-native-rag/op-sqlite`](packages/op-sqlite/README.md):

## :iphone: Quickstart - Example App

For a complete example app that demonstrates how to use the library, check out the [example app](example).

## :books: Usage

We offer three ways to integrate RAG, depending on your needs.

### 1. Using the `useRAG` Hook

The easiest way to get started. Good for simple use cases where you want to quickly set up RAG.

```tsx
import React, { useState } from 'react';
import { Text } from 'react-native';

import { useRAG, MemoryVectorStore } from 'react-native-rag';
import {
  ALL_MINILM_L6_V2,
  ALL_MINILM_L6_V2_TOKENIZER,
  LLAMA3_2_1B_QLORA,
  LLAMA3_2_1B_TOKENIZER,
  LLAMA3_2_TOKENIZER_CONFIG,
} from 'react-native-executorch';
import {
  ExecuTorchEmbeddings,
  ExecuTorchLLM,
} from '@react-native-rag/executorch';

const vectorStore = new MemoryVectorStore({
  embeddings: new ExecuTorchEmbeddings({
    modelSource: ALL_MINILM_L6_V2,
    tokenizerSource: ALL_MINILM_L6_V2_TOKENIZER,
  }),
});

const llm = new ExecuTorchLLM({
  modelSource: LLAMA3_2_1B_QLORA,
  tokenizerSource: LLAMA3_2_1B_TOKENIZER,
  tokenizerConfigSource: LLAMA3_2_TOKENIZER_CONFIG,
});

const app = () => {
  const rag = useRAG({ vectorStore, llm });
  
  return (
    <Text>{rag.response}</Text>
  );
};
```

### 2. Using the `RAG` Class

For more control over components and configuration.

```tsx
import React, { useEffect, useState } from 'react';
import { Text } from 'react-native';

import { RAG, MemoryVectorStore } from 'react-native-rag';
import {
  ExecuTorchEmbeddings,
  ExecuTorchLLM,
} from '@react-native-rag/executorch';
import {
  ALL_MINILM_L6_V2,
  ALL_MINILM_L6_V2_TOKENIZER,
  LLAMA3_2_1B_QLORA,
  LLAMA3_2_1B_TOKENIZER,
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
        tokenizerSource: LLAMA3_2_1B_TOKENIZER,
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

  return (
    <Text>{response}</Text>
  );
}
```

### 3. Using RAG Components Separately

For advanced use cases requiring fine-grained control.

This is the recommended way you if you want to implement semantic search in your app, use the `VectorStore` and `Embeddings` classes directly.

```tsx
import React, { useEffect, useState } from 'react';
import { Text } from 'react-native';

import { MemoryVectorStore } from 'react-native-rag';
import {
  ExecuTorchEmbeddings,
  ExecuTorchLLM,
} from '@react-native-rag/executorch';
import {
  ALL_MINILM_L6_V2,
  ALL_MINILM_L6_V2_TOKENIZER,
  LLAMA3_2_1B_QLORA,
  LLAMA3_2_1B_TOKENIZER,
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
      // NOTE: Calling load on VectorStore will automatically load the embeddings model
      // so loading the embeddings model separately is not necessary in this case.
      const embeddings = await new ExecuTorchEmbeddings({
        modelSource: ALL_MINILM_L6_V2,
        tokenizerSource: ALL_MINILM_L6_V2_TOKENIZER,
      }).load();

      // Instantiate and load the Large Language Model
      const llm = await new ExecuTorchLLM({
        modelSource: LLAMA3_2_1B_QLORA,
        tokenizerSource: LLAMA3_2_1B_TOKENIZER,
        tokenizerConfigSource: LLAMA3_2_TOKENIZER_CONFIG,
        responseCallback: setResponse,
      }).load();

      // Instantiate and initialize the Vector Store
      const vectorStore = await new MemoryVectorStore({ embeddings }).load();

      setEmbeddings(embeddings);
      setLLM(llm);
      setVectorStore(vectorStore);
    };
    initializeRAG();
  }, []);
  
  return (
    <Text>{response}</Text>
  );
}
```

## :book: API Reference

### Hooks

#### `useRAG(params: UseRAGParams)`

A React hook for Retrieval Augmented Generation (RAG). Manages the RAG system lifecycle, loading, unloading, generation, and document storage.

**Parameters:**

  * `params`: An object containing:
      * `vectorStore`: An instance of a class that implements `VectorStore` interface.
      * `llm`: An instance of a class that implements `LLM` interface.
      * `preventLoad` (optional): A boolean to defer loading the RAG system.

**Returns:** An object with the following properties:

  * `response` (`string`): The current generated text from the LLM.
  * `isReady` (`boolean`): True if the RAG system (Vector Store and LLM) is loaded.
  * `isGenerating` (`boolean`): True if the LLM is currently generating a response.
  * `isStoring` (`boolean`): True if a document operation (add, update, delete) is in progress.
  * `error` (`string | null`): The last error message, if any.
  * `generate`: A function to generate text. See `RAG.generate()` for details.
  * `interrupt`: A function to stop the current generation. See `RAG.interrupt()` for details.
  * `splitAddDocument`: A function to split and add a document. See `RAG.splitAddDocument()` for details.
  * `addDocument`: Adds a document. See `RAG.addDocument()` for details.
  * `updateDocument`: Updates a document. See `RAG.updateDocument()` for details.
  * `deleteDocument`: Deletes a document. See `RAG.deleteDocument()` for details.

### Classes

#### `RAG`

The core class for managing the RAG workflow.

**`constructor(params: RAGParams)`**

  * `params`: An object containing:
      * `vectorStore`: An instance that implements `VectorStore` interface.
      * `llm`: An instance that implements `LLM` interface.

**Methods:**

  * `async load(): Promise<this>`: Initializes the vector store and loads the LLM.
  * `async unload(): Promise<void>`: Unloads the vector store and LLM.
  * `async generate(input: Message[] | string, options?: { augmentedGeneration?: boolean; k?: number; predicate?: (value: SearchResult) => boolean; questionGenerator?: Function; promptGenerator?: Function; callback?: (token: string) => void }): Promise<string>` Generates a response.
    * `input` (`Message[] | string`): A string or an array of `Message` objects.
    * `options` (object, optional): Generation options.
      * `augmentedGeneration` (`boolean`, optional): If `true` (default), retrieves context from the vector store to augment the prompt.
      * `k` (`number`, optional): Number of documents to retrieve (default: `3`).
      * `predicate` (`function`, optional): Function to filter retrieved documents (default: includes all).
      * `questionGenerator` (`function`, optional): Custom question generator.
      * `promptGenerator` (`function`, optional): Custom prompt generator.
      * `callback` (`function`, optional): A function that receives tokens as they are generated.
  * `async splitAddDocument(document: string, metadataGenerator?: (chunks: string[]) => Record<string, any>[], textSplitter?: TextSplitter): Promise<string[]>`: Splits a document into chunks and adds them to the vector store.
  * `async addDocument(document: string, metadata?: Record<string, any>): Promise<string>`: Adds a single document to the vector store.
  * `async updateDocument(id: string, document?: string, metadata?: Record<string, any>): Promise<void>`: Updates a document in the vector store.
  * `async deleteDocument(id: string): Promise<void>`: Deletes a document from the vector store.
  * `async interrupt(): Promise<void>`: Interrupts the ongoing LLM generation.

#### `MemoryVectorStore`

An in-memory implementation of the `VectorStore` interface. Useful for development and testing without persistent storage or when you don't need to save documents across app restarts.

**`constructor(params: { embeddings: Embeddings })`**

  * `params`: Requires an `embeddings` instance to generate vectors for documents.

* `async load(): Promise<this>`: Loads the Embeddings model.

* `async unload(): Promise<void>`: Unloads the Embeddings model.

### Interfaces (for Custom Components)

These interfaces define the contracts for creating your own custom components.

#### `Embeddings`

  * `load: () => Promise<this>`: Loads the embedding model.
  * `unload: () => Promise<void>`: Unloads the model.
  * `embed: (text: string) => Promise<number[]>`: Generates an embedding for a given text.

#### `LLM`

  * `load: () => Promise<this>`: Loads the language model.
  * `interrupt: () => Promise<void>`: Stops the current text generation.
  * `unload: () => Promise<void>`: Unloads the model.
  * `generate: (messages: Message[], callback: (token: string) => void) => Promise<string>`: Generates a response from a list of messages, streaming tokens to the callback.

#### `VectorStore`

  * `load: () => Promise<this>`: Initializes the vector store.
  * `unload: () => Promise<void>`: Unloads the vector store and releases resources.
  * `add(document: string, metadata?: Record<string, any>): Promise<string>`: Adds a document.
  * `update(id: string, document?: string, metadata?: Record<string, any>): Promise<void>`: Updates a document.
  * `delete(id: string): Promise<void>`: Deletes a document.
  * `similaritySearch(query: string, k?: number, predicate?: (value: SearchResult) => boolean): Promise<SearchResult[]>`: Searches for `k` similar documents. Which can be filtered with an optional `predicate` function.

#### `TextSplitter`

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

### Types

```typescript
interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

type ResourceSource = string | number | object;

interface SearchResult {
  id: string;
  content: string;
  metadata?: Record<string, any>;
  similarity: number;
}
```

## :jigsaw: Using Custom Components

Bring your own components by creating classes that implement the `LLM`, `Embeddings`, `VectorStore` and `TextSplitter` interfaces. This allows you to use any model or service that fits your needs.

```typescript
interface Embeddings {
  load: () => Promise<this>;
  unload: () => Promise<void>;
  embed: (text: string) => Promise<number[]>;
}

interface LLM {
  load: () => Promise<this>;
  interrupt: () => Promise<void>;
  unload: () => Promise<void>;
  generate: (
    messages: Message[],
    callback: (token: string) => void
  ) => Promise<string>;
}

interface TextSplitter {
  splitText: (text: string) => Promise<string[]>;
}

interface VectorStore {
  load: () => Promise<this>;
  unload: () => Promise<void>;
  add(document: string, metadata?: Record<string, any>): Promise<string>;
  update(
    id: string,
    document?: string,
    metadata?: Record<string, any>
  ): Promise<void>;
  delete(id: string): Promise<void>;
  similaritySearch(
    query: string,
    k?: number,
    predicate?: (value: SearchResult) => boolean
  ): Promise<SearchResult[]>;
}
```

## :electric_plug: Plugins

  * [`@react-native-rag/executorch`](packages/executorch/README.md): On-device inference with `react-native-executorch`.
  * [`@react-native-rag/op-sqlite`](packages/op-sqlite/README.md): Persisting vector stores using SQLite.

## :handshake: Contributing

Contributions are welcome! See the [contributing guide](CONTRIBUTING.md) to learn about the development workflow.

## :page_facing_up: License

MIT

## React Native RAG is created by Software Mansion

Since 2012 [Software Mansion](https://swmansion.com) is a software agency with experience in building web and mobile apps. We are Core React Native Contributors and experts in dealing with all kinds of React Native issues. We can help you build your next dream product – [Hire us](https://swmansion.com/contact/projects?utm_source=react-native-rag&utm_medium=readme).

[![swm](https://logo.swmansion.com/logo?color=white&variant=desktop&width=150&tag=react-native-rag-github 'Software Mansion')](https://swmansion.com)
