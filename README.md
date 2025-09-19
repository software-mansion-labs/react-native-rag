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

<img width="2720" alt="Private Mind promo" src="https://github.com/user-attachments/assets/2a5ebb32-0146-4b8e-875f-b25bb5cc50e4" />

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
import React from 'react';
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

const App = () => {
  const rag = useRAG({ vectorStore, llm });
  return <Text>{rag.response}</Text>;
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

const App = () => {
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
      const ragInstance = new RAG({ llm, vectorStore });

      await ragInstance.load();
      setRag(ragInstance);
    };
    initializeRAG();
  }, []);

  return <Text>{response}</Text>;
};
```

### 3. Using RAG Components Separately

For advanced use cases requiring fine-grained control.

This is the recommended way if you want to implement semantic search in your app - use the `VectorStore` and `Embeddings` classes directly.

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

const App = () => {
  const [embeddings, setEmbeddings] = useState<ExecuTorchEmbeddings | null>(null);
  const [llm, setLLM] = useState<ExecuTorchLLM | null>(null);
  const [vectorStore, setVectorStore] = useState<MemoryVectorStore | null>(null);
  const [response, setResponse] = useState<string | null>(null);

  useEffect(() => {
    const initialize = async () => {
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
    initialize();
  }, []);

  return <Text>{response}</Text>;
};
```

## :jigsaw: Using Custom Components

Bring your own components by creating classes that implement the `LLM`, `Embeddings`, `VectorStore` and `TextSplitter` interfaces. This allows you to use any model or service that fits your needs.

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
