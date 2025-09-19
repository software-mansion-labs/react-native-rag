# `@react-native-rag/op-sqlite`

This package provides a `VectorStore` implementation that uses `op-sqlite` for persistent storage of vector embeddings in a React Native application. It's designed to be used with the `react-native-rag` library.

## Features

  * **Persistent Storage**: Your vector data is saved on the device, so it's available across app launches.
  * **Efficient Similarity Search**: Leverages the vector search capabilities of `op-sqlite` for fast and accurate retrieval.
  * **Seamless Integration**: Designed to work as a drop-in replacement for other `VectorStore` implementations in `react-native-rag`.

## Installation

First, install the necessary packages:

```bash
npm install @react-native-rag/op-sqlite @op-engineering/op-sqlite
```

You also need to follow the installation instructions for `@op-engineering/op-sqlite`.

## Configuration

To enable the required features for vector search, you must add the following configuration to your `package.json` file:

```json
"op-sqlite": {
  "libsql": true,
  "sqliteVec": true
}
```

This configuration ensures that the necessary `libsql` and `sqliteVec` extensions are enabled for `op-sqlite`.

## Usage

To use `OPSQLiteVectorStore`, you need to instantiate it with an `Embeddings` implementation. Here's how you can use it with the `useRAG` hook from `react-native-rag`:

```typescript
import { useRAG } from 'react-native-rag';
import { OPSQLiteVectorStore } from '@react-native-rag/op-sqlite';
import { YourLLM, YourEmbeddings } from './your-implementations'; // Provide your own LLM and Embeddings

const App = () => {
  const { isReady, generate, addDocument } = useRAG({
    llm: new YourLLM(),
    vectorStore: new OPSQLiteVectorStore({
      name: 'my-vector-db',
      embeddings: new YourEmbeddings(),
    }),
  });

  // ... your component logic
};
```

## React Native RAG is created by Software Mansion

Since 2012 [Software Mansion](https://swmansion.com) is a software agency with experience in building web and mobile apps. We are Core React Native Contributors and experts in dealing with all kinds of React Native issues. We can help you build your next dream product – [Hire us](https://swmansion.com/contact/projects?utm_source=react-native-rag&utm_medium=readme).

[![swm](https://logo.swmansion.com/logo?color=white&variant=desktop&width=150&tag=react-native-rag-github 'Software Mansion')](https://swmansion.com)