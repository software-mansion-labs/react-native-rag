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

## API Reference

### `OPSQLiteVectorStore`

#### `constructor(params: OPSQLiteVectorStoreParams)`

Initializes a new instance of the vector store.

  * `params.name`: The name for the SQLite database file (e.g., `'my-vectors.db'`).
  * `params.embeddings`: An instance of a class that implements the `Embeddings` interface, used to convert documents into numerical vectors.

#### Methods

The `OPSQLiteVectorStore` class implements the `VectorStore` interface from `react-native-rag`.

  * #### `load(): Promise<this>`

    Initializes the vector store by preparing the database. It creates the necessary `vectors` table with a specialized index for efficient similarity searching. This asynchronous method should be called once before performing any other operations.

  * #### `unload(): Promise<void>`

    Releases any resources used by the vector store. This will call `unload` on the associated `Embeddings` instance and close the SQLite database connection. Use this to free resources when the vector store is no longer needed.

  * #### `add(document: string, metadata?: object): Promise<string>`

    Adds a new document to the store. It generates a vector embedding from the document's content and saves the document, its embedding, and any optional metadata.

      * **Returns**: A `Promise` that resolves with the unique ID (`uuid`) of the newly added document.

  * #### `update(id: string, document?: string, metadata?: object): Promise<void>`

    Modifies an existing document identified by its ID. You can update the document's text content, its metadata, or both. If the document content is changed, a new vector embedding will be automatically generated and stored.

      * **Throws**: An error if the document with the specified `id` does not exist.

  * #### `delete(id: string): Promise<void>`

    Removes a document and its corresponding vector from the store using its unique ID.

      * **Throws**: An error if the document with the specified `id` does not exist.

  * #### `similaritySearch(query: string, k?: number): Promise<object[]>`

    Finds the most relevant documents in the store based on a query string. It generates an embedding for the query and returns the top `k` documents that are most similar.

      * `query`: The text to search for.
      * `k`: The number of similar documents to return. Defaults to `3`.
      * **Returns**: A `Promise` that resolves with an array of result objects, each containing `{ id, content, metadata, similarity }`. The `similarity` score ranges from 0 to 1, where 1 means a perfect match.

  * #### `deleteVectorStore(): Promise<void>`

    A utility method to completely delete the vector table from the database. This will erase all stored documents and embeddings. Use with caution.

## React Native RAG is created by Software Mansion

Since 2012 [Software Mansion](https://swmansion.com) is a software agency with experience in building web and mobile apps. We are Core React Native Contributors and experts in dealing with all kinds of React Native issues. We can help you build your next dream product – [Hire us](https://swmansion.com/contact/projects?utm_source=react-native-rag&utm_medium=readme).

[![swm](https://logo.swmansion.com/logo?color=white&variant=desktop&width=150&tag=react-native-rag-github 'Software Mansion')](https://swmansion.com)