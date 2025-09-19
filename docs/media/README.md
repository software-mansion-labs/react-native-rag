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

## React Native RAG is created by Software Mansion

Since 2012 [Software Mansion](https://swmansion.com) is a software agency with experience in building web and mobile apps. We are Core React Native Contributors and experts in dealing with all kinds of React Native issues. We can help you build your next dream product – [Hire us](https://swmansion.com/contact/projects?utm_source=react-native-rag&utm_medium=readme).

[![swm](https://logo.swmansion.com/logo?color=white&variant=desktop&width=150&tag=react-native-rag-github 'Software Mansion')](https://swmansion.com)