# `@react-native-rag/executorch`

This package provides implementations for the `Embeddings` and `LLM` interfaces from `react-native-rag`, using `react-native-executorch` to run AI models on-device. This enables you to perform inference directly on the user's device, ensuring privacy and offline capabilities.

## Installation

```bash
npm install @react-native-rag/executorch react-native-executorch
```

You also need to install a resource fetcher for your setup (e.g. `react-native-executorch-expo-resource-fetcher` for Expo projects) and call `initExecutorch` in your app before using any ExecuTorch modules:

```typescript
import { initExecutorch } from 'react-native-executorch';
import { ExpoResourceFetcher } from 'react-native-executorch-expo-resource-fetcher';

initExecutorch({ resourceFetcher: ExpoResourceFetcher });
```

## Usage

### `ExecuTorchEmbeddings`

This class allows you to use an ExecuTorch-compatible model to generate text embeddings.

```typescript
import { models } from 'react-native-executorch';
import { ExecuTorchEmbeddings } from '@react-native-rag/executorch';

const embeddings = new ExecuTorchEmbeddings(
  models.text_embedding.all_minilm_l6_v2()
);
```

### `ExecuTorchLLM`

This class allows you to use an ExecuTorch-compatible language model for text generation.

```typescript
import { models } from 'react-native-executorch';
import { ExecuTorchLLM } from '@react-native-rag/executorch';

const llm = new ExecuTorchLLM(models.llm.lfm2_5_1_2b_instruct());
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