# `@react-native-rag/executorch`

This package provides implementations for the `Embeddings` and `LLM` interfaces from `react-native-rag`, using `react-native-executorch` to run AI models on-device. This enables you to perform inference directly on the user's device, ensuring privacy and offline capabilities.

## Installation

```bash
npm install @react-native-rag/executorch react-native-executorch react-native-worklets react-native-blob-util
```

`react-native-worklets` and `react-native-blob-util` are peer dependencies of `react-native-executorch` 0.10 and of this package. On Expo SDK 55 and 56 install `react-native-worklets` with your package manager rather than `npx expo install`, which would pick an older bundled version.

Requirements inherited from `react-native-executorch` 0.10:

- React Native 0.83+ (bare) or Expo SDK 55+ with development builds. Expo Go is not supported.
- The New Architecture enabled.
- iOS 17.0+ and Android 13+ (`minSdkVersion` 26+).

With Expo, set the iOS deployment target through [`expo-build-properties`](https://docs.expo.dev/versions/latest/sdk/build-properties/):

```json
{
  "expo": {
    "plugins": [["expo-build-properties", { "ios": { "deploymentTarget": "17.0" } }]]
  }
}
```

Models are downloaded on first `load()` and cached on the device. No `initExecutorch` call or resource fetcher package is needed anymore.

## Usage

### `ExecuTorchEmbeddings`

This class allows you to use an ExecuTorch-compatible model to generate text embeddings.

```typescript
import { models } from 'react-native-executorch';
import { ExecuTorchEmbeddings } from '@react-native-rag/executorch';

const embeddings = new ExecuTorchEmbeddings(
  models.textEmbeddings.ALL_MINILM_L6_V2.DEFAULT
);
```

Parameters:

| Name                 | Type                         | Description                                                     |
| -------------------- | ---------------------------- | --------------------------------------------------------------- |
| `modelPath`          | `string`                     | URL or local path of the embedding model (`.pte`).              |
| `tokenizerPath`      | `string`                     | URL or local path of the tokenizer (`tokenizer.json`).          |
| `defaultPrompt`      | `string` (optional)          | Prompt prepended to every input before embedding.               |
| `onDownloadProgress` | `(progress: number) => void` | Download progress callback in the `0-1` range.                  |

### `ExecuTorchLLM`

This class allows you to use an ExecuTorch-compatible language model for text generation.

```typescript
import { models } from 'react-native-executorch';
import { ExecuTorchLLM } from '@react-native-rag/executorch';

const llm = new ExecuTorchLLM({
  ...models.llm.LFM2_5_1_2B.DEFAULT,
  generationConfig: { temperature: 0.7, maxNewTokens: 512 },
});
```

Parameters:

| Name                  | Type                         | Description                                                                 |
| --------------------- | ---------------------------- | --------------------------------------------------------------------------- |
| `modelPath`           | `string`                     | URL or local path of the LLM (`.pte`).                                      |
| `tokenizerPath`       | `string`                     | URL or local path of the tokenizer (`tokenizer.json`).                      |
| `tokenizerConfigPath` | `string`                     | URL or local path of the tokenizer config (`tokenizer_config.json`).        |
| `generationConfig`    | `LLMGenerationConfig`        | `temperature`, `maxNewTokens`, `ignoreEos`. See `react-native-executorch`.  |
| `onDownloadProgress`  | `(progress: number) => void` | Download progress callback in the `0-1` range.                              |

Each `generate()` call is stateless: the whole message history is rendered through the model's chat template and fed to the model from a fresh KV cache. The model itself is loaded once by `load()` and kept in memory until `unload()`.

### Choosing a backend

`models.*.DEFAULT` resolves at import time to the best backend linked into your app: Core ML or MLX on iOS devices, Vulkan on Android devices, XNNPACK on the iOS simulator and as the universal fallback. You can pin a variant explicitly, for example `models.textEmbeddings.ALL_MINILM_L6_V2.XNNPACK_FP32`.

> The Android emulator has no GPU that the Vulkan backend can use, and `DEFAULT` does not fall back automatically there. Pin an `XNNPACK_*` variant when running on the emulator.

To limit the native binaries downloaded at install time, add a `react-native-executorch` block to your app's `package.json`:

```json
{
  "react-native-executorch": {
    "features": ["llm", "textEmbeddings"]
  }
}
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

## Migrating from 0.9

- Install `react-native-worklets` and `react-native-blob-util`. Remove `react-native-executorch-expo-resource-fetcher` (or the bare fetcher) and the `initExecutorch(...)` call.
- Model registry accessors changed in `react-native-executorch`: `models.text_embedding.all_minilm_l6_v2()` is now `models.textEmbeddings.ALL_MINILM_L6_V2.DEFAULT`, and `models.llm.qwen3_0_6b()` is now `models.llm.QWEN3_0_6B.DEFAULT`.
- Constructor fields follow the new registry shape: `modelSource`, `tokenizerSource` and `tokenizerConfigSource` are now `modelPath`, `tokenizerPath` and `tokenizerConfigPath`, and only accept strings (URLs or local paths).
- `chatConfig` on `ExecuTorchLLM` is now `generationConfig` with the `react-native-executorch` `LLMGenerationConfig` shape. Put a system prompt in the message history instead, as the first message with `role: 'system'`.
- The `responseCallback` and `messageHistoryCallback` parameters were removed. They were never wired up. Use the token callback passed to `generate()`.
- Minimum iOS version is now 17.0.

## React Native RAG is created by Software Mansion

Since 2012 [Software Mansion](https://swmansion.com) is a software agency with experience in building web and mobile apps. We are Core React Native Contributors and experts in dealing with all kinds of React Native issues. We can help you build your next dream product – [Hire us](https://swmansion.com/contact/projects?utm_source=react-native-rag&utm_medium=readme).

[![swm](https://logo.swmansion.com/logo?color=white&variant=desktop&width=150&tag=react-native-rag-github 'Software Mansion')](https://swmansion.com)
