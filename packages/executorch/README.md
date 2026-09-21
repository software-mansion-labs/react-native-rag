# `@react-native-rag/executorch`

This package provides implementations for the `Embeddings` and `LLM` interfaces from `react-native-rag`, using `react-native-executorch` to run AI models on-device. This enables you to perform inference directly on the user's device, ensuring privacy and offline capabilities.

## Installation

```bash
npm install @react-native-rag/executorch react-native-executorch react-native-worklets react-native-blob-util
```

`react-native-worklets` and `react-native-blob-util` are peer dependencies of `react-native-executorch` 0.10 and of this package. On Expo SDK 55 and 56 install `react-native-worklets` with your package manager rather than `npx expo install`, which would pick an older bundled version.

Requirements inherited from `react-native-executorch` 0.10:

- React Native 0.83+ (bare) or Expo SDK 55+ with development builds. Expo Go is not supported. The upper bound comes from `react-native-worklets`: 0.10.x and 0.11.x support React Native 0.83 to 0.86, 0.12.x up to 0.87.
- The New Architecture enabled.
- iOS 17.0+ and Android 13+ (`minSdkVersion` 26+).
- Optional, Android only: `@kesha-antonov/react-native-background-downloader` (4.4.0+) lets model downloads continue in the background. Without it downloads use the system `DownloadManager`.

With Expo, set the iOS deployment target and the Android `minSdkVersion` through [`expo-build-properties`](https://docs.expo.dev/versions/latest/sdk/build-properties/) (Expo defaults to `minSdkVersion` 24):

```json
{
  "expo": {
    "plugins": [
      [
        "expo-build-properties",
        { "ios": { "deploymentTarget": "17.0" }, "android": { "minSdkVersion": 26 } }
      ]
    ]
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
| `systemPrompt`        | `string` (optional)          | Prepended when the history has no `system` message. Defaults to `DEFAULT_SYSTEM_PROMPT`; pass `''` to disable. |
| `stopRegex`           | `RegExp` (optional)          | Stops generation as soon as the response matches; the match is cut from the result. |
| `onDownloadProgress`  | `(progress: number) => void` | Download progress callback in the `0-1` range.                              |

Each `generate()` call is stateless: the whole message history is rendered through the model's chat template and fed to the model from a fresh KV cache. The model itself is loaded once by `load()` and kept in memory until `unload()`.

When the history does not fit the model's context window, the oldest turns are dropped until the prompt leaves room for the response (`generationConfig.maxNewTokens`, or 512 tokens when unset). System messages and the last message are always kept.

Some models run past their end of turn, for example Qwen 3 can emit `<|endoftext|>` and continue with made-up turns. Pass a `stopRegex` to cut generation there:

```typescript
const llm = new ExecuTorchLLM({
  ...models.llm.QWEN3_0_6B.DEFAULT,
  stopRegex: /<\|endoftext\|>/,
});
```

The pattern is checked against the accumulated response after every token. The match is always removed from the value returned by `generate()`. Tokens are streamed to the callback before the check, so for a pattern that spans several tokens the callback may receive the beginning of the match; single-token stops such as `<|endoftext|>` are cut cleanly from both.

Qwen 3 also needs its official chat template. The `tokenizer_config.json` that `react-native-executorch` 0.10 downloads for `QWEN3_*` carries a Qwen 2.5 style template that replays earlier `<think>` blocks into the prompt and adds its own system line; from the second turn on the model then ends with `<|endoftext|>` before giving an answer. Until the asset is updated, point `tokenizerConfigPath` at the previous release's file, which has the official template:

```typescript
const llm = new ExecuTorchLLM({
  ...models.llm.QWEN3_0_6B.DEFAULT,
  tokenizerConfigPath:
    'https://huggingface.co/software-mansion/react-native-executorch-qwen-3/resolve/v0.9.0/tokenizer_config.json',
  stopRegex: /<\|endoftext\|>/,
});
```

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
- `chatConfig` on `ExecuTorchLLM` is now `generationConfig` with the `react-native-executorch` `LLMGenerationConfig` shape. `chatConfig.systemPrompt` is now the top-level `systemPrompt` parameter (a `system` message in the history takes precedence). `chatConfig.contextStrategy` is gone: the oldest turns are dropped automatically when the history does not fit the context window.
- The `responseCallback` and `messageHistoryCallback` parameters were removed. They were never wired up. Use the token callback passed to `generate()`.
- Minimum iOS version is now 17.0.

## React Native RAG is created by Software Mansion

Since 2012 [Software Mansion](https://swmansion.com) is a software agency with experience in building web and mobile apps. We are Core React Native Contributors and experts in dealing with all kinds of React Native issues. We can help you build your next dream product – [Hire us](https://swmansion.com/contact/projects?utm_source=react-native-rag&utm_medium=readme).

[![swm](https://logo.swmansion.com/logo?color=white&variant=desktop&width=150&tag=react-native-rag-github 'Software Mansion')](https://swmansion.com)
