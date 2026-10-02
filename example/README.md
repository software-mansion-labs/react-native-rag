# React Native RAG Example App

## ⚡ Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/software-mansion-labs/react-native-rag.git
cd react-native-rag/example
yarn
```

### ▶️ Run the Example

```bash
yarn expo run:android
# or
yarn expo run:ios
```

## 🚀 Using the Example App

1. Tap the **document icon** to open the document editor.
2. Add or edit documents as needed.
3. Tap the **add button**  to save and index your document for later retrieval.
4. Enter a question about your document in the text input field.
5. Use the **magnifying glass icon** to toggle Retrieval Augmented Generation (RAG) mode:

   * **Enabled**: The app will search for information in your documents to provide contextually relevant answers.
   * **Disabled**: The app will respond using the LLM without referencing your documents.


## Notes

- This example uses [React Native ExecuTorch](https://github.com/software-mansion/react-native-executorch) to run LLMs and Text Embeddings locally. Make sure you have enough RAM available on your device, simulator/emulator to run the models. Checkout [Memory Usage Benchmarks](https://docs.swmansion.com/react-native-executorch/docs/benchmarks/memory-usage#llms).

