import type { LLM, Message } from 'react-native-rag';
import {
  download,
  llm,
  wrapAsync,
  type LLMModel,
} from 'react-native-executorch';
import RNBlobUtil from 'react-native-blob-util';
import { scheduleOnRN } from 'react-native-worklets';

/**
 * Parameters for {@link ExecuTorchLLM}.
 */
interface ExecuTorchLLMParams extends LLMModel {
  /** Download progress callback (0-1). */
  onDownloadProgress?: (progress: number) => void;
  /** Generation configuration forwarded to ExecuTorch (temperature, max tokens, ...). */
  generationConfig?: llm.LLMGenerationConfig;
}

/**
 * Runs one full generation on the worklet runtime.
 * Resets the KV cache, prefills the rendered prompt and decodes until EOS or stop.
 * Tokens are forwarded to the React Native thread via `scheduleOnRN`.
 */
function generateWorklet(
  runner: llm.LLMRunner,
  prompt: llm.Prompt,
  config: llm.LLMGenerationConfig,
  eosToken: string,
  onToken: (token: string) => void
): string {
  'worklet';
  let response = '';
  runner.reset();
  runner.generate(prompt, config, (token: string) => {
    if (token === eosToken) return;
    response += token;
    scheduleOnRN(onToken, token);
  });
  return response;
}

/**
 * ExecuTorch-based implementation of {@link LLM} for React Native.
 *
 * Each {@link generate} call is stateless: the full message history is rendered
 * through the model's chat template and fed to the runner from a fresh KV cache.
 * The model itself is loaded once in {@link load} and kept in memory.
 */
export class ExecuTorchLLM implements LLM {
  private runner: llm.LLMRunner | null = null;
  private preprocessor: llm.ChatPreprocessor | null = null;
  private eosToken = '';

  private model: LLMModel;
  private onDownloadProgress: (progress: number) => void;
  private generationConfig: llm.LLMGenerationConfig;

  /**
   * Creates a new ExecuTorch LLM instance.
   * @param params - Parameters for the instance.
   * @param params.modelPath - Path or URL of the LLM model (`.pte`).
   * @param params.tokenizerPath - Path or URL of the tokenizer (`tokenizer.json`).
   * @param params.tokenizerConfigPath - Path or URL of the tokenizer config (`tokenizer_config.json`).
   * @param params.onDownloadProgress - Download progress callback (0-1).
   * @param params.generationConfig - Generation configuration forwarded to ExecuTorch.
   */
  constructor({
    onDownloadProgress = () => {},
    generationConfig = {},
    ...model
  }: ExecuTorchLLMParams) {
    this.model = model;
    this.onDownloadProgress = onDownloadProgress;
    // Never echo the rendered prompt back through the token stream.
    this.generationConfig = { echo: false, ...generationConfig };
  }

  /**
   * Downloads (if needed) and loads the model, tokenizer and chat template via `react-native-executorch`.
   * @returns Promise that resolves to the same instance.
   */
  async load() {
    if (!this.runner) {
      const resolved = await download(this.model, {
        onProgress: this.onDownloadProgress,
      });

      const tokenizerConfigStr = await RNBlobUtil.fs.readFile(
        resolved.tokenizerConfigPath,
        'utf8'
      );
      const { chatTemplate, eosToken } = llm.parseTokenizerConfig(
        JSON.parse(tokenizerConfigStr)
      );
      this.eosToken = eosToken;

      this.preprocessor = llm.createChatPreprocessor({
        chatTemplate,
        modalities: resolved.modalities,
        preprocessorConfig: resolved.preprocessorConfig,
      });

      this.runner = await wrapAsync(llm.createLLMRunner)(
        resolved.modelPath,
        resolved.tokenizerPath,
        resolved.modalities
      );
    }
    return this;
  }

  /**
   * Interrupts the current generation. The pending {@link generate} promise
   * resolves with the tokens produced so far.
   */
  async interrupt() {
    this.runner?.stop();
  }

  /**
   * Unloads the underlying model and releases its native resources.
   */
  async unload() {
    this.preprocessor?.dispose();
    this.preprocessor = null;
    this.runner?.dispose();
    this.runner = null;
  }

  /**
   * Generates a completion from a list of messages, streaming tokens to `callback`.
   * @param messages - Conversation history for the model.
   * @param callback - Token-level streaming callback.
   * @returns Promise that resolves to the full generated string.
   */
  async generate(messages: Message[], callback: (token: string) => void) {
    if (!this.runner || !this.preprocessor) {
      throw new Error('LLM not loaded. Call load() first.');
    }

    const prompt = this.preprocessor.process(messages, messages.length, {
      addGenPrompt: true,
    });
    try {
      return await wrapAsync(generateWorklet)(
        this.runner,
        prompt,
        this.generationConfig,
        this.eosToken,
        callback
      );
    } finally {
      this.preprocessor.clear();
    }
  }
}
