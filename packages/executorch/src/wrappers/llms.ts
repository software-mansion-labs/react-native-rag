import type { LLM, Message } from 'react-native-rag';
import {
  createResourceScope,
  download,
  llm,
  nlp,
  wrapAsync,
  type LLMModel,
  type ResourceScope,
} from 'react-native-executorch';
import RNBlobUtil from 'react-native-blob-util';
import { scheduleOnRN } from 'react-native-worklets';

/**
 * System prompt used when the message history has none.
 * Mirrors the default that `react-native-executorch` 0.9 injected.
 */
export const DEFAULT_SYSTEM_PROMPT =
  "You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Offer suggestions tactfully when appropriate to improve outcomes. Engage in productive collaboration with the user. Don't return too much text.";

/** Tokens kept free for the response when `generationConfig.maxNewTokens` is not set. */
const DEFAULT_RESPONSE_RESERVE_TOKENS = 512;

/**
 * Parameters for {@link ExecuTorchLLM}.
 */
interface ExecuTorchLLMParams extends LLMModel {
  /** Download progress callback (0-1). */
  onDownloadProgress?: (progress: number) => void;
  /** Generation configuration forwarded to ExecuTorch (temperature, max tokens, ...). */
  generationConfig?: llm.LLMGenerationConfig;
  /**
   * System prompt prepended when the message history contains no `system` message.
   * Defaults to {@link DEFAULT_SYSTEM_PROMPT}. Pass an empty string to disable.
   */
  systemPrompt?: string;
  /**
   * Stops generation as soon as the accumulated response matches. The matched text is
   * cut from the returned response. Tokens are streamed to the callback before the
   * check, so a pattern spanning several tokens may reach the callback partially;
   * single-token stops are cut cleanly from both. Useful for models that run past
   * their end of turn, e.g. `/<\|endoftext\|>/` for Qwen.
   */
  stopRegex?: RegExp;
}

/**
 * Runs one full generation on the worklet runtime.
 * Resets the KV cache, prefills the rendered prompt and decodes until EOS, stop
 * or a `stopRegex` match. Tokens are forwarded to the React Native thread via `scheduleOnRN`.
 */
function generateWorklet(
  runner: llm.LLMRunner,
  prompt: llm.Prompt,
  options: {
    readonly config: llm.LLMGenerationConfig;
    readonly eosToken: string;
    readonly stopRegex?: RegExp;
    readonly onToken: (token: string) => void;
  }
): string {
  'worklet';
  const { config, eosToken, stopRegex, onToken } = options;

  let response = '';
  let stopped = false;
  runner.reset();
  runner.generate(prompt, config, (token: string) => {
    if (stopped || token === eosToken) return;

    if (stopRegex) {
      stopRegex.lastIndex = 0;
      const match = stopRegex.exec(response + token);
      if (match) {
        response = (response + token).slice(0, match.index);
        stopped = true;
        runner.stop();
        return;
      }
    }

    response += token;
    scheduleOnRN(onToken, token);
  });
  return response;
}

function countTokensWorklet(tokenizer: nlp.Tokenizer, text: string): number {
  'worklet';
  return tokenizer.encode(text).length;
}

/**
 * ExecuTorch-based implementation of {@link LLM} for React Native.
 *
 * Each {@link generate} call is stateless: the full message history is rendered
 * through the model's chat template and fed to the runner from a fresh KV cache.
 * When the history does not fit the model's context window, the oldest turns are dropped.
 * The model itself is loaded once in {@link load} and kept in memory.
 */
export class ExecuTorchLLM implements LLM {
  private scope: ResourceScope | null = null;
  private runner: llm.LLMRunner | null = null;
  private preprocessor: llm.ChatPreprocessor | null = null;
  private tokenizer: nlp.Tokenizer | null = null;
  private eosToken = '';

  private model: LLMModel;
  private onDownloadProgress: (progress: number) => void;
  private generationConfig: llm.LLMGenerationConfig;
  private systemPrompt: string;
  private stopRegex: RegExp | undefined;

  /**
   * Creates a new ExecuTorch LLM instance.
   * @param params - Parameters for the instance.
   * @param params.modelPath - Path or URL of the LLM model (`.pte`).
   * @param params.tokenizerPath - Path or URL of the tokenizer (`tokenizer.json`).
   * @param params.tokenizerConfigPath - Path or URL of the tokenizer config (`tokenizer_config.json`).
   * @param params.onDownloadProgress - Download progress callback (0-1).
   * @param params.generationConfig - Generation configuration forwarded to ExecuTorch.
   * @param params.systemPrompt - System prompt used when the history has none. Empty string disables it.
   * @param params.stopRegex - Stops generation as soon as the response matches.
   */
  constructor({
    onDownloadProgress = () => {},
    generationConfig = {},
    systemPrompt = DEFAULT_SYSTEM_PROMPT,
    stopRegex,
    ...model
  }: ExecuTorchLLMParams) {
    this.model = model;
    this.onDownloadProgress = onDownloadProgress;
    // Never echo the rendered prompt back through the token stream.
    this.generationConfig = { echo: false, ...generationConfig };
    this.systemPrompt = systemPrompt;
    this.stopRegex = stopRegex;
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

      // Everything native is owned by the scope, so a failure halfway releases what was already created.
      const scope = createResourceScope();
      try {
        const preprocessor = scope.track(
          llm.createChatPreprocessor({
            chatTemplate,
            modalities: resolved.modalities,
            preprocessorConfig: resolved.preprocessorConfig,
          })
        );
        const tokenizer = scope.track(
          await wrapAsync(nlp.loadTokenizer)(resolved.tokenizerPath)
        );
        const runner = scope.track(
          await wrapAsync(llm.createLLMRunner)(
            resolved.modelPath,
            resolved.tokenizerPath,
            resolved.modalities
          )
        );

        this.scope = scope;
        this.eosToken = eosToken;
        this.preprocessor = preprocessor;
        this.tokenizer = tokenizer;
        this.runner = runner;
      } catch (error) {
        scope.dispose();
        throw error;
      }
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
    this.scope?.dispose();
    this.scope = null;
    this.runner = null;
    this.preprocessor = null;
    this.tokenizer = null;
  }

  /**
   * Drops the oldest non-system messages until the rendered prompt leaves room
   * for the response in the model's context window. System messages and the
   * last message are always kept.
   */
  private async fitToContext(
    messages: Message[],
    runner: llm.LLMRunner,
    preprocessor: llm.ChatPreprocessor,
    tokenizer: nlp.Tokenizer
  ): Promise<Message[]> {
    const { maxSeqLen } = runner.getKVCacheState();
    const reserve = Math.min(
      this.generationConfig.maxNewTokens ?? DEFAULT_RESPONSE_RESERVE_TOKENS,
      Math.floor(maxSeqLen / 2)
    );
    const budget = maxSeqLen - reserve;
    const countTokens = wrapAsync(countTokensWorklet);

    let fitted = messages;
    for (;;) {
      const { text } = preprocessor.render(fitted, { addGenPrompt: true });
      if ((await countTokens(tokenizer, text)) <= budget) return fitted;

      const last = fitted.length - 1;
      const oldest = fitted.findIndex(
        (message, index) => message.role !== 'system' && index < last
      );
      if (oldest === -1) return fitted;

      // Drop the oldest message, plus the assistant replies that would be left dangling at the start.
      let end = oldest + 1;
      while (end < last && fitted[end]!.role === 'assistant') end++;
      fitted = [...fitted.slice(0, oldest), ...fitted.slice(end)];
    }
  }

  /**
   * Generates a completion from a list of messages, streaming tokens to `callback`.
   * @param messages - Conversation history for the model.
   * @param callback - Token-level streaming callback.
   * @returns Promise that resolves to the full generated string.
   */
  async generate(messages: Message[], callback: (token: string) => void) {
    const { runner, preprocessor, tokenizer } = this;
    if (!runner || !preprocessor || !tokenizer) {
      throw new Error('LLM not loaded. Call load() first.');
    }

    const hasSystemMessage = messages.some(({ role }) => role === 'system');
    const history: Message[] =
      this.systemPrompt && !hasSystemMessage
        ? [{ role: 'system', content: this.systemPrompt }, ...messages]
        : messages;

    try {
      const fitted = await this.fitToContext(
        history,
        runner,
        preprocessor,
        tokenizer
      );
      const prompt = preprocessor.process(fitted, fitted.length, {
        addGenPrompt: true,
      });
      return await wrapAsync(generateWorklet)(runner, prompt, {
        config: this.generationConfig,
        eosToken: this.eosToken,
        stopRegex: this.stopRegex,
        onToken: callback,
      });
    } catch (error) {
      // Leave a clean KV cache behind. The reset must never replace the error it is unwinding.
      try {
        runner.reset();
      } catch {}
      throw error;
    } finally {
      preprocessor.clear();
    }
  }
}
