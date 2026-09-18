import type { Message } from 'react-native-rag';
import { DEFAULT_SYSTEM_PROMPT, ExecuTorchLLM } from '../wrappers/llms';

jest.mock('react-native-executorch', () => {
  const runner = {
    reset: jest.fn(),
    stop: jest.fn(),
    dispose: jest.fn(),
    getKVCacheState: jest.fn(() => ({ maxSeqLen: 2048 })),
    generate: jest.fn(
      (_prompt: string, _config: unknown, onToken: (token: string) => void) => {
        ['Hel', 'lo', '<eos>'].forEach(onToken);
        return {};
      }
    ),
  };
  const preprocessor = {
    process: jest.fn((history: unknown) => JSON.stringify(history)),
    // One "token" per character of message content keeps the budget maths readable.
    render: jest.fn((history: { content: string }[]) => ({
      text: history.map(({ content }) => content).join(''),
      mediaMap: new Map(),
    })),
    clear: jest.fn(),
    dispose: jest.fn(),
  };
  const tokenizer = {
    encode: jest.fn((text: string) => new Int32Array(text.length)),
    dispose: jest.fn(),
  };
  return {
    __runner: runner,
    __preprocessor: preprocessor,
    __tokenizer: tokenizer,
    createResourceScope: () => {
      const tracked: { dispose: () => void }[] = [];
      return {
        track: (resource: { dispose: () => void }) => {
          tracked.push(resource);
          return resource;
        },
        dispose: () => tracked.splice(0).forEach((r) => r.dispose()),
      };
    },
    nlp: { loadTokenizer: jest.fn(() => tokenizer) },
    download: jest.fn(async (source: unknown, options?: any) => {
      options?.onProgress?.(1);
      return source;
    }),
    wrapAsync:
      (fn: (...args: any[]) => any) =>
      async (...args: any[]) =>
        fn(...args),
    llm: {
      parseTokenizerConfig: jest.fn(() => ({
        chatTemplate: 'TEMPLATE',
        eosToken: '<eos>',
      })),
      createChatPreprocessor: jest.fn(() => preprocessor),
      createLLMRunner: jest.fn(() => runner),
    },
  };
});
jest.mock('react-native-blob-util', () => ({
  fs: { readFile: jest.fn(async () => '{"chat_template":"TEMPLATE"}') },
}));
jest.mock('react-native-worklets', () => ({
  scheduleOnRN: (fn: (...args: any[]) => void, ...args: any[]) => fn(...args),
}));

const mockExecutorch = jest.requireMock('react-native-executorch');
const mockRunner = mockExecutorch.__runner;
const mockPreprocessor = mockExecutorch.__preprocessor;
const mockTokenizer = mockExecutorch.__tokenizer;

const model = {
  modelPath: 'https://example.com/model.pte',
  tokenizerPath: 'https://example.com/tokenizer.json',
  tokenizerConfigPath: 'https://example.com/tokenizer_config.json',
};
// Most tests assert on the exact history, so the default system prompt is off.
const params = { ...model, systemPrompt: '' };

beforeEach(() => {
  jest.clearAllMocks();
});

describe('ExecuTorchLLM', () => {
  it('downloads, parses the tokenizer config and creates the runner once', async () => {
    const onDownloadProgress = jest.fn();
    const llm = new ExecuTorchLLM({ ...params, onDownloadProgress });

    await llm.load();
    await llm.load();

    expect(mockExecutorch.download).toHaveBeenCalledTimes(1);
    expect(mockExecutorch.download).toHaveBeenCalledWith(
      model,
      expect.objectContaining({ onProgress: onDownloadProgress })
    );
    expect(onDownloadProgress).toHaveBeenCalledWith(1);
    expect(mockExecutorch.llm.createChatPreprocessor).toHaveBeenCalledWith(
      expect.objectContaining({ chatTemplate: 'TEMPLATE' })
    );
    expect(mockExecutorch.llm.createLLMRunner).toHaveBeenCalledTimes(1);
    expect(mockExecutorch.llm.createLLMRunner).toHaveBeenCalledWith(
      model.modelPath,
      model.tokenizerPath,
      undefined
    );
  });

  it('throws when generating before load', async () => {
    const llm = new ExecuTorchLLM(params);
    await expect(llm.generate([], () => {})).rejects.toThrow(/load\(\)/);
  });

  it('renders the full history, streams tokens without EOS and returns the text', async () => {
    const llm = new ExecuTorchLLM({
      ...params,
      generationConfig: { temperature: 0.2 },
    });
    await llm.load();

    const messages: Message[] = [
      { role: 'system', content: 'You are helpful.' },
      { role: 'user', content: 'Hi' },
    ];
    const tokens: string[] = [];
    const result = await llm.generate(messages, (t) => tokens.push(t));

    expect(mockPreprocessor.process).toHaveBeenCalledWith(messages, 2, {
      addGenPrompt: true,
    });
    expect(mockRunner.reset).toHaveBeenCalledTimes(1);
    expect(mockRunner.generate).toHaveBeenCalledWith(
      JSON.stringify(messages),
      { echo: false, temperature: 0.2 },
      expect.any(Function)
    );
    expect(tokens).toEqual(['Hel', 'lo']);
    expect(result).toBe('Hello');
    expect(mockPreprocessor.clear).toHaveBeenCalledTimes(1);
  });

  it('is stateless across turns and keeps augmented RAG turns intact', async () => {
    const llm = new ExecuTorchLLM(params);
    await llm.load();

    const turn1: Message[] = [{ role: 'user', content: 'Q1' }];
    await llm.generate(turn1, () => {});

    const turn2: Message[] = [
      { role: 'user', content: 'Q1' },
      { role: 'assistant', content: 'A1' },
      { role: 'user', content: 'Q2' },
      { role: 'user', content: 'Message: Q2\nContext: doc' },
    ];
    await llm.generate(turn2, () => {});

    expect(mockPreprocessor.process).toHaveBeenNthCalledWith(1, turn1, 1, {
      addGenPrompt: true,
    });
    expect(mockPreprocessor.process).toHaveBeenNthCalledWith(2, turn2, 4, {
      addGenPrompt: true,
    });
    expect(mockRunner.reset).toHaveBeenCalledTimes(2);
    expect(mockExecutorch.llm.createLLMRunner).toHaveBeenCalledTimes(1);
  });

  it('clears the preprocessor even when generation fails', async () => {
    const llm = new ExecuTorchLLM(params);
    await llm.load();
    mockRunner.generate.mockImplementationOnce(() => {
      throw new Error('boom');
    });

    await expect(
      llm.generate([{ role: 'user', content: 'x' }], () => {})
    ).rejects.toThrow('boom');
    expect(mockPreprocessor.clear).toHaveBeenCalledTimes(1);
  });

  it('maps interrupt to runner.stop and unload to dispose', async () => {
    const llm = new ExecuTorchLLM(params);
    await llm.interrupt();
    expect(mockRunner.stop).not.toHaveBeenCalled();

    await llm.load();
    await llm.interrupt();
    expect(mockRunner.stop).toHaveBeenCalledTimes(1);

    await llm.unload();
    expect(mockPreprocessor.dispose).toHaveBeenCalledTimes(1);
    expect(mockRunner.dispose).toHaveBeenCalledTimes(1);
    await expect(
      llm.generate([{ role: 'user', content: 'x' }], () => {})
    ).rejects.toThrow(/load\(\)/);
  });
  it('releases the preprocessor and tokenizer when the runner fails to load', async () => {
    const llm = new ExecuTorchLLM(params);
    mockExecutorch.llm.createLLMRunner.mockImplementationOnce(() => {
      throw new Error('load failed');
    });

    await expect(llm.load()).rejects.toThrow('load failed');
    expect(mockPreprocessor.dispose).toHaveBeenCalledTimes(1);
    expect(mockTokenizer.dispose).toHaveBeenCalledTimes(1);
    expect(mockRunner.dispose).not.toHaveBeenCalled();

    // A retry starts from scratch and succeeds.
    await llm.load();
    expect(mockExecutorch.llm.createChatPreprocessor).toHaveBeenCalledTimes(2);
    await expect(
      llm.generate([{ role: 'user', content: 'x' }], () => {})
    ).resolves.toBe('Hello');
  });

  it('resets the runner after a failed generation without masking the error', async () => {
    const llm = new ExecuTorchLLM(params);
    await llm.load();
    mockRunner.generate.mockImplementationOnce(() => {
      throw new Error('boom');
    });
    // First reset is the one at the start of the worklet, second is the cleanup.
    mockRunner.reset
      .mockImplementationOnce(() => {})
      .mockImplementationOnce(() => {
        throw new Error('reset failed');
      });

    await expect(
      llm.generate([{ role: 'user', content: 'x' }], () => {})
    ).rejects.toThrow('boom');
    expect(mockRunner.reset).toHaveBeenCalledTimes(2);
  });

  it('stops on stopRegex and cuts the match from the response', async () => {
    const llm = new ExecuTorchLLM({ ...params, stopRegex: /<\|endoftext\|>/ });
    await llm.load();
    mockRunner.generate.mockImplementationOnce(
      (_prompt: string, _config: unknown, onToken: (token: string) => void) => {
        ['Done', '.', '<|endoftext|>', 'Human:', ' more'].forEach(onToken);
        return {};
      }
    );

    const tokens: string[] = [];
    const result = await llm.generate([{ role: 'user', content: 'x' }], (t) =>
      tokens.push(t)
    );

    expect(mockRunner.stop).toHaveBeenCalledTimes(1);
    expect(tokens).toEqual(['Done', '.']);
    expect(result).toBe('Done.');
  });

  it('prepends the default system prompt only when the history has none', async () => {
    const llm = new ExecuTorchLLM(model);
    await llm.load();

    const user: Message = { role: 'user', content: 'Hi' };
    await llm.generate([user], () => {});
    expect(mockPreprocessor.process).toHaveBeenLastCalledWith(
      [{ role: 'system', content: DEFAULT_SYSTEM_PROMPT }, user],
      2,
      { addGenPrompt: true }
    );

    const own: Message[] = [{ role: 'system', content: 'Be brief.' }, user];
    await llm.generate(own, () => {});
    expect(mockPreprocessor.process).toHaveBeenLastCalledWith(own, 2, {
      addGenPrompt: true,
    });
  });

  it('drops the oldest turns until the prompt fits the context window', async () => {
    // maxSeqLen 40, maxNewTokens 10 -> 30 "tokens" (characters) of prompt budget.
    mockRunner.getKVCacheState.mockReturnValue({ maxSeqLen: 40 });
    const llm = new ExecuTorchLLM({
      ...params,
      generationConfig: { maxNewTokens: 10 },
    });
    await llm.load();

    const system: Message = { role: 'system', content: 'S'.repeat(5) };
    const last: Message = { role: 'user', content: 'L'.repeat(10) };
    const messages: Message[] = [
      system,
      { role: 'user', content: 'a'.repeat(10) },
      { role: 'assistant', content: 'b'.repeat(10) },
      { role: 'user', content: 'c'.repeat(10) },
      { role: 'assistant', content: 'd'.repeat(5) },
      last,
    ];
    await llm.generate(messages, () => {});

    // 50 chars -> drop the first user turn with its reply (30) -> fits.
    const expected = [system, messages[3], messages[4], last];
    expect(mockPreprocessor.process).toHaveBeenCalledWith(expected, 4, {
      addGenPrompt: true,
    });
  });

  it('keeps system and last message even when they alone exceed the budget', async () => {
    mockRunner.getKVCacheState.mockReturnValue({ maxSeqLen: 40 });
    const llm = new ExecuTorchLLM(params);
    await llm.load();

    const messages: Message[] = [
      { role: 'system', content: 'S'.repeat(30) },
      { role: 'user', content: 'old' },
      { role: 'user', content: 'L'.repeat(30) },
    ];
    await llm.generate(messages, () => {});

    expect(mockPreprocessor.process).toHaveBeenCalledWith(
      [messages[0], messages[2]],
      2,
      { addGenPrompt: true }
    );
  });
});
