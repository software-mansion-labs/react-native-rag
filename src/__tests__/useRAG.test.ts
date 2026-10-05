import { act, renderHook, waitFor } from '@testing-library/react-native';
import { useRAG } from '../hooks/rag';
import type { LLM } from '../interfaces/llm';
import type { VectorStore } from '../interfaces/vectorStore';

jest.mock('../rag/rag', () => {
  const instance = {
    load: jest.fn(async () => {}),
    unload: jest.fn(async () => {}),
    generate: jest.fn(async () => 'answer'),
    interrupt: jest.fn(async () => {}),
    splitAddDocument: jest.fn(async () => ['id-1']),
    addDocument: jest.fn(async () => 'id-2'),
    updateDocument: jest.fn(async () => {}),
    deleteDocument: jest.fn(async () => {}),
  };
  return { __instance: instance, RAG: jest.fn(() => instance) };
});

const { RAG: MockRAG, __instance: rag } = jest.requireMock('../rag/rag');

const vectorStore = {} as VectorStore;
const llm = {} as LLM;

function deferred<T>() {
  let resolve!: (v: T) => void;
  let reject!: (e: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

async function renderReady() {
  const hook = await renderHook(() => useRAG({ vectorStore, llm }));
  await waitFor(() => expect(hook.result.current.isReady).toBe(true));
  return hook;
}

beforeEach(() => {
  jest.clearAllMocks();
  rag.load.mockImplementation(async () => {});
  rag.generate.mockImplementation(async () => 'answer');
});

describe('useRAG: lifecycle', () => {
  test('constructs RAG with the given store and llm and loads on mount', async () => {
    const { result } = await renderReady();

    expect(MockRAG).toHaveBeenCalledWith({ vectorStore, llm });
    expect(rag.load).toHaveBeenCalled();
    expect(result.current.error).toBeNull();
    expect(result.current.isGenerating).toBe(false);
    expect(result.current.isStoring).toBe(false);
    expect(result.current.response).toBe('');
  });

  test('preventLoad skips loading', async () => {
    const { result } = await renderHook(() =>
      useRAG({ vectorStore, llm, preventLoad: true })
    );
    await act(async () => {});

    expect(rag.load).not.toHaveBeenCalled();
    expect(result.current.isReady).toBe(false);
  });

  test('exposes a load failure through error and stays not ready', async () => {
    rag.load.mockRejectedValueOnce(new Error('no model'));
    const { result } = await renderHook(() => useRAG({ vectorStore, llm }));

    await waitFor(() => expect(result.current.error).toBe('no model'));
    expect(result.current.isReady).toBe(false);
  });

  test('unloads on unmount once ready', async () => {
    const { unmount } = await renderReady();
    expect(rag.unload).not.toHaveBeenCalled();

    await unmount();
    await waitFor(() => expect(rag.unload).toHaveBeenCalledTimes(1));
  });

  test('does not unload on unmount when never ready', async () => {
    const { unmount } = await renderHook(() =>
      useRAG({ vectorStore, llm, preventLoad: true })
    );
    await act(async () => {});
    await unmount();
    await act(async () => {});

    expect(rag.unload).not.toHaveBeenCalled();
  });
});

describe('useRAG: generate', () => {
  test('throws before ready', async () => {
    const { result } = await renderHook(() =>
      useRAG({ vectorStore, llm, preventLoad: true })
    );
    await act(async () => {});

    await expect(result.current.generate({ input: 'hi' })).rejects.toThrow(
      'RAG not ready.'
    );
    expect(rag.generate).not.toHaveBeenCalled();
  });

  test('delegates, streams tokens into response and forwards the user callback', async () => {
    rag.generate.mockImplementation(async (params: any) => {
      params.callback('Hel');
      params.callback('lo');
      return 'Hello';
    });
    const userCallback = jest.fn();
    const { result } = await renderReady();

    let returned: string | undefined;
    await act(async () => {
      returned = await result.current.generate({
        input: 'hi',
        augmentedGeneration: false,
        nResults: 5,
        callback: userCallback,
      });
    });

    expect(returned).toBe('Hello');
    expect(result.current.response).toBe('Hello');
    expect(userCallback.mock.calls).toEqual([['Hel'], ['lo']]);
    expect(rag.generate).toHaveBeenCalledWith(
      expect.objectContaining({
        input: 'hi',
        augmentedGeneration: false,
        nResults: 5,
        callback: expect.any(Function),
      })
    );
    expect(result.current.isGenerating).toBe(false);
    expect(result.current.error).toBeNull();
  });

  test('resets response at the start of each generation', async () => {
    rag.generate.mockImplementation(async (params: any) => {
      params.callback('x');
      return 'x';
    });
    const { result } = await renderReady();

    await act(async () => {
      await result.current.generate({ input: 'one' });
    });
    expect(result.current.response).toBe('x');

    rag.generate.mockImplementation(async () => 'silent');
    await act(async () => {
      await result.current.generate({ input: 'two' });
    });
    expect(result.current.response).toBe('');
  });

  test('sets isGenerating while pending and rejects a concurrent call', async () => {
    const pending = deferred<string>();
    rag.generate.mockReturnValueOnce(pending.promise);
    const { result } = await renderReady();

    let first: Promise<string> | undefined;
    await act(async () => {
      first = result.current.generate({ input: 'one' });
    });
    expect(result.current.isGenerating).toBe(true);

    await expect(result.current.generate({ input: 'two' })).rejects.toThrow(
      'RAG busy generating.'
    );
    expect(rag.generate).toHaveBeenCalledTimes(1);

    await act(async () => {
      pending.resolve('done');
      await first;
    });
    expect(result.current.isGenerating).toBe(false);
  });

  test('records a generation error, rethrows and clears isGenerating', async () => {
    rag.generate.mockRejectedValueOnce(new Error('boom'));
    const { result } = await renderReady();

    await act(async () => {
      await expect(result.current.generate({ input: 'x' })).rejects.toThrow(
        'boom'
      );
    });

    expect(result.current.error).toBe('boom');
    expect(result.current.isGenerating).toBe(false);
  });
});

describe('useRAG: interrupt', () => {
  test('throws before ready', async () => {
    const { result } = await renderHook(() =>
      useRAG({ vectorStore, llm, preventLoad: true })
    );
    await act(async () => {});

    await expect(result.current.interrupt()).rejects.toThrow('RAG not ready.');
  });

  test('throws when nothing is generating', async () => {
    const { result } = await renderReady();

    await expect(result.current.interrupt()).rejects.toThrow(
      'RAG not generating.'
    );
    expect(rag.interrupt).not.toHaveBeenCalled();
  });

  test('delegates to RAG.interrupt while generating', async () => {
    const pending = deferred<string>();
    rag.generate.mockReturnValueOnce(pending.promise);
    const { result } = await renderReady();

    let gen: Promise<string> | undefined;
    await act(async () => {
      gen = result.current.generate({ input: 'x' });
    });
    expect(result.current.isGenerating).toBe(true);

    await result.current.interrupt();
    expect(rag.interrupt).toHaveBeenCalledTimes(1);

    await act(async () => {
      pending.resolve('partial');
      await gen;
    });
  });
});

describe('useRAG: document operations', () => {
  test.each([
    ['splitAddDocument', { document: 'd' }],
    ['addDocument', { document: 'd' }],
    ['updateDocument', { id: '1', document: 'd' }],
    ['deleteDocument', { predicate: () => true }],
  ] as const)('%s throws before ready', async (method, params) => {
    const { result } = await renderHook(() =>
      useRAG({ vectorStore, llm, preventLoad: true })
    );
    await act(async () => {});

    await expect(
      (result.current[method] as (p: any) => Promise<unknown>)(params)
    ).rejects.toThrow('RAG not ready.');
    expect(rag[method]).not.toHaveBeenCalled();
  });

  test('splitAddDocument delegates and returns the chunk ids', async () => {
    const { result } = await renderReady();
    const textSplitter = { splitText: async (t: string) => [t] };
    const metadataGenerator = (chunks: string[]) => chunks.map(() => ({}));

    let ids: string[] = [];
    await act(async () => {
      ids = await result.current.splitAddDocument({
        document: 'doc',
        textSplitter,
        metadataGenerator,
      });
    });

    expect(ids).toEqual(['id-1']);
    expect(rag.splitAddDocument).toHaveBeenCalledWith({
      document: 'doc',
      textSplitter,
      metadataGenerator,
    });
    expect(result.current.isStoring).toBe(false);
  });

  test('sets isStoring while pending and rejects a concurrent store', async () => {
    const pending = deferred<string>();
    rag.addDocument.mockReturnValueOnce(pending.promise);
    const { result } = await renderReady();

    let first: Promise<string> | undefined;
    await act(async () => {
      first = result.current.addDocument({ document: 'a' });
    });
    expect(result.current.isStoring).toBe(true);

    await expect(
      result.current.splitAddDocument({ document: 'b' })
    ).rejects.toThrow('RAG busy storing.');
    await expect(
      result.current.updateDocument({ id: '1', document: 'b' })
    ).rejects.toThrow('RAG busy storing.');
    await expect(
      result.current.deleteDocument({ predicate: () => true })
    ).rejects.toThrow('RAG busy storing.');

    await act(async () => {
      pending.resolve('id');
      await first;
    });
    expect(result.current.isStoring).toBe(false);
  });

  test('addDocument and splitAddDocument record errors and rethrow', async () => {
    rag.addDocument.mockRejectedValueOnce(new Error('add failed'));
    rag.splitAddDocument.mockRejectedValueOnce(new Error('split failed'));
    const { result } = await renderReady();

    await act(async () => {
      await expect(
        result.current.addDocument({ document: 'x' })
      ).rejects.toThrow('add failed');
    });
    expect(result.current.error).toBe('add failed');

    await act(async () => {
      await expect(
        result.current.splitAddDocument({ document: 'x' })
      ).rejects.toThrow('split failed');
    });
    expect(result.current.error).toBe('split failed');
    expect(result.current.isStoring).toBe(false);
  });

  test('updateDocument and deleteDocument rethrow and clear isStoring', async () => {
    rag.updateDocument.mockRejectedValueOnce(new Error('update failed'));
    rag.deleteDocument.mockRejectedValueOnce(new Error('delete failed'));
    const { result } = await renderReady();

    await act(async () => {
      await expect(result.current.updateDocument({ id: '1' })).rejects.toThrow(
        'update failed'
      );
      await expect(
        result.current.deleteDocument({ predicate: () => true })
      ).rejects.toThrow('delete failed');
    });
    expect(result.current.isStoring).toBe(false);
  });

  test('a successful operation clears a previous error', async () => {
    rag.addDocument.mockRejectedValueOnce(new Error('add failed'));
    const { result } = await renderReady();

    await act(async () => {
      await result.current.addDocument({ document: 'x' }).catch(() => {});
    });
    expect(result.current.error).toBe('add failed');

    await act(async () => {
      await result.current.addDocument({ document: 'y' });
    });
    expect(result.current.error).toBeNull();
  });
});
