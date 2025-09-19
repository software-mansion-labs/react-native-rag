import { RAG } from '../rag/rag';
import { MemoryVectorStore } from '../vector_stores/memoryVectorStore';
import type { LLM } from '../interfaces/llm';
import type { QueryResult, Message } from '../types/common';
import type { Embeddings } from '../interfaces/embeddings';

jest.mock('../text-splitters/langchain', () => {
  class MockRecursiveCharacterTextSplitter {
    private chunkSize: number;
    private chunkOverlap: number;
    constructor({
      chunkSize,
      chunkOverlap,
    }: {
      chunkSize: number;
      chunkOverlap: number;
    }) {
      this.chunkSize = chunkSize;
      this.chunkOverlap = chunkOverlap;
    }
    async splitText(text: string): Promise<string[]> {
      const chunks: string[] = [];
      if (this.chunkSize <= 0) return [text];
      let start = 0;
      while (start < text.length) {
        const end = Math.min(start + this.chunkSize, text.length);
        chunks.push(text.slice(start, end));
        if (end === text.length) break;
        start = end - this.chunkOverlap;
        if (start < 0) start = 0;
        if (start >= end) start = end;
      }
      return chunks;
    }
  }

  return {
    __esModule: true,
    RecursiveCharacterTextSplitter: MockRecursiveCharacterTextSplitter,
  };
});

import { RecursiveCharacterTextSplitter } from '../text-splitters/langchain';

class MockEmbeddings implements Embeddings {
  public dim: number;
  public loaded = false;
  public unloadCount = 0;
  public embedCalls: string[] = [];

  constructor(dim = 4) {
    this.dim = dim;
  }

  async load(): Promise<this> {
    this.loaded = true;
    return this;
  }

  async unload(): Promise<void> {
    this.unloadCount += 1;
    this.loaded = false;
  }

  async embed(text: string): Promise<number[]> {
    this.embedCalls.push(text);
    const v = new Array(this.dim).fill(0);
    for (let i = 0; i < text.length; i++) {
      v[i % this.dim] += text.charCodeAt(i);
    }
    return v;
  }
}

class MockLLM implements LLM {
  public loaded = false;
  public generateCalls: { input: Message[]; tokensStreamed: string[] }[] = [];
  public interrupted = 0;

  async load(): Promise<this> {
    this.loaded = true;
    return this;
  }

  async unload(): Promise<void> {
    this.loaded = false;
  }

  async interrupt(): Promise<void> {
    this.interrupted += 1;
  }

  async generate(
    input: Message[],
    callback?: (token: string) => void
  ): Promise<string> {
    const streamed: string[] = [];
    if (callback) {
      for (const t of ['tok1', 'tok2']) {
        callback(t);
        streamed.push(t);
      }
    }
    this.generateCalls.push({ input, tokensStreamed: streamed });
    return 'final-answer';
  }
}

describe('RAG (integration with MemoryVectorStore + MockEmbeddings)', () => {
  test('load()/unload() load both store (via embeddings) and LLM', async () => {
    const embeddings = new MockEmbeddings(4);
    const store = new MemoryVectorStore({ embeddings });
    const llm = new MockLLM();
    const rag = new RAG({ vectorStore: store, llm });

    await rag.load();
    expect(embeddings.loaded).toBe(true);
    expect(llm.loaded).toBe(true);

    await rag.unload();
    expect(embeddings.loaded).toBe(false);
    expect(embeddings.unloadCount).toBe(1);
    expect(llm.loaded).toBe(false);
  });

  test('splitAddDocument(): default splitter => one chunk for short doc; ids returned; chunk is retrievable', async () => {
    const embeddings = new MockEmbeddings(8);
    const store = new MemoryVectorStore({ embeddings });
    const llm = new MockLLM();
    const rag = new RAG({ vectorStore: store, llm });

    await rag.load();

    const doc = 'a short document that fits in one chunk';
    const ids = await rag.splitAddDocument(doc);

    expect(ids).toHaveLength(1);
    expect(embeddings.embedCalls).toContain(doc);

    const results = await store.query({ queryTexts: [doc], nResults: 3 });
    const topIds = results[0]!.map((r) => r.id);
    expect(ids.some((id) => topIds.includes(id))).toBe(true);
  });

  test('splitAddDocument(): supports custom TextSplitter and metadata generator (mocked RecursiveCharacterTextSplitter)', async () => {
    const embeddings = new MockEmbeddings(4);
    const store = new MemoryVectorStore({ embeddings });
    const llm = new MockLLM();
    const rag = new RAG({ vectorStore: store, llm });

    await rag.load();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 5,
      chunkOverlap: 1,
    });
    const metaGen = (chunks: string[]) =>
      chunks.map((c, i) => ({ idx: i, len: c.length }));

    const doc = 'DOC-DOC';
    const ids = await rag.splitAddDocument(doc, metaGen, splitter);

    expect(ids.length).toBeGreaterThan(1);

    const res = await store.query({ queryTexts: ['DOC'], nResults: 10 });
    const gotIds = new Set(res[0]!.map((r) => r.id));
    expect(ids.some((id) => gotIds.has(id))).toBe(true);
  });

  test('addDocument()/updateDocument()/deleteDocument() delegate into the store', async () => {
    const embeddings = new MockEmbeddings(3);
    const store = new MemoryVectorStore({ embeddings });
    const llm = new MockLLM();
    const rag = new RAG({ vectorStore: store, llm });

    await rag.load();

    const spyAdd = jest.spyOn(store, 'add');
    const spyUpdate = jest.spyOn(store, 'update');
    const spyDelete = jest.spyOn(store, 'delete');

    await rag.addDocument({
      ids: ['id1'],
      embeddings: [[1, 0, 0]],
      documents: ['alpha'],
      metadatas: [{ a: 1 }],
    });
    expect(spyAdd).toHaveBeenCalledTimes(1);

    await rag.updateDocument({ ids: ['id1'], documents: ['alpha-new'] });
    expect(spyUpdate).toHaveBeenCalledTimes(1);

    await rag.deleteDocument({ ids: ['id1'] });
    expect(spyDelete).toHaveBeenCalledTimes(1);
  });

  test('generate(): augmentedGeneration=false -> sends original input to LLM; no retrieval', async () => {
    const embeddings = new MockEmbeddings(4);
    const store = new MemoryVectorStore({ embeddings });
    const llm = new MockLLM();
    const rag = new RAG({ vectorStore: store, llm });

    await rag.load();

    const qSpy = jest.spyOn(store, 'query');
    const input: Message[] = [{ role: 'user', content: 'hello there' }];

    const out = await rag.generate({ input, augmentedGeneration: false });
    expect(out).toBe('final-answer');
    expect(qSpy).not.toHaveBeenCalled();

    const llmInput = llm.generateCalls[0]!.input;
    expect(llmInput).toEqual(input);
  });

  test('generate(): retrieves context and appends augmented prompt; streams tokens', async () => {
    const embeddings = new MockEmbeddings(4);
    const store = new MemoryVectorStore({ embeddings });
    const llm = new MockLLM();
    const rag = new RAG({ vectorStore: store, llm });

    await rag.load();

    await store.add({
      ids: ['d1', 'd2'],
      documents: ['bananas are yellow', 'apples are red'],
    });

    const tokens: string[] = [];
    const out = await rag.generate({
      input: [{ role: 'user', content: 'What color are bananas?' }],
      nResults: 2,
      callback: (t) => tokens.push(t),
    });

    expect(out).toBe('final-answer');
    expect(tokens).toEqual(['tok1', 'tok2']);

    const llmInput = llm.generateCalls[0]!.input;
    expect(llmInput).toHaveLength(2);
    expect(llmInput[0]!.content).toBe('What color are bananas?');
    const augmented = llmInput[1]!.content;
    expect(augmented).toMatch(/Message: What color are bananas\?/i);
    expect(augmented).toMatch(/Context:/i);
    expect(augmented).toMatch(/bananas are yellow|apples are red/i);
  });

  test('generate(): supports custom questionGenerator and promptGenerator', async () => {
    const embeddings = new MockEmbeddings(4);
    const store = new MemoryVectorStore({ embeddings });
    const llm = new MockLLM();
    const rag = new RAG({ vectorStore: store, llm });

    await rag.load();

    await store.add({
      ids: ['z'],
      documents: ['custom context'],
    });

    const qSpy = jest.spyOn(store, 'query');

    const questionGenerator = (_msgs: Message[]) => 'custom-question';
    const promptGenerator = (msgs: Message[], docs: QueryResult[]) =>
      `PROMPT(${msgs[msgs.length - 1]!.content}) :: ${docs.map((d) => d.document).join('|')}`;

    await rag.generate({
      input: [{ role: 'user', content: 'orig' }],
      questionGenerator,
      promptGenerator,
    });

    expect(qSpy.mock.calls[0]![0]!.queryTexts).toEqual(['custom-question']);

    const llmInput = llm.generateCalls[0]!.input;
    expect(llmInput[1]!.content).toBe('PROMPT(orig) :: custom context');
  });

  test('generate(): forwards predicate and nResults to store.query', async () => {
    const embeddings = new MockEmbeddings(4);
    const store = new MemoryVectorStore({ embeddings });
    const llm = new MockLLM();
    const rag = new RAG({ vectorStore: store, llm });

    await rag.load();

    await store.add({
      ids: ['a', 'b', 'c'],
      documents: ['keep this', 'drop this', 'keep as well'],
      metadatas: [{ role: 'x' }, { role: 'y' }, { role: 'x' }],
    });

    const predicate = (r: QueryResult) => (r as any).metadata?.role === 'x';
    const qSpy = jest.spyOn(store, 'query');

    await rag.generate({ input: 'filter please', nResults: 2, predicate });

    const args = qSpy.mock.calls[0]![0]!;
    expect(args.nResults).toBe(2);
    expect(args.predicate).toBe(predicate);
  });

  test('generate(): throws on empty conversation or missing last content', async () => {
    const embeddings = new MockEmbeddings(4);
    const store = new MemoryVectorStore({ embeddings });
    const llm = new MockLLM();
    const rag = new RAG({ vectorStore: store, llm });

    await rag.load();

    await expect(rag.generate({ input: [] as Message[] })).rejects.toThrow(
      /No messages provided/i
    );

    await expect(
      rag.generate({ input: [{ role: 'user', content: '' }] })
    ).rejects.toThrow(/Last message has no content/i);
  });

  test('interrupt(): delegates to LLM', async () => {
    const embeddings = new MockEmbeddings(4);
    const store = new MemoryVectorStore({ embeddings });
    const llm = new MockLLM();
    const rag = new RAG({ vectorStore: store, llm });

    await rag.interrupt();
    expect(llm.interrupted).toBe(1);
  });
});
