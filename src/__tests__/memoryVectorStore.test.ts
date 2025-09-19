import { MemoryVectorStore } from '../vector_stores/memoryVectorStore';
import type { Embeddings } from '../interfaces/embeddings';

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

describe('MemoryVectorStore', () => {
  test('add() with provided embeddings + metadata', async () => {
    const emb = new MockEmbeddings(3);
    const store = new MemoryVectorStore({ embeddings: emb });

    const ids = ['a', 'b'];
    const documents = ['hello', 'world'];
    const embeddings = [
      [1, 2, 3],
      [3, 2, 1],
    ];
    const metadatas = [{ tag: 't1' }, { tag: 't2' }];

    await store.add({ ids, documents, embeddings, metadatas });

    const res = await store.query({
      queryEmbeddings: [[1, 2, 3]],
      nResults: 2,
    });

    expect(res).toHaveLength(1);
    expect(res[0]!.map((r) => r.id)).toEqual(['a', 'b']);
    expect(res[0]![0]!.metadata).toEqual({ tag: 't1' });
  });

  test('add() without embeddings computes them via embeddings.embed', async () => {
    const emb = new MockEmbeddings(4);
    const store = new MemoryVectorStore({ embeddings: emb });

    await store.add({
      ids: ['x', 'y'],
      documents: ['Doc X', 'Doc Y'],
    });

    expect(emb.embedCalls).toEqual(['Doc X', 'Doc Y']);
    const res = await store.query({ queryTexts: ['Doc X'], nResults: 2 });
    expect(res[0]![0]!.id).toBe('x');
  });

  test('add() rejects duplicate ids', async () => {
    const store = new MemoryVectorStore({ embeddings: new MockEmbeddings(2) });

    await store.add({ ids: ['1'], documents: ['one'] });
    await expect(store.add({ ids: ['1'], documents: ['uno'] })).rejects.toThrow(
      /id already exists/i
    );
  });

  test('add() rejects array length mismatches', async () => {
    const store = new MemoryVectorStore({ embeddings: new MockEmbeddings(2) });

    await expect(
      store.add({
        ids: ['1', '2'],
        documents: ['only-one'],
      })
    ).rejects.toThrow(/array length must match ids length/i);
  });

  test('add() rejects embedding dimension mismatch', async () => {
    const store = new MemoryVectorStore({ embeddings: new MockEmbeddings(3) });

    await store.add({
      ids: ['a'],
      documents: ['foo'],
      embeddings: [[1, 1, 1]],
    });

    await expect(
      store.add({
        ids: ['b'],
        documents: ['bar'],
        embeddings: [[1, 2]],
      })
    ).rejects.toThrow(
      /embedding dimension .* does not match collection dimension/i
    );
  });

  test('update() updates document + recomputes embedding when no embedding provided', async () => {
    const emb = new MockEmbeddings(4);
    const store = new MemoryVectorStore({ embeddings: emb });

    await store.add({
      ids: ['k'],
      documents: ['old doc'],
    });

    emb.embedCalls = [];

    await store.update({
      ids: ['k'],
      documents: ['new doc'],
    });

    expect(emb.embedCalls).toEqual(['new doc']);

    const res = await store.query({ queryTexts: ['new doc'], nResults: 1 });
    expect(res[0]![0]!.id).toBe('k');
    expect(res[0]![0]!.document).toBe('new doc');
  });

  test('update() updates embedding + metadata directly', async () => {
    const store = new MemoryVectorStore({ embeddings: new MockEmbeddings(3) });

    await store.add({
      ids: ['m'],
      documents: ['alpha'],
      metadatas: [{ a: 1 }],
    });

    await store.update({
      ids: ['m'],
      embeddings: [[9, 9, 9]],
      metadatas: [{ b: 2 }],
    });

    const res = await store.query({
      queryEmbeddings: [[9, 9, 9]],
      nResults: 1,
    });

    expect(res[0]![0]!.id).toBe('m');
    expect(res[0]![0]!.metadata).toEqual({ b: 2 });
  });

  test('update() rejects unknown ids', async () => {
    const store = new MemoryVectorStore({ embeddings: new MockEmbeddings(2) });
    await expect(
      store.update({ ids: ['missing'], documents: ['nope'] })
    ).rejects.toThrow(/id not found/i);
  });

  test('delete() by ids works', async () => {
    const store = new MemoryVectorStore({ embeddings: new MockEmbeddings(2) });
    await store.add({
      ids: ['a', 'b', 'c'],
      documents: ['A', 'B', 'C'],
    });

    await store.delete({ ids: ['b'] });

    const res = await store.query({ queryTexts: ['A'], nResults: 10 });
    const remaining = new Set(res[0]!.map((r) => r.id));
    expect(remaining.has('b')).toBe(false);
  });

  test('delete() by predicate works', async () => {
    const store = new MemoryVectorStore({ embeddings: new MockEmbeddings(2) });
    await store.add({
      ids: ['a', 'b', 'c'],
      documents: ['keep', 'drop', 'keep-too'],
      metadatas: [{ role: 'x' }, { role: 'y' }, { role: 'x' }],
    });

    await store.delete({
      predicate: (row) => row.metadata!.role === 'y',
    });

    const res = await store.query({ queryTexts: ['keep'], nResults: 10 });
    const ids = new Set(res[0]!.map((r) => r.id));
    expect(ids.has('b')).toBe(false);
  });

  test('delete() by ids + predicate works', async () => {
    const store = new MemoryVectorStore({ embeddings: new MockEmbeddings(2) });
    await store.add({
      ids: ['a', 'b'],
      documents: ['doc-a', 'doc-b'],
      metadatas: [{ keep: false }, { keep: true }],
    });

    await store.delete({
      ids: ['a', 'b'],
      predicate: (row) => row.metadata!.keep === false,
    });

    const res = await store.query({ queryTexts: ['doc'], nResults: 10 });
    const ids = new Set(res[0]!.map((r) => r.id));
    expect(ids.has('a')).toBe(false);
    expect(ids.has('b')).toBe(true);
  });

  test('query() by text returns top n sorted, supports ids filter and predicate', async () => {
    const store = new MemoryVectorStore({ embeddings: new MockEmbeddings(4) });
    await store.add({
      ids: ['p', 'q', 'r'],
      documents: ['apple pie', 'banana split', 'ripe banana'],
      metadatas: [{ cat: 'x' }, { cat: 'y' }, { cat: 'y' }],
    });

    const res = await store.query({
      queryTexts: ['banana'],
      nResults: 2,
      ids: ['q', 'r'],
      predicate: (row) => row.metadata!.cat === 'y',
    });

    expect(res).toHaveLength(1);
    expect(res[0]).toHaveLength(2);
    expect(res[0]!.map((x) => x.id)).toEqual(['q', 'r']);
    expect(res[0]![0]!.similarity).toBeGreaterThanOrEqual(
      res[0]![1]!.similarity
    );
  });

  test('query() by embeddings works and enforces dimension', async () => {
    const store = new MemoryVectorStore({ embeddings: new MockEmbeddings(3) });
    await store.add({
      ids: ['u', 'v'],
      documents: ['foo', 'bar'],
      embeddings: [
        [1, 0, 0],
        [0, 1, 0],
      ],
    });

    const ok = await store.query({
      queryEmbeddings: [[0.9, 0.1, 0]],
      nResults: 1,
    });
    expect(ok[0]![0]!.id).toBe('u');

    await expect(
      store.query({
        queryEmbeddings: [[1, 2]],
        nResults: 1,
      })
    ).rejects.toThrow(
      /embedding dimension .* does not match collection dimension/i
    );
  });

  test('query() arg validation: exactly one of queryTexts or queryEmbeddings must be provided', async () => {
    const store = new MemoryVectorStore({ embeddings: new MockEmbeddings(2) });
    await store.add({ ids: ['a'], documents: ['alpha'] });

    await expect(store.query({ nResults: 1 })).rejects.toThrow(
      /exactly one of queryTexts or queryEmbeddings must be provided/i
    );

    await expect(
      store.query({ queryTexts: ['x'], queryEmbeddings: [[1, 2]], nResults: 1 })
    ).rejects.toThrow(
      /exactly one of queryTexts or queryEmbeddings must be provided/i
    );

    await expect(
      store.query({ queryTexts: ['alpha'], nResults: 1 })
    ).resolves.toBeDefined();
  });

  test('query() rejects unknown ids in ids filter', async () => {
    const store = new MemoryVectorStore({ embeddings: new MockEmbeddings(2) });
    await store.add({ ids: ['a'], documents: ['alpha'] });

    await expect(
      store.query({ queryTexts: ['alpha'], ids: ['missing'], nResults: 1 })
    ).rejects.toThrow(/id not found/i);
  });
});
