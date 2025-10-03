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
  test('add() with provided embedding + metadata', async () => {
    const emb = new MockEmbeddings(3);
    const store = new MemoryVectorStore({ embeddings: emb });
    await store.load();

    await store.add({
      id: 'a',
      document: 'hello',
      embedding: [1, 2, 3],
      metadata: { tag: 't1' },
    });
    await store.add({
      id: 'b',
      document: 'world',
      embedding: [3, 2, 1],
      metadata: { tag: 't2' },
    });

    const res = await store.query({
      queryEmbedding: [1, 2, 3],
      nResults: 2,
    });

    expect(res.map((r) => r.id)).toEqual(['a', 'b']);
    expect(res[0]!.metadata).toEqual({ tag: 't1' });
  });

  test('add() without embedding computes it via embeddings.embed', async () => {
    const emb = new MockEmbeddings(4);
    const store = new MemoryVectorStore({ embeddings: emb });
    await store.load();
    emb.embedCalls = [];

    await store.add({ id: 'x', document: 'Doc X' });
    await store.add({ id: 'y', document: 'Doc Y' });

    expect(emb.embedCalls).toEqual(['Doc X', 'Doc Y']);
    const res = await store.query({ queryText: 'Doc X', nResults: 2 });
    expect(res[0]!.id).toBe('x');
  });

  test('add() rejects duplicate ids', async () => {
    const store = new MemoryVectorStore({ embeddings: new MockEmbeddings(2) });
    await store.load();

    await store.add({ id: '1', document: 'one' });
    await expect(store.add({ id: '1', document: 'uno' })).rejects.toThrow(
      /id already exists/i
    );
  });

  test('add() rejects when both document and embedding are missing', async () => {
    const store = new MemoryVectorStore({ embeddings: new MockEmbeddings(2) });
    await store.load();
    await expect(store.add({ id: 'x' })).rejects.toThrow(
      /document and embedding cannot be both undefined/i
    );
  });

  test('add() rejects embedding dimension mismatch', async () => {
    const store = new MemoryVectorStore({ embeddings: new MockEmbeddings(3) });
    await store.load();

    await store.add({ id: 'a', document: 'foo', embedding: [1, 1, 1] });

    await expect(
      store.add({ id: 'b', document: 'bar', embedding: [1, 2] })
    ).rejects.toThrow(
      /embedding dimension .* does not match collection embedding dimension/i
    );
  });

  test('update() updates document + recomputes embedding when no embedding provided', async () => {
    const emb = new MockEmbeddings(4);
    const store = new MemoryVectorStore({ embeddings: emb });
    await store.load();

    await store.add({ id: 'k', document: 'old doc' });

    emb.embedCalls = [];

    await store.update({ id: 'k', document: 'new doc' });

    expect(emb.embedCalls).toEqual(['new doc']);

    const res = await store.query({ queryText: 'new doc', nResults: 1 });
    expect(res[0]!.id).toBe('k');
    expect(res[0]!.document).toBe('new doc');
  });

  test('update() updates embedding + metadata directly', async () => {
    const store = new MemoryVectorStore({ embeddings: new MockEmbeddings(3) });
    await store.load();

    await store.add({ id: 'm', document: 'alpha', metadata: { a: 1 } });

    await store.update({ id: 'm', embedding: [9, 9, 9], metadata: { b: 2 } });

    const res = await store.query({ queryEmbedding: [9, 9, 9], nResults: 1 });

    expect(res[0]!.id).toBe('m');
    expect(res[0]!.metadata).toEqual({ b: 2 });
  });

  test('update() rejects unknown ids', async () => {
    const store = new MemoryVectorStore({ embeddings: new MockEmbeddings(2) });
    await store.load();
    await expect(
      store.update({ id: 'missing', document: 'nope' })
    ).rejects.toThrow(/id not found/i);
  });

  test('delete() by predicate (by id) works', async () => {
    const store = new MemoryVectorStore({ embeddings: new MockEmbeddings(2) });
    await store.load();
    await store.add({ id: 'a', document: 'A' });
    await store.add({ id: 'b', document: 'B' });
    await store.add({ id: 'c', document: 'C' });

    await store.delete({ predicate: (row) => row.id === 'b' });

    const res = await store.query({ queryText: 'A', nResults: 10 });
    const remaining = new Set(res.map((r) => r.id));
    expect(remaining.has('b')).toBe(false);
  });

  test('delete() by predicate works', async () => {
    const store = new MemoryVectorStore({ embeddings: new MockEmbeddings(2) });
    await store.load();
    await store.add({ id: 'a', document: 'keep', metadata: { role: 'x' } });
    await store.add({ id: 'b', document: 'drop', metadata: { role: 'y' } });
    await store.add({ id: 'c', document: 'keep-too', metadata: { role: 'x' } });

    await store.delete({ predicate: (row) => row.metadata!.role === 'y' });

    const res = await store.query({ queryText: 'keep', nResults: 10 });
    const ids = new Set(res.map((r) => r.id));
    expect(ids.has('b')).toBe(false);
  });

  test('delete() supports complex predicates', async () => {
    const store = new MemoryVectorStore({ embeddings: new MockEmbeddings(2) });
    await store.load();
    await store.add({ id: 'a', document: 'doc-a', metadata: { keep: false } });
    await store.add({ id: 'b', document: 'doc-b', metadata: { keep: true } });

    await store.delete({ predicate: (row) => row.metadata!.keep === false });

    const res = await store.query({ queryText: 'doc', nResults: 10 });
    const ids = new Set(res.map((r) => r.id));
    expect(ids.has('a')).toBe(false);
    expect(ids.has('b')).toBe(true);
  });

  test('query() by text returns top n sorted and supports predicate', async () => {
    const store = new MemoryVectorStore({ embeddings: new MockEmbeddings(4) });
    await store.load();
    await store.add({ id: 'p', document: 'apple pie', metadata: { cat: 'x' } });
    await store.add({
      id: 'q',
      document: 'banana split',
      metadata: { cat: 'y' },
    });
    await store.add({
      id: 'r',
      document: 'ripe banana',
      metadata: { cat: 'y' },
    });

    const res = await store.query({
      queryText: 'banana',
      nResults: 2,
      predicate: (row) => row.metadata!.cat === 'y',
    });

    expect(res).toHaveLength(2);
    expect(res.map((x) => x.id)).toEqual(['q', 'r']);
    expect(res[0]!.similarity).toBeGreaterThanOrEqual(res[1]!.similarity);
  });

  test('query() by embeddings works and enforces dimension', async () => {
    const store = new MemoryVectorStore({ embeddings: new MockEmbeddings(3) });
    await store.load();
    await store.add({ id: 'u', document: 'foo', embedding: [1, 0, 0] });
    await store.add({ id: 'v', document: 'bar', embedding: [0, 1, 0] });

    const ok = await store.query({
      queryEmbedding: [0.9, 0.1, 0],
      nResults: 1,
    });
    expect(ok[0]!.id).toBe('u');

    await expect(
      store.query({ queryEmbedding: [1, 2], nResults: 1 })
    ).rejects.toThrow(
      /queryEmbedding dimension .* does not match collection embedding dimension/i
    );
  });

  test('query() arg validation: queryText and queryEmbedding cannot both be undefined', async () => {
    const store = new MemoryVectorStore({ embeddings: new MockEmbeddings(2) });
    await store.load();
    await store.add({ id: 'a', document: 'alpha' });

    await expect(store.query({ nResults: 1 })).rejects.toThrow(
      /queryText and queryEmbedding cannot be both undefined/i
    );

    await expect(
      store.query({ queryText: 'alpha', nResults: 1 })
    ).resolves.toBeDefined();
  });
});
