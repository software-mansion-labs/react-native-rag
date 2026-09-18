import { cosine, dotProduct, magnitude } from '../utils/vectorMath';

describe('dotProduct', () => {
  test('sums the pairwise products', () => {
    expect(dotProduct([1, 2, 3], [4, 5, 6])).toBe(32);
  });

  test('is zero for orthogonal vectors', () => {
    expect(dotProduct([1, 0], [0, 1])).toBe(0);
  });

  test('handles negative components', () => {
    expect(dotProduct([-1, 2], [3, -4])).toBe(-11);
  });

  test('returns zero for empty vectors', () => {
    expect(dotProduct([], [])).toBe(0);
  });

  test('throws on length mismatch', () => {
    expect(() => dotProduct([1, 2], [1])).toThrow(
      'Vectors must be of the same length'
    );
  });
});

describe('magnitude', () => {
  test('returns the Euclidean norm', () => {
    expect(magnitude([3, 4])).toBe(5);
  });

  test('ignores sign', () => {
    expect(magnitude([-3, -4])).toBe(5);
  });

  test('is zero for the zero vector and the empty vector', () => {
    expect(magnitude([0, 0, 0])).toBe(0);
    expect(magnitude([])).toBe(0);
  });
});

describe('cosine', () => {
  test('is 1 for identical direction', () => {
    expect(cosine([1, 2, 3], [2, 4, 6])).toBeCloseTo(1, 10);
  });

  test('is -1 for opposite direction', () => {
    expect(cosine([1, 0], [-1, 0])).toBeCloseTo(-1, 10);
  });

  test('is 0 for orthogonal vectors', () => {
    expect(cosine([1, 0], [0, 1])).toBeCloseTo(0, 10);
  });

  test('is scale invariant', () => {
    expect(cosine([1, 1], [5, 5])).toBeCloseTo(cosine([1, 1], [1, 1]), 10);
  });

  test('matches a hand-computed value', () => {
    // dot = 11, |a| = sqrt(5), |b| = 5
    expect(cosine([1, 2], [3, 4])).toBeCloseTo(11 / (Math.sqrt(5) * 5), 10);
  });

  test('propagates the length mismatch error', () => {
    expect(() => cosine([1, 2, 3], [1, 2])).toThrow(
      'Vectors must be of the same length'
    );
  });

  test('is NaN for a zero vector, as documented inputs must be non-zero', () => {
    expect(cosine([0, 0], [1, 1])).toBeNaN();
  });
});
