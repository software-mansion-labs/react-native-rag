/**
 * Returns the dot product of two equal-length vectors.
 * @throws {Error} If vector lengths differ.
 */
export function dotProduct(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must be of the same length');
  }
  return a.reduce((sum, ai, i) => sum + ai * b[i]!, 0);
}

/**
 * Returns the Euclidean (L2) norm of `a`.
 */
export function magnitude(a: number[]): number {
  return Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
}

/**
 * Returns cosine similarity of two vectors in [-1, 1].
 * Inputs must be non-zero vectors.
 */
export function cosine(a: number[], b: number[]): number {
  return dotProduct(a, b) / (magnitude(a) * magnitude(b));
}
