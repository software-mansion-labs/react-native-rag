/**
 * Calculates the dot product of two vectors.
 * The dot product is a scalar value that represents the sum of the products of corresponding components of two vectors.
 * It is a measure of how much two vectors are in the same direction.
 * @param a The first vector (array of numbers).
 * @param b The second vector (array of numbers).
 * @returns The dot product of the two vectors.
 * @throws {Error} If the vectors are not of the same length.
 */
export function dotProduct(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must be of the same length');
  }
  return a.reduce((sum, ai, i) => sum + ai * b[i]!, 0);
}

/**
 * Calculates the Euclidean magnitude (or L2-norm) of a vector.
 * The magnitude represents the length of the vector from the origin to its coordinates.
 * @param a The vector (array of numbers).
 * @returns The magnitude of the vector.
 */
export function magnitude(a: number[]): number {
  return Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
}

/**
 * Calculates the cosine similarity between two vectors.
 * Cosine similarity measures the cosine of the angle between two non-zero vectors.
 * It is a measure of similarity between two vectors, ranging from -1 (opposite) to 1 (identical),
 * with 0 indicating orthogonality (no similarity).
 * Note: both vectors must have non-zero magnitude to avoid division by zero.
 * @param a - The first vector.
 * @param b - The second vector.
 * @returns The cosine similarity between the two vectors.
 */
export function cosine(a: number[], b: number[]): number {
  return dotProduct(a, b) / (magnitude(a) * magnitude(b));
}
