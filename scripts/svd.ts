/*
 * The largest singular values of a matrix, by a randomized range finder with power iterations (Halko, Martinsson &
 * Tropp 2011), for the offline exports that ask how low-rank a weight update is.
 */
export type Mat = { m: number; n: number; a: Float64Array }
/** Y = A · X (A m × n, X n × k, row-major). */
export function mul(A: Mat, X: Float64Array, k: number) {
  const Y = new Float64Array(A.m * k)
  for (let i = 0; i < A.m; i++) for (let p = 0; p < A.n; p++) { const v = A.a[i * A.n + p]; if (!v) continue; const xo = p * k, yo = i * k; for (let j = 0; j < k; j++) Y[yo + j] += v * X[xo + j] }
  return Y
}
/** Y = Aᵀ · X (X m × k). */
export function mulT(A: Mat, X: Float64Array, k: number) {
  const Y = new Float64Array(A.n * k)
  for (let i = 0; i < A.m; i++) { const xo = i * k; for (let p = 0; p < A.n; p++) { const v = A.a[i * A.n + p]; if (!v) continue; const yo = p * k; for (let j = 0; j < k; j++) Y[yo + j] += v * X[xo + j] } }
  return Y
}
/** Orthonormalize the k columns of an r × k matrix in place (modified Gram–Schmidt, applied twice). */
export function orth(Y: Float64Array, r: number, k: number) {
  for (let j = 0; j < k; j++) {
    for (let pass = 0; pass < 2; pass++) // twice, so columns stay orthogonal even when one is nearly dependent on the others
      for (let i = 0; i < j; i++) { let d = 0; for (let t = 0; t < r; t++) d += Y[t * k + i] * Y[t * k + j]; for (let t = 0; t < r; t++) Y[t * k + j] -= d * Y[t * k + i] }
    let s = 0; for (let t = 0; t < r; t++) s += Y[t * k + j] ** 2
    s = Math.sqrt(s) || 1; for (let t = 0; t < r; t++) Y[t * k + j] /= s
  }
  return Y
}
/** Eigenvalues of a symmetric k × k matrix (cyclic Jacobi). */
export function eig(S: Float64Array, k: number) {
  const a = Float64Array.from(S)
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0
    for (let p = 0; p < k; p++) for (let q = p + 1; q < k; q++) off += a[p * k + q] ** 2
    if (off < 1e-22) break
    for (let p = 0; p < k; p++) for (let q = p + 1; q < k; q++) {
      const apq = a[p * k + q]; if (Math.abs(apq) < 1e-300) continue
      const th = (a[q * k + q] - a[p * k + p]) / (2 * apq), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c
      for (let r = 0; r < k; r++) { const arp = a[r * k + p], arq = a[r * k + q]; a[r * k + p] = c * arp - s * arq; a[r * k + q] = s * arp + c * arq }
      for (let r = 0; r < k; r++) { const apr = a[p * k + r], aqr = a[q * k + r]; a[p * k + r] = c * apr - s * aqr; a[q * k + r] = s * apr + c * aqr }
    }
  }
  return Array.from({ length: k }, (_, i) => a[i * k + i]).sort((x, y) => y - x)
}
/** The k largest singular values of A (randomized range finder with power iterations, Halko et al. 2011). */
export function topSingular(A: Mat, k: number, iters = 4) {
  let seed = 12345
  const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff - 0.5 }
  let Y = orth(mul(A, Float64Array.from({ length: A.n * k }, rnd), k), A.m, k)
  for (let it = 0; it < iters; it++) Y = orth(mul(A, orth(mulT(A, Y, k), A.n, k), k), A.m, k)
  const B = mulT(A, Y, k) // n × k, = (Yᵀ A)ᵀ
  const G = new Float64Array(k * k)
  for (let i = 0; i < k; i++) for (let j = i; j < k; j++) { let s = 0; for (let t = 0; t < A.n; t++) s += B[t * k + i] * B[t * k + j]; G[i * k + j] = G[j * k + i] = s }
  return eig(G, k).map((v) => Math.sqrt(Math.max(0, v)))
}
/** Sum of squares. */
export const frob2 = (a: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return s }
/** Share of ‖A‖² that the top r singular values hold, for each r. */
export const captured = (sigma: number[], total: number, ranks: number[]) => ranks.map((r) => sigma.slice(0, r).reduce((x, v) => x + v * v, 0) / total)
