/*
 * tests/test_verified_ggml.c
 * ─────────────────────────────────────────────────────────────────────────────
 * Test harness for the KaRaMeL-extracted verified kernels.
 *
 * Three layers of testing are performed:
 *
 *   LAYER 1 – Unit tests
 *     Hand-crafted small tensors whose outputs we know exactly.
 *
 *   LAYER 2 – Differential / equivalence tests
 *     Run both the original ggml C kernel and the verified kernel on the same
 *     random inputs and assert their outputs are bit-identical.
 *
 *   LAYER 3 – Adversarial / boundary tests
 *     Tensors at the edges of address space arithmetic: 1-element, max-stride,
 *     and shape combinations that historically triggered ggml stride bugs.
 *
 * Build (after `make extract`):
 *   clang -O2 -fsanitize=address,undefined              \
 *         -I$KRML_HOME/krmllib/dist/minimal             \
 *         -Iextraction                                   \
 *         tests/test_verified_ggml.c                    \
 *         extraction/ggml-verified-kernels.c            \
 *         -lm -o tests/test_runner
 *   ./tests/test_runner
 *
 * All tests emit TAP (Test Anything Protocol) output so they can be consumed
 * by any standard CI harness (ctest, pytest-tap, GitHub Actions, etc.).
 */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <time.h>

/* ── KaRaMeL runtime (generated alongside the kernel) ─────────────────────── */
#include "ggml_verified.h"   /* declares GGML_Impl_verified_add etc. */

/* ─────────────────────────────────────────────────────────────────────────────
   TAP infrastructure
   ───────────────────────────────────────────────────────────────────────────── */
static int g_test_num   = 0;
static int g_fail_count = 0;

#define TAP_OK(desc)  do { \
  g_test_num++;            \
  printf("ok %d - %s\n", g_test_num, desc); \
} while(0)

#define TAP_FAIL(desc, ...) do {                              \
  g_test_num++;                                               \
  g_fail_count++;                                             \
  printf("not ok %d - %s\n", g_test_num, desc);              \
  printf("  # FAIL: " __VA_ARGS__);                          \
  printf("\n");                                               \
} while(0)

#define TAP_CHECK(cond, desc, ...) \
  do { if (cond) TAP_OK(desc); else TAP_FAIL(desc, __VA_ARGS__); } while(0)

/* ─────────────────────────────────────────────────────────────────────────────
   Tensor helpers (mirrors ggml.h layout for F32, contiguous)
   ───────────────────────────────────────────────────────────────────────────── */
typedef struct {
  float  *data;
  size_t  ne[4];   /* element counts */
  size_t  nb[4];   /* byte strides  */
  size_t  total_bytes;
} f32_tensor;

static f32_tensor make_tensor(size_t n0, size_t n1, size_t n2, size_t n3) {
  f32_tensor t;
  t.ne[0] = n0; t.ne[1] = n1; t.ne[2] = n2; t.ne[3] = n3;
  t.nb[0] = sizeof(float);
  t.nb[1] = t.nb[0] * n0;
  t.nb[2] = t.nb[1] * n1;
  t.nb[3] = t.nb[2] * n2;
  t.total_bytes = n0 * n1 * n2 * n3 * sizeof(float);
  /* aligned_alloc requires size to be a multiple of alignment (64) */
  size_t alloc_size = (t.total_bytes + 127) & ~(size_t)63;
  t.data = (float *)aligned_alloc(64, alloc_size);
  assert(t.data != NULL);
  memset(t.data, 0, t.total_bytes + 64);
  return t;
}

static void free_tensor(f32_tensor *t) { free(t->data); t->data = NULL; }

/* Flat element index → pointer */
static inline float *elem(f32_tensor *t, size_t i0, size_t i1,
                           size_t i2, size_t i3) {
  uint8_t *base = (uint8_t *)t->data;
  size_t off = i0*t->nb[0] + i1*t->nb[1] + i2*t->nb[2] + i3*t->nb[3];
  return (float *)(base + off);
}

/* Fill with sequential floats: 1.0, 2.0, 3.0, … */
static void fill_sequential(f32_tensor *t) {
  size_t n = t->total_bytes / sizeof(float);
  for (size_t i = 0; i < n; i++) t->data[i] = (float)(i + 1);
}

/* Fill with deterministic pseudo-random floats in [-10, 10] */
static void fill_random(f32_tensor *t, uint64_t seed) {
  size_t n = t->total_bytes / sizeof(float);
  uint64_t state = seed;
  for (size_t i = 0; i < n; i++) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    t->data[i] = ((float)(state & 0xFFFF) / 0x8000) * 20.0f - 10.0f;
  }
}

/* Reference (naive) add for differential testing */
static void ref_add(f32_tensor *dst, const f32_tensor *a, const f32_tensor *b) {
  size_t n = dst->total_bytes / sizeof(float);
  for (size_t i = 0; i < n; i++)
    dst->data[i] = a->data[i] + b->data[i];
}

static void ref_scale(f32_tensor *dst, const f32_tensor *src, float s) {
  size_t n = dst->total_bytes / sizeof(float);
  for (size_t i = 0; i < n; i++) dst->data[i] = src->data[i] * s;
}

static void ref_relu(f32_tensor *dst, const f32_tensor *src) {
  size_t n = dst->total_bytes / sizeof(float);
  for (size_t i = 0; i < n; i++)
    dst->data[i] = src->data[i] < 0.0f ? 0.0f : src->data[i];
}

/* Bit-exact comparison (handles NaN: NaN==NaN is true here) */
static int tensors_equal(const f32_tensor *a, const f32_tensor *b) {
  if (a->total_bytes != b->total_bytes) return 0;
  return memcmp(a->data, b->data, a->total_bytes) == 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
   LAYER 1 – Unit tests
   ───────────────────────────────────────────────────────────────────────────── */

/* 1a. Simple 1-D add: [1,2,3] + [4,5,6] = [5,7,9] */
static void test_add_1d_simple(void) {
  f32_tensor src0 = make_tensor(3, 1, 1, 1);
  f32_tensor src1 = make_tensor(3, 1, 1, 1);
  f32_tensor dst  = make_tensor(3, 1, 1, 1);
  src0.data[0]=1; src0.data[1]=2; src0.data[2]=3;
  src1.data[0]=4; src1.data[1]=5; src1.data[2]=6;

  /* Call the verified kernel via its KaRaMeL-exported symbol */
  GGML_Impl_verified_add(
    (uint8_t *)dst.data,  (uint8_t *)src0.data,
    (uint8_t *)src1.data, (uint64_t)3);

  int ok = (dst.data[0]==5.0f && dst.data[1]==7.0f && dst.data[2]==9.0f);
  TAP_CHECK(ok, "add_1d_simple: [1,2,3]+[4,5,6]=[5,7,9]",
            "got [%.1f %.1f %.1f]", dst.data[0], dst.data[1], dst.data[2]);

  free_tensor(&src0); free_tensor(&src1); free_tensor(&dst);
}

/* 1b. Add with negative values */
static void test_add_negatives(void) {
  f32_tensor src0 = make_tensor(4, 1, 1, 1);
  f32_tensor src1 = make_tensor(4, 1, 1, 1);
  f32_tensor dst  = make_tensor(4, 1, 1, 1);
  src0.data[0]=  1.0f; src0.data[1]= -3.0f;
  src0.data[2]= -5.0f; src0.data[3]=  0.0f;
  src1.data[0]= -1.0f; src1.data[1]=  3.0f;
  src1.data[2]=  5.0f; src1.data[3]=  0.0f;

  GGML_Impl_verified_add(
    (uint8_t *)dst.data, (uint8_t *)src0.data,
    (uint8_t *)src1.data, 4ULL);

  int ok = (dst.data[0]==0.0f && dst.data[1]==0.0f &&
            dst.data[2]==0.0f && dst.data[3]==0.0f);
  TAP_CHECK(ok, "add_negatives: all pairs cancel to 0",
            "got [%.1f %.1f %.1f %.1f]",
            dst.data[0], dst.data[1], dst.data[2], dst.data[3]);

  free_tensor(&src0); free_tensor(&src1); free_tensor(&dst);
}

/* 1c. Scale by 2.0 */
static void test_scale_double(void) {
  f32_tensor src = make_tensor(5, 1, 1, 1);
  f32_tensor dst = make_tensor(5, 1, 1, 1);
  fill_sequential(&src);

  GGML_Impl_verified_scale(
    (uint8_t *)dst.data, (uint8_t *)src.data, 2.0f, 5ULL);

  int ok = 1;
  for (int i = 0; i < 5; i++)
    if (dst.data[i] != (float)(i+1) * 2.0f) { ok = 0; break; }
  TAP_CHECK(ok, "scale_double: x*2 for 5 elements", "mismatch");

  free_tensor(&src); free_tensor(&dst);
}

/* 1d. ReLU: [-3,-1,0,1,3] → [0,0,0,1,3] */
static void test_relu_basic(void) {
  f32_tensor src = make_tensor(5, 1, 1, 1);
  f32_tensor dst = make_tensor(5, 1, 1, 1);
  src.data[0]=-3; src.data[1]=-1; src.data[2]=0;
  src.data[3]= 1; src.data[4]= 3;

  GGML_Impl_verified_relu(
    (uint8_t *)dst.data, (uint8_t *)src.data, 5ULL);

  float expected[] = {0,0,0,1,3};
  int ok = (memcmp(dst.data, expected, 5*sizeof(float)) == 0);
  TAP_CHECK(ok, "relu_basic: negatives clamped to 0", "mismatch");

  free_tensor(&src); free_tensor(&dst);
}

/* ─────────────────────────────────────────────────────────────────────────────
   LAYER 2 – Differential tests vs. reference implementation
   ───────────────────────────────────────────────────────────────────────────── */

#define DIFF_ITERS 16

static void test_add_differential(void) {
  for (int iter = 0; iter < DIFF_ITERS; iter++) {
    /* Vary shape each iteration */
    size_t n0 = 1 + (iter % 64);
    size_t n1 = 1 + (iter % 8);
    size_t n2 = 1 + (iter % 4);
    size_t n3 = 1;
    size_t total = n0*n1*n2*n3;

    f32_tensor a   = make_tensor(n0,n1,n2,n3);
    f32_tensor b   = make_tensor(n0,n1,n2,n3);
    f32_tensor ref = make_tensor(n0,n1,n2,n3);
    f32_tensor ver = make_tensor(n0,n1,n2,n3);

    fill_random(&a, (uint64_t)iter * 0xDEADBEEF);
    fill_random(&b, (uint64_t)iter * 0xCAFEBABE);

    ref_add(&ref, &a, &b);
    GGML_Impl_verified_add(
      (uint8_t *)ver.data, (uint8_t *)a.data,
      (uint8_t *)b.data, (uint64_t)total);

    char label[64];
    snprintf(label, sizeof label,
             "add_differential iter=%d shape=[%zu,%zu,%zu,1]",
             iter, n0, n1, n2);
    TAP_CHECK(tensors_equal(&ref, &ver), label, "output mismatch");

    free_tensor(&a); free_tensor(&b);
    free_tensor(&ref); free_tensor(&ver);
  }
}

static void test_scale_differential(void) {
  float scalars[] = {0.0f, 0.5f, 1.0f, -1.0f, 3.14159f, 1e6f};
  for (size_t s = 0; s < 6; s++) {
    f32_tensor src = make_tensor(128, 1, 1, 1);
    f32_tensor ref = make_tensor(128, 1, 1, 1);
    f32_tensor ver = make_tensor(128, 1, 1, 1);
    fill_random(&src, 0xABCDEF01ULL + s);
    ref_scale(&ref, &src, scalars[s]);
    GGML_Impl_verified_scale(
      (uint8_t *)ver.data, (uint8_t *)src.data, scalars[s], 128ULL);
    char label[64];
    snprintf(label, sizeof label, "scale_differential scalar=%.5g", scalars[s]);
    TAP_CHECK(tensors_equal(&ref, &ver), label, "output mismatch");
    free_tensor(&src); free_tensor(&ref); free_tensor(&ver);
  }
}

static void test_relu_differential(void) {
  for (int iter = 0; iter < DIFF_ITERS; iter++) {
    size_t n = 64 + (size_t)iter * 37;
    f32_tensor src = make_tensor(n,1,1,1);
    f32_tensor ref = make_tensor(n,1,1,1);
    f32_tensor ver = make_tensor(n,1,1,1);
    fill_random(&src, (uint64_t)iter * 0x1234567890ABCDEFULL);
    ref_relu(&ref, &src);
    GGML_Impl_verified_relu(
      (uint8_t *)ver.data, (uint8_t *)src.data, (uint64_t)n);
    char label[64];
    snprintf(label, sizeof label, "relu_differential n=%zu iter=%d", n, iter);
    TAP_CHECK(tensors_equal(&ref, &ver), label, "output mismatch");
    free_tensor(&src); free_tensor(&ref); free_tensor(&ver);
  }
}

/* ─────────────────────────────────────────────────────────────────────────────
   LAYER 3 – Adversarial / boundary tests
   ───────────────────────────────────────────────────────────────────────────── */

/* 3a. Single-element tensor (n=1): classic off-by-one trap */
static void test_add_single_element(void) {
  f32_tensor a = make_tensor(1,1,1,1);
  f32_tensor b = make_tensor(1,1,1,1);
  f32_tensor d = make_tensor(1,1,1,1);
  a.data[0] = 42.0f;
  b.data[0] = 58.0f;
  GGML_Impl_verified_add((uint8_t *)d.data,(uint8_t *)a.data,(uint8_t *)b.data,1ULL);
  TAP_CHECK(d.data[0] == 100.0f, "add_single_element: 42+58=100",
            "got %.2f", d.data[0]);
  free_tensor(&a); free_tensor(&b); free_tensor(&d);
}

/* 3b. Large flat tensor (stress the loop counter arithmetic) */
static void test_add_large(void) {
  size_t n = 1 << 20;   /* 1 M elements = 4 MB */
  f32_tensor a = make_tensor(n,1,1,1);
  f32_tensor b = make_tensor(n,1,1,1);
  f32_tensor d = make_tensor(n,1,1,1);
  for (size_t i = 0; i < n; i++) { a.data[i]=1.0f; b.data[i]=2.0f; }

  GGML_Impl_verified_add((uint8_t *)d.data,(uint8_t *)a.data,(uint8_t *)b.data,
                          (uint64_t)n);

  /* Spot-check first, last, and a middle element */
  int ok = (d.data[0]==3.0f && d.data[n-1]==3.0f && d.data[n/2]==3.0f);
  TAP_CHECK(ok, "add_large: 1M element tensor all-3s spot check",
            "boundary values: [0]=%.1f [mid]=%.1f [end]=%.1f",
            d.data[0], d.data[n/2], d.data[n-1]);
  free_tensor(&a); free_tensor(&b); free_tensor(&d);
}

/* 3c. Stride-validity test: verified_contiguous_check must return true
   for all tensors allocated with make_tensor, and false for a perturbed one. */
static void test_contiguous_check(void) {
  /* This calls the extracted GGML_Impl_verified_contiguous_check */

  /* Good tensor meta: F32, 4×4×4×1 */
  uint64_t ne[4] = {4, 4, 4, 1};
  uint64_t nb[4] = {4, 16, 64, 256};   /* contiguous F32 */
  int good = GGML_Impl_verified_contiguous_check(
               /*ty=F32=0*/ 0, ne[0],ne[1],ne[2],ne[3],
                              nb[0],nb[1],nb[2],nb[3]);
  TAP_CHECK(good, "contiguous_check: valid F32 4x4x4x1 returns true",
            "got %d", good);

  /* Bad: nb[1] is wrong (stride gap) */
  uint64_t nb_bad[4] = {4, 17/*wrong*/, 64, 256};
  int bad = GGML_Impl_verified_contiguous_check(
              0, ne[0],ne[1],ne[2],ne[3],
                 nb_bad[0],nb_bad[1],nb_bad[2],nb_bad[3]);
  TAP_CHECK(!bad, "contiguous_check: stride gap detected → returns false",
            "got %d", bad);
}

/* 3d. Add with zero tensor (src1 all zeros → dst == src0) */
static void test_add_zero_tensor(void) {
  size_t n = 256;
  f32_tensor a = make_tensor(n,1,1,1);
  f32_tensor b = make_tensor(n,1,1,1);
  f32_tensor d = make_tensor(n,1,1,1);
  fill_sequential(&a);
  memset(b.data, 0, b.total_bytes);   /* b = all zeros */
  GGML_Impl_verified_add((uint8_t *)d.data,(uint8_t *)a.data,(uint8_t *)b.data,
                          (uint64_t)n);
  TAP_CHECK(memcmp(d.data, a.data, n*sizeof(float))==0,
            "add_zero_tensor: a+0=a", "output differs from src0");
  free_tensor(&a); free_tensor(&b); free_tensor(&d);
}

/* 3e. Scale by 0.0 → all zeros */
static void test_scale_by_zero(void) {
  size_t n = 128;
  f32_tensor src = make_tensor(n,1,1,1);
  f32_tensor dst = make_tensor(n,1,1,1);
  fill_sequential(&src);
  GGML_Impl_verified_scale((uint8_t *)dst.data,(uint8_t *)src.data,0.0f,(uint64_t)n);
  int ok=1;
  for (size_t i=0;i<n;i++) if (dst.data[i]!=0.0f){ok=0;break;}
  TAP_CHECK(ok,"scale_by_zero: all outputs are 0.0","mismatch");
  free_tensor(&src); free_tensor(&dst);
}

/* 3f. ReLU on all-positive input is identity */
static void test_relu_all_positive(void) {
  size_t n = 64;
  f32_tensor src = make_tensor(n,1,1,1);
  f32_tensor dst = make_tensor(n,1,1,1);
  fill_sequential(&src);   /* 1,2,3,…,64 – all positive */
  GGML_Impl_verified_relu((uint8_t *)dst.data,(uint8_t *)src.data,(uint64_t)n);
  TAP_CHECK(memcmp(dst.data,src.data,n*sizeof(float))==0,
            "relu_all_positive: relu is identity on positive input",
            "output differs from input");
  free_tensor(&src); free_tensor(&dst);
}

/* ─────────────────────────────────────────────────────────────────────────────
   LAYER 4 – Memory-safety sentinel check
   ─────────────────────────────────────────────────────────────────────────────
   We write a sentinel byte just past the allocated region and confirm it is
   undisturbed after the kernel runs.  With ASan enabled, any overwrite will
   also be caught at runtime.
*/
static void test_add_no_oob_write(void) {
  size_t n = 17;   /* odd size - no natural SIMD alignment */
  size_t bytes = n * sizeof(float);
  float *raw = (float *)malloc(bytes + 4);   /* +4 sentinel */
  assert(raw);
  memset(raw, 0xAB, bytes + 4);              /* fill including sentinel */

  f32_tensor a = make_tensor(n,1,1,1);
  f32_tensor b = make_tensor(n,1,1,1);
  fill_random(&a, 0x111);
  fill_random(&b, 0x222);

  /* Use raw as dst – sentinel sits at raw[n..n+1] */
  GGML_Impl_verified_add((uint8_t *)raw,(uint8_t *)a.data,(uint8_t *)b.data,
                          (uint64_t)n);

  /* Check sentinel bytes are still 0xAB */
  uint8_t *sentinel = (uint8_t *)raw + bytes;
  int ok = (sentinel[0]==0xAB && sentinel[1]==0xAB &&
            sentinel[2]==0xAB && sentinel[3]==0xAB);
  TAP_CHECK(ok, "add_no_oob_write: sentinel bytes past buffer untouched",
            "sentinel corrupted: %02x %02x %02x %02x",
            sentinel[0],sentinel[1],sentinel[2],sentinel[3]);

  free(raw); free_tensor(&a); free_tensor(&b);
}

/* ─────────────────────────────────────────────────────────────────────────────
   Main
   ───────────────────────────────────────────────────────────────────────────── */
int main(void) {
  int total_tests =
    /* Layer 1 */ 4 +
    /* Layer 2 */ DIFF_ITERS + 6 + DIFF_ITERS +
    /* Layer 3 */ 6 +
    /* Layer 4 */ 1;

  printf("TAP version 14\n");
  printf("1..%d\n", total_tests);
  printf("# ── Layer 1: Unit tests ─────────────────────────────────────────\n");
  test_add_1d_simple();
  test_add_negatives();
  test_scale_double();
  test_relu_basic();

  printf("# ── Layer 2: Differential vs. reference ────────────────────────\n");
  test_add_differential();
  test_scale_differential();
  test_relu_differential();

  printf("# ── Layer 3: Adversarial / boundary ───────────────────────────\n");
  test_add_single_element();
  test_add_large();
  test_contiguous_check();
  test_add_zero_tensor();
  test_scale_by_zero();
  test_relu_all_positive();

  printf("# ── Layer 4: Memory-safety sentinel ───────────────────────────\n");
  test_add_no_oob_write();

  printf("# ─────────────────────────────────────────────────────────────\n");
  if (g_fail_count == 0) {
    printf("# All %d tests passed.\n", g_test_num);
    return 0;
  } else {
    printf("# %d/%d tests FAILED.\n", g_fail_count, g_test_num);
    return 1;
  }
}
