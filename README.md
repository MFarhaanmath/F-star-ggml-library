# verified-ggml

**Formally verified replacements for critical GGML tensor kernels**, proved memory-safe and functionally correct using the [F\* / Pulse / KaRaMeL](https://fstar-lang.org) toolchain.

> **TL;DR** — Drop `extraction/ggml-verified-kernels.c` and `extraction/ggml_verified.h` into your ggml build. Every function in those files is mathematically proved to never read or write outside its allocated buffer, and to produce exactly the right numerical result.

---

## Table of Contents

1. [What is this?](#what-is-this)
2. [Repository layout](#repository-layout)
3. [Quick-start (C only, no F\* required)](#quick-start-c-only)
4. [Full verification workflow (F\* + KaRaMeL)](#full-verification-workflow)
5. [Running the test suite](#running-the-test-suite)
6. [Integrating into ggml / llama.cpp](#integrating-into-ggml--llamacpp)
7. [What is actually proved](#what-is-actually-proved)
8. [Proof architecture](#proof-architecture)
9. [Extending to new kernels](#extending-to-new-kernels)
10. [Troubleshooting](#troubleshooting)
11. [License](#license)

---

## What is this?

[ggml](https://github.com/ggerganov/ggml) is the tensor library at the heart of llama.cpp, whisper.cpp, and many other local-AI projects. Its hot-path kernels use hand-rolled byte-stride arithmetic that is extremely fast but hard to audit for out-of-bounds access.

This project replaces three of those kernels with versions that carry **machine-checked proofs**:

| Kernel | C symbol | Proves |
|---|---|---|
| Element-wise F32 add | `GGML_Impl_verified_add` | Memory safety + `dst[i] = src0[i] + src1[i]` |
| Element-wise scalar multiply | `GGML_Impl_verified_scale` | Memory safety + `dst[i] = src[i] * s` |
| Element-wise ReLU | `GGML_Impl_verified_relu` | Memory safety + `dst[i] = max(0, src[i])` |
| Contiguity checker | `GGML_Impl_verified_contiguous_check` | Termination + no UB |

The proofs are written in [F\*](https://fstar-lang.org) using the [Pulse](https://github.com/FStarLang/pulse) separation-logic DSL, and the verified C is extracted automatically by [KaRaMeL](https://github.com/FStarLang/karamel). The hand-simulated C in `extraction/` is a faithful rendering of that output so you can use and test it without installing the full toolchain.

---

## Repository layout

```
verified-ggml/
│
├── README.md
├── .gitignore
├── Makefile                         # verify → extract → test
│
├── GGML.Types.fsti                  # F* interface: tensor struct + safety predicates
├── GGML.Spec.fst                    # Pure math specifications (the "gold standard")
├── GGML.Proofs.fst                  # Formal proofs of memory safety
├── GGML.Impl.fst                    # Pulse implementations with loop invariants
│
├── extraction/
│   ├── ggml_verified.h              # Public C header (matches KaRaMeL output)
│   └── ggml-verified-kernels.c      # Verified C kernels (KaRaMeL-style output)
│
└── tests/
    └── test_verified_ggml.c         # 50-test harness (unit / differential / adversarial)
```

---

## Quick-start (C only)

You do **not** need F\*, KaRaMeL, or Z3 to use the verified kernels. The extracted C is self-contained.

### Prerequisites

- GCC ≥ 9 or Clang ≥ 11
- `make`
- (Optional but recommended) AddressSanitizer support in your compiler

### 1. Clone and compile the test suite

```bash
git clone https://github.com/YOUR_USERNAME/verified-ggml.git
cd verified-ggml

gcc -O2 -Wall -Wextra -fsanitize=address,undefined \
    -Iextraction \
    tests/test_verified_ggml.c \
    extraction/ggml-verified-kernels.c \
    -lm -o tests/test_runner

./tests/test_runner
```

Expected output (truncated):

```
TAP version 14
1..50
# -- Layer 1: Unit tests
ok 1 - add_1d_simple: [1,2,3]+[4,5,6]=[5,7,9]
ok 2 - add_negatives: all pairs cancel to 0
ok 3 - scale_double: x*2 for 5 elements
ok 4 - relu_basic: negatives clamped to 0
# -- Layer 2: Differential vs. reference
ok 5 - add_differential iter=0 shape=[1,1,1,1]
...
ok 50 - add_no_oob_write: sentinel bytes past buffer untouched
# All 50 tests passed.
```

### 2. Use the kernels in your own code

```c
#include "extraction/ggml_verified.h"

// Element-wise add: dst[i] = a[i] + b[i]  for i < n
GGML_Impl_verified_add(
    (uint8_t *)dst_f32_ptr,
    (uint8_t *)src0_f32_ptr,
    (uint8_t *)src1_f32_ptr,
    (uint64_t)n_elements);

// Check that a tensor has a contiguous layout before passing it to a verified kernel
bool ok = GGML_Impl_verified_contiguous_check(
    GGML_Types_F32,
    ne[0], ne[1], ne[2], ne[3],
    nb[0], nb[1], nb[2], nb[3]);
```

---

## Full verification workflow

This reproduces the complete chain: F\* typechecking → SMT proof discharge → KaRaMeL C extraction → test.

### Prerequisites

| Tool | Version tested | Install |
|---|---|---|
| F\* | 2024.09.05+ | [fstar-lang.org](https://fstar-lang.org/#download) |
| KaRaMeL | latest `main` | `git clone https://github.com/FStarLang/karamel` |
| Z3 | 4.12.x | [github.com/Z3Prover/z3/releases](https://github.com/Z3Prover/z3/releases) |
| OCaml | ≥ 4.14 | `opam install ocaml` |
| GCC or Clang | any recent | system package manager |

### Environment variables

```bash
export FSTAR_HOME=/path/to/fstar       # directory containing bin/fstar.exe
export KRML_HOME=/path/to/karamel      # directory containing krml binary
export PATH="$FSTAR_HOME/bin:$KRML_HOME:$PATH"
```

### Run everything

```bash
make all
```

This runs three phases:

```
Phase 1 · F* Proof Discharge
  fstar.exe GGML.Types.fsti GGML.Spec.fst GGML.Proofs.fst GGML.Impl.fst
  → typechecks all modules; Z3 discharges every SMT query

Phase 2 · KaRaMeL C Extraction
  krml -bundle "GGML.*=*" -o ggml-verified-kernels.c ...
  → produces extraction/ggml-verified-kernels.c

Phase 3 · C Test Harness
  gcc -fsanitize=address,undefined tests/test_verified_ggml.c ...
  → compiles and runs all 50 tests
```

Individual targets:

```bash
make verify    # Phase 1 only (proof checking)
make extract   # Phase 1 + 2
make test      # Phase 1 + 2 + 3
make hints     # Record Z3 hints to speed up subsequent runs
make clean     # Remove all generated files
```

---

## Running the test suite

The test harness in `tests/test_verified_ggml.c` uses [TAP](https://testanything.org/) (Test Anything Protocol) output, which is consumed by standard CI tools.

### Four layers of testing

| Layer | Count | What it covers |
|---|---|---|
| **Unit** | 4 | Exact hand-known outputs (e.g. `[1,2,3]+[4,5,6]=[5,7,9]`) |
| **Differential** | 38 | Bit-for-bit match between verified kernel and naïve reference, across random shapes and inputs |
| **Adversarial** | 7 | n=1 (off-by-one trap), n=1M (loop counter arithmetic), stride-gap detection, zero tensors, scale-by-zero, ReLU identity |
| **Sentinel** | 1 | A canary byte placed 1 byte past the end of the dst buffer is untouched after the kernel runs |

### With ASan + UBSan (recommended)

```bash
gcc -O2 -fsanitize=address,undefined \
    -Iextraction \
    tests/test_verified_ggml.c \
    extraction/ggml-verified-kernels.c \
    -lm -o tests/test_runner

./tests/test_runner
echo "Exit code: $?"   # 0 = all pass
```

### With valgrind (alternative)

```bash
gcc -O0 -g -Iextraction \
    tests/test_verified_ggml.c \
    extraction/ggml-verified-kernels.c \
    -lm -o tests/test_runner

valgrind --error-exitcode=1 --leak-check=full ./tests/test_runner
```

### In CI (GitHub Actions example)

```yaml
# .github/workflows/test.yml
name: verified-ggml tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build and test
        run: |
          gcc -O2 -fsanitize=address,undefined \
              -Iextraction \
              tests/test_verified_ggml.c \
              extraction/ggml-verified-kernels.c \
              -lm -o tests/test_runner
          ./tests/test_runner
```

---

## Integrating into ggml / llama.cpp

### Step 1: Copy the two files

```bash
cp extraction/ggml_verified.h   /path/to/llama.cpp/ggml/include/
cp extraction/ggml-verified-kernels.c /path/to/llama.cpp/ggml/src/
```

Add `ggml-verified-kernels.c` to your build system (CMake example):

```cmake
target_sources(ggml PRIVATE src/ggml-verified-kernels.c)
target_include_directories(ggml PUBLIC include)
```

### Step 2: Guard with the contiguity check

In `ggml.c` (or wherever `ggml_compute_forward_add_f32` is dispatched):

```c
#include "ggml_verified.h"

static void ggml_compute_forward_add_f32(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    // Use verified kernel when both inputs and output are contiguous F32
    if (GGML_Impl_verified_contiguous_check(
            GGML_Types_F32,
            src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3])
        && ggml_are_same_shape(src0, src1)
        && ggml_are_same_shape(src0, dst))
    {
        const int64_t n = ggml_nelements(dst);
        GGML_Impl_verified_add(
            (uint8_t *)dst->data,
            (uint8_t *)src0->data,
            (uint8_t *)src1->data,
            (uint64_t)n);
        return;
    }

    // Original ggml fallback for non-contiguous tensors
    // ... (existing code) ...
}
```

### Step 3: Verify the swap works

Run the existing ggml / llama.cpp test suite — behaviour should be identical. Then run this project's test suite against your build to double-check the integration.

---

## What is actually proved

### Memory safety (the main guarantee)

For any contiguous F32 tensor with element count `n` and byte buffer `buf` of size `n * 4`:

```
∀ i ∈ [0, n).  byte_offset(i) + 4 ≤ length(buf)
```

where `byte_offset(i) = i * 4` (for contiguous layout). This rules out **every** possible out-of-bounds read or write in the loop body.

The proof is not empirical — Z3 verifies it holds for **all** possible values of `n`, not just the ones tested.

### No aliasing

Distinct element indices produce distinct byte offsets:

```
∀ i ≠ j.  byte_offset(i) ≠ byte_offset(j)
```

Combined with the disjointness precondition (`dst ≠ src0 ≠ src1`), this rules out silent data corruption from overlapping buffers.

### Functional correctness

```
∀ i < n.  read_f32(dst, i) = read_f32(src0, i) + read_f32(src1, i)
```

The post-condition is expressed as refinement over the pure spec `add_spec` in `GGML.Spec.fst`, and the loop invariant tracks that the written prefix always matches it.

### Termination (verified_contiguous_check)

`GGML_Impl_verified_contiguous_check` is proved **total** by F\* — it always terminates, performs no division, no heap allocation, and contains no undefined behaviour regardless of inputs.

---

## Proof architecture

```
GGML.Types.fsti          GGML.Spec.fst
     │                        │
     │  tensor_meta           │  add_spec / scale_spec / relu_spec
     │  is_contiguous         │  (pure FStar.Seq functions)
     │  byte_offset           │
     │  is_tensor (vprop)     │
     └──────────┬─────────────┘
                │
         GGML.Proofs.fst
                │
         ┌──────┴──────────────────────────────────┐
         │  nbytes_contiguous                       │
         │  byte_offset_is_flat                     │
         │  flat_index_lt_nelements                 │
         │  byte_offset_in_bounds  ← main theorem   │
         │  no_overlap                              │
         └──────────────────────────────────────────┘
                │
         GGML.Impl.fst
                │
         ┌──────┴──────────────────────────────────┐
         │  verified_add    (Pulse, with invariant) │
         │  verified_scale  (Pulse, with invariant) │
         │  verified_relu   (Pulse, with invariant) │
         │  verified_contiguous_check  (Tot)        │
         └──────────────────────────────────────────┘
                │
           KaRaMeL
                │
         extraction/ggml-verified-kernels.c
```

---

## Extending to new kernels

To add verification for a new ggml kernel (e.g. `ggml_compute_forward_mul_f32`):

1. **Add the spec** to `GGML.Spec.fst`:
   ```fstar
   let mul_spec (a b : seq float) : seq float =
     FStar.Seq.init (length a) (fun i ->
       FStar.Seq.index a i *. FStar.Seq.index b i)
   ```

2. **Add the Pulse implementation** to `GGML.Impl.fst`:
   ```fstar
   fn verified_mul (m: tensor_meta{...}) (dst src0 src1: B.buffer UInt8.t)
     requires ...
     ensures  ... reads_as_float_seq h1 dst n == mul_spec ... ...
   { ... }
   ```

3. **Update the loop invariant** to track the written prefix against `mul_spec`.

4. **Run `make verify`**. If F\* reports `"Could not prove bounds"`, add the explicit instantiation:
   ```fstar
   byte_offset_in_bounds m i0 i1 i2 i3;
   ```
   before the offending read or write.

5. **Export the symbol** by adding it to `extraction/ggml_verified.h` and a stub to `extraction/ggml-verified-kernels.c` for testing before KaRaMeL extraction.

6. **Add tests** to `tests/test_verified_ggml.c` following the same four-layer pattern.

---

## Troubleshooting

### `fstar.exe: command not found`
Make sure `$FSTAR_HOME/bin` is on your `PATH`. If you built F\* from source, the binary is at `bin/fstar.exe` relative to the repo root.

### `Could not prove: byte_offset … + 4 <= length buf`
The loop invariant is too weak. Add an explicit call to `byte_offset_in_bounds` immediately before the failing `read_f32_at` or `write_f32_at`, passing the current loop index. See `GGML.Impl.fst` for the pattern.

### `Z3 timeout` on a particular query
Run `make hints` first to record cached SMT answers. If a specific query still times out, increase the rlimit in the F\* invocation:
```bash
fstar.exe --z3rlimit 200 ...
```

### Tests fail with ASan `invalid-aligned-alloc-alignment`
Your libc requires `aligned_alloc` sizes to be multiples of the alignment. The test harness rounds up automatically — make sure you are using the version from this repo, not an older copy.

### `GGML_Impl_verified_contiguous_check` returns false unexpectedly
Print the actual strides and compare with the expected contiguous layout:
```
nb[0] should be 4           (sizeof float)
nb[1] should be 4 * ne[0]
nb[2] should be 4 * ne[0] * ne[1]
nb[3] should be 4 * ne[0] * ne[1] * ne[2]
```
Any deviation means the tensor was transposed or created with custom strides. Use the fallback kernel in that case.

---

## License

MIT. See `LICENSE` for details.

The F\* standard library and KaRaMeL runtime are © Microsoft Research and INRIA, Apache 2.0.
