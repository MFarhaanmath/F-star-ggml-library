(*
  GGML.Spec.fst
  ──────────────────────────────────────────────────────────────────────────────
  Pure "gold-standard" specifications for tensor operations.
  These are the mathematical definitions that the Pulse implementations must
  refine.  They operate on immutable FStar.Seq.seq float values, completely
  separate from any heap or pointer reasoning.

  Operations covered:
    1. add_spec        – element-wise addition (the primary verification target)
    2. scale_spec      – element-wise scalar multiply
    3. relu_spec       – element-wise ReLU activation
    4. contiguous_spec – helper: what a contiguous tensor view looks like as seq
*)

module GGML.Spec

open FStar.Seq
open FStar.Math.Lemmas

(* We model F32 values as FStar floats (IEEE 754 doubles in F*, close enough
   for our memory-safety goals; IEEE precision proofs are left for future work) *)

(* ──────────────────────────────────────────────
   1. Flat float sequence helpers
   ────────────────────────────────────────────── *)

(* Well-typed element count: every nat *)
let seq_wf (#a:Type) (s: seq a) (len:nat) : prop =
  length s = len

(* ──────────────────────────────────────────────
   2. add_spec
   ──────────────────────────────────────────────
   Matches ggml_vec_add_f32 / ggml_compute_forward_add_f32
   for the contiguous, same-shape case:
     dst[i] = src0[i] + src1[i]   for all i < n

   If lengths differ we return src0 unchanged (safety fallback).
*)
let add_spec (src0 src1 : seq float) : seq float =
  if length src0 <> length src1 then src0
  else
    FStar.Seq.init (length src0) (fun i ->
      FStar.Seq.index src0 i +. FStar.Seq.index src1 i)

(* ──────────────────────────────────────────────
   3. scale_spec
   ──────────────────────────────────────────────
   Matches ggml_compute_forward_scale_f32:
     dst[i] = src[i] * scalar
*)
let scale_spec (src : seq float) (scalar : float) : seq float =
  FStar.Seq.init (length src) (fun i ->
    FStar.Seq.index src i *. scalar)

(* ──────────────────────────────────────────────
   4. relu_spec
   ──────────────────────────────────────────────
   Matches GGML_UNARY_OP_RELU:
     dst[i] = max(0.0, src[i])
*)
let relu_spec (src : seq float) : seq float =
  FStar.Seq.init (length src) (fun i ->
    let v = FStar.Seq.index src i in
    if v < 0.0 then 0.0 else v)

(* ──────────────────────────────────────────────
   5. Key algebraic lemmas
   ────────────────────────────────────────────── *)

(* add_spec is commutative *)
val add_spec_comm
  (a b : seq float)
  : Lemma
    (requires length a = length b)
    (ensures  add_spec a b == add_spec b a)
let add_spec_comm a b =
  assert (length (add_spec a b) = length (add_spec b a));
  FStar.Classical.forall_intro (fun i ->
    assert (FStar.Seq.index (add_spec a b) i =
            FStar.Seq.index a i +. FStar.Seq.index b i);
    assert (FStar.Seq.index (add_spec b a) i =
            FStar.Seq.index b i +. FStar.Seq.index a i))

(* Length is preserved *)
val add_spec_length
  (a b : seq float)
  : Lemma
    (requires length a = length b)
    (ensures  length (add_spec a b) = length a)
let add_spec_length a b = ()

(* scale_spec with 1.0 is identity *)
val scale_spec_one
  (s : seq float)
  : Lemma (scale_spec s 1.0 == s)
let scale_spec_one s =
  FStar.Seq.lemma_eq_intro (scale_spec s 1.0) s

(* relu_spec is idempotent: relu(relu(x)) = relu(x) *)
val relu_spec_idempotent
  (s : seq float)
  : Lemma (relu_spec (relu_spec s) == relu_spec s)
let relu_spec_idempotent s =
  let r = relu_spec s in
  FStar.Seq.lemma_eq_intro (relu_spec r) r

(* ──────────────────────────────────────────────
   6. Refinement relation
   ──────────────────────────────────────────────
   When the Pulse implementation finishes, we assert that the bytes in
   the output buffer, interpreted as a float seq, equal add_spec src0 src1.

   This is the top-level correctness statement we aim to prove.
*)

(*
  reads_as_float_seq h buf n
    – the first (n * 4) bytes of buf, interpreted as little-endian IEEE-754
      floats, equal the sequence s.
  (In the actual proof, this is defined via LowStar.Endianness.)
*)
val reads_as_float_seq
  (h   : FStar.HyperStack.mem)
  (buf : LowStar.Buffer.buffer FStar.UInt8.t)
  (n   : nat)
  : GTot (seq float)

(* Correctness statement for verified_add (see GGML.Impl.fst):

   Post-condition:
     reads_as_float_seq h1 dst n
     == add_spec (reads_as_float_seq h0 src0 n)
                 (reads_as_float_seq h0 src1 n)
*)
