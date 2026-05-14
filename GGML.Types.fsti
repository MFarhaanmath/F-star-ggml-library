(*
  GGML.Types.fsti
  ──────────────────────────────────────────────────────────────────────────────
  F* interface that mirrors the ggml_tensor struct from ggml.h.

  Key facts extracted from the real ggml source
  (ggml-org/ggml, src/ggml.c  –  from uploaded corpus):

    • GGML_MAX_DIMS = 4
    • ggml_tensor has:
        - type  : ggml_type  (enum, e.g. F32=0, F16=1, Q4_0=2, …)
        - ne[4] : int64_t    – number of elements per dimension
        - nb[4] : size_t     – byte-stride per dimension
        - data  : void *     – raw byte buffer
    • Contiguity condition (from ggml_is_contiguous_n):
        nb[0] == ggml_type_size(type)
        nb[1] == nb[0] * ne[0]      (for a standard, non-block type like F32)
        nb[2] == nb[1] * ne[1]
        nb[3] == nb[2] * ne[2]
    • Safety predicate for element (i0,i1,i2,i3):
        byte offset = i0*nb[0] + i1*nb[1] + i2*nb[2] + i3*nb[3]
        must satisfy:  offset + type_size <= total_buffer_bytes

  We model only F32 tensors in this initial verification target.
*)

module GGML.Types

open FStar.UInt64
open FStar.Int64
open LowStar.Buffer
open FStar.HyperStack.ST

(* ──────────────────────────────────────────────
   1.  Element type
   ────────────────────────────────────────────── *)

type ggml_type =
  | F32          (* type_size = 4  blck_size = 1 *)
  | F16          (* type_size = 2  blck_size = 1 *)
  | Q4_0         (* type_size = 18 blck_size = 32 *)
  | Q8_0         (* type_size = 34 blck_size = 32 *)

(* Bytes consumed by one storage block of the given type.
   Mirrors ggml_type_size() / ggml_blck_size() in ggml.c *)
val type_size : ggml_type -> Tot nat
let type_size = function
  | F32  -> 4
  | F16  -> 2
  | Q4_0 -> 18
  | Q8_0 -> 34

val blck_size : ggml_type -> Tot pos
let blck_size = function
  | F32  -> 1
  | F16  -> 1
  | Q4_0 -> 32
  | Q8_0 -> 32

(* Bytes per element (type_size / blck_size, exact for our four types) *)
val bytes_per_elem : ggml_type -> Tot nat
let bytes_per_elem t =
  type_size t / blck_size t     (* 4,2,0,1 – Q4_0 rounds down to 0 per element;
                                   handled correctly via block arithmetic later *)

(* ──────────────────────────────────────────────
   2.  Pure tensor descriptor (no heap)
   ────────────────────────────────────────────── *)

(* Mirrors the ne[4] / nb[4] arrays in ggml_tensor.
   We use nat throughout; the C code uses int64_t / size_t. *)
noeq type tensor_meta = {
  ty   : ggml_type;
  ne   : (n0:nat & n1:nat & n2:nat & n3:nat);   (* element counts *)
  nb   : (b0:nat & b1:nat & b2:nat & b3:nat);   (* byte strides   *)
}

(* Total elements *)
let nelements (m: tensor_meta) : nat =
  let (| n0, n1, n2, n3 |) = m.ne in
  n0 * n1 * n2 * n3

(* Total buffer bytes needed *)
let nbytes (m: tensor_meta) : nat =
  let (| n0, n1, n2, n3 |) = m.ne in
  let (| b0, b1, b2, b3 |) = m.nb in
  (* worst-case last element offset + one element size *)
  (n0 - 1) * b0 + (n1 - 1) * b1 + (n2 - 1) * b2 + (n3 - 1) * b3 + type_size m.ty

(* ──────────────────────────────────────────────
   3.  Well-formedness predicate  (PURE)
   ────────────────────────────────────────────── *)

(*
  A tensor_meta is well-formed when:
    (a) all ne[i] >= 1
    (b) strides are consistent with a contiguous F32 layout
        (we restrict to contiguous tensors for this initial proof target):
          nb[0] = type_size(ty)
          nb[1] = nb[0] * ne[0]
          nb[2] = nb[1] * ne[1]
          nb[3] = nb[2] * ne[2]

  The contiguity requirement matches the GGML_ASSERT(ggml_is_contiguous(…))
  guards found in ggml_compute_forward_add_f32 in the real ggml source.
*)
let is_contiguous (m: tensor_meta) : prop =
  let (| n0, n1, n2, n3 |) = m.ne in
  let (| b0, b1, b2, b3 |) = m.nb in
  n0 >= 1 /\ n1 >= 1 /\ n2 >= 1 /\ n3 >= 1
  /\ b0 = type_size m.ty
  /\ b1 = b0 * n0
  /\ b2 = b1 * n1
  /\ b3 = b2 * n2

(* ──────────────────────────────────────────────
   4.  Safety predicate: byte-offset stays in-bounds
   ────────────────────────────────────────────── *)

(*
  For every valid 4-D index (i0,i1,i2,i3), the byte offset
      off = i0*nb[0] + i1*nb[1] + i2*nb[2] + i3*nb[3]
  must satisfy:
      off + type_size <= buffer_size_bytes

  For contiguous tensors this reduces to:
      flat_index < nelements
  which we prove below as a lemma.
*)
let valid_index (m: tensor_meta) (i0 i1 i2 i3 : nat) : prop =
  let (| n0, n1, n2, n3 |) = m.ne in
  i0 < n0 /\ i1 < n1 /\ i2 < n2 /\ i3 < n3

let byte_offset (m: tensor_meta) (i0 i1 i2 i3 : nat) : nat =
  let (| _, b1, b2, b3 |) = m.nb in
  let b0 = type_size m.ty in
  i0 * b0 + i1 * b1 + i2 * b2 + i3 * b3

(* KEY SAFETY LEMMA (statement only – proved in GGML.Proofs.fst):
   If is_contiguous m and valid_index m i0 i1 i2 i3 then
     byte_offset m i0 i1 i2 i3 + type_size m.ty <= nbytes m  *)
val byte_offset_in_bounds
  (m : tensor_meta)
  (i0 i1 i2 i3 : nat)
  : Lemma
    (requires is_contiguous m /\ valid_index m i0 i1 i2 i3)
    (ensures  byte_offset m i0 i1 i2 i3 + type_size m.ty <= nbytes m)

(* ──────────────────────────────────────────────
   5.  Low* heap predicate
   ────────────────────────────────────────────── *)

(*
  is_tensor h meta buf
    – buf points to a live byte buffer of exactly (nbytes meta) bytes
    – meta is contiguous
*)
val is_tensor
  (h   : HS.mem)
  (meta: tensor_meta)
  (buf : B.buffer UInt8.t)
  : prop
