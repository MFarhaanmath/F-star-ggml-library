# ==============================================================================
# Makefile – verified-ggml
# Drives F* typechecking, proof discharge, KaRaMeL C extraction,
# and the C-level test harness.
#
# Prerequisites (must be on PATH or set via env vars):
#   FSTAR_HOME   – root of the F* installation
#   KRML_HOME    – root of the KaRaMeL installation
#   Z3           – Z3 SMT solver (4.12.x recommended)
#   CC           – C compiler (clang or gcc)
#
# Usage:
#   make verify        – typecheck + discharge all proofs
#   make extract       – produce verified C files via KaRaMeL
#   make test          – compile and run the C test harness
#   make all           – verify + extract + test
#   make clean
# ==============================================================================

FSTAR    ?= $(FSTAR_HOME)/bin/fstar.exe
KRML     ?= $(KRML_HOME)/krml
CC       ?= clang
CFLAGS   ?= -O2 -Wall -Wextra -fsanitize=address,undefined

# ── Source files ───────────────────────────────────────────────────────────────
FST_SOURCES = \
  GGML.Types.fsti   \
  GGML.Spec.fst     \
  GGML.Proofs.fst   \
  GGML.Impl.fst

# ── F* flags ──────────────────────────────────────────────────────────────────
#   --use_hints            reuse cached Z3 proofs where possible
#   --record_hints         record new hints on first run
#   --query_stats          print per-query Z3 statistics (useful for debugging)
#   --admit_smt_queries    ONLY for iteration – remove before final check
FSTAR_OPTS = \
  --include $(FSTAR_HOME)/ulib \
  --include $(KRML_HOME)/krmllib \
  --use_hints \
  --record_hints \
  --query_stats \
  --warn_error +241   # treat incomplete patterns as errors

# ── KaRaMeL flags ─────────────────────────────────────────────────────────────
KRML_OPTS = \
  -bundle "GGML.Types+GGML.Spec+GGML.Proofs+GGML.Impl=*" \
  -minimal \
  -add-include '"ggml_verified.h"' \
  -o ggml-verified-kernels.c \
  -tmpdir extraction/

# ==============================================================================
# Top-level targets
# ==============================================================================

.PHONY: all verify extract test clean hints

all: verify extract test

# ── 1. Verify (typecheck + SMT proofs) ────────────────────────────────────────
verify: $(FST_SOURCES)
	@echo "═══════════════════════════════════════════"
	@echo " Phase 1 · F* Proof Discharge"
	@echo "═══════════════════════════════════════════"
	$(FSTAR) $(FSTAR_OPTS) $(FST_SOURCES)
	@echo "✓ All proofs verified."

# ── 2. Record SMT hints (run once, commit .hints files) ───────────────────────
hints: $(FST_SOURCES)
	$(FSTAR) $(FSTAR_OPTS) --record_hints $(FST_SOURCES)

# ── 3. Extract to C via KaRaMeL ───────────────────────────────────────────────
extract: verify
	@echo "═══════════════════════════════════════════"
	@echo " Phase 2 · KaRaMeL C Extraction"
	@echo "═══════════════════════════════════════════"
	mkdir -p extraction
	$(KRML) $(KRML_OPTS) $(FST_SOURCES)
	@echo "✓ Extracted to extraction/ggml-verified-kernels.c"

# ── 4. Compile + run C test harness ───────────────────────────────────────────
test: extract tests/test_verified_ggml.c
	@echo "═══════════════════════════════════════════"
	@echo " Phase 3 · C Test Harness"
	@echo "═══════════════════════════════════════════"
	$(CC) $(CFLAGS) \
	  -I$(KRML_HOME)/krmllib/dist/minimal \
	  -Iextraction \
	  tests/test_verified_ggml.c \
	  extraction/ggml-verified-kernels.c \
	  -lm \
	  -o tests/test_runner
	./tests/test_runner
	@echo "✓ All C tests passed."

clean:
	rm -f *.hints
	rm -rf extraction/
	rm -f tests/test_runner
	rm -f *.checked

# ==============================================================================
# Per-module dependency graph (for parallel -j builds)
# ==============================================================================
GGML.Types.fsti.checked:  GGML.Types.fsti
	$(FSTAR) $(FSTAR_OPTS) --cache_dir . $<

GGML.Spec.fst.checked:    GGML.Spec.fst GGML.Types.fsti.checked
	$(FSTAR) $(FSTAR_OPTS) --cache_dir . $<

GGML.Proofs.fst.checked:  GGML.Proofs.fst GGML.Types.fsti.checked
	$(FSTAR) $(FSTAR_OPTS) --cache_dir . $<

GGML.Impl.fst.checked:    GGML.Impl.fst \
                           GGML.Types.fsti.checked \
                           GGML.Spec.fst.checked   \
                           GGML.Proofs.fst.checked
	$(FSTAR) $(FSTAR_OPTS) --cache_dir . $<
