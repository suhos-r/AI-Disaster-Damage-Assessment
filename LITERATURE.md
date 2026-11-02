# Literature basis and what this project does differently

This file tracks the major technical advancements in the pipeline against the
published methods they build on, and states plainly where a component is a
validated improvement versus where it is a heuristic placeholder standing in
for a tool this environment doesn't have installed. Advancements are in
implementation order; each one names the naive baseline, the literature that
motivated the change, and what the change actually buys.

---

## 1. Confidence-zone gating before pocket detection

**Files:** `plddt_segmenter.py`, `afdb_downloader.py`

**Naive baseline:** run pocket detection (fpocket, etc.) on the raw AlphaFold
model, exactly as downloaded.

**Problem with the baseline:** AlphaFold reports per-residue pLDDT, and low
values are not "less precise coordinates" — Akdel et al. (*Nat Struct Mol Biol*
2022) and the disorder-prediction literature (Ruff & Pappu, *J Mol Biol* 2021)
show sub-50 pLDDT regions correlate strongly with intrinsically disordered
regions that have no single fold at all. Running a cavity detector over a
disordered loop produces geometrically real but biologically meaningless
"pockets" — coordinates exist, but they describe a random instantaneous
snapshot of a floppy chain, not a binding site.

**What we built:** `plddt_segmenter.py` smooths the per-residue confidence
trace with a centered moving average, extracts contiguous chain-aware
high-confidence zones above a threshold (default 50), and writes a
`.high-confidence.pdb` containing only retained residues. Every downstream
step — pocket detection, selectivity scanning, the antidote pipeline's triage —
runs on this trimmed structure, never the raw model.

**Why this is better than scoring the raw structure:** it removes an entire
class of false positive before any expensive computation runs, rather than
trying to down-weight it after the fact. `benchmark_fpocket.py` (Phase 6)
quantifies the resulting reduction in spurious pocket count directly.

---

## 2. Multi-signal pocket authenticity re-scoring

**Files:** `pocket_finder.py`, `pocket_verifier.py`

**Naive baseline:** trust fpocket's own druggability score as the final
ranking signal.

**Problem with the baseline:** fpocket's score is a single alpha-sphere-based
number (Le Guilloux et al., *BMC Bioinformatics* 2009) computed from geometry
alone. It has no way to know whether the residues it clustered are backbone-
confident, and geometric druggability and structural trustworthiness are
different questions — a perfectly shaped cavity built from residues AlphaFold
was unsure about is not a pocket you should act on.

**What we built:** `pocket_verifier.py` computes four independent signals and
combines them as a weighted composite (30% pLDDT confidence, 30% spatial
enclosure/depth via radius-of-gyration compactness, 25% 3D contact network
density, 15% physicochemical balance), each following a different failure mode
than fpocket's own score. The enclosure term is a compactness-based proxy for
SiteMap's enclosure descriptor (Halgren, *J Chem Inf Model* 2009), which is
one of the stronger published discriminators between real sites and surface
noise.

**Why this is better:** it catches the specific failure mode fpocket cannot
see by construction — a geometrically plausible cavity sitting on
low-confidence backbone — and reports it as a distinct "Unverified /
False-Positive Candidate" state rather than silently ranking it alongside
trustworthy pockets.

**Where this stage still improved further:** `antidote_finder.py`'s
`triage_pocket` (item 5 below) generalizes this same idea with a cleaner,
literature-anchored enclosure calculation (raw neighbor-count buriedness
against the SiteMap definition) and adds a hard rejection gate, rather than
only a soft score.

---

## 3. Physics-informed neural network for pocket matching

**Files:** `gnn_matcher.py`, `partner_gnn.py`, `train_partner_gnn.py`

**Naive baseline:** a lookup table of hand-picked "this pocket type binds that
protein type" rules — which is literally what `existing_protein_finder.py`'s
rule-based fallback path still is, kept intentionally as a degrade path.

**Problem with the baseline:** fixed rules can't account for interacting
effects — e.g., a pocket that is both acidic *and* enclosed *and* backbone-
confident should score differently than one that is merely acidic. A pure
lookup table also can't be improved by training data.

**What we built:** a genuine 3-layer Graph Convolutional Network (Kipf &
Welling-style GCN layers with symmetric-normalized adjacency) that embeds a
pocket's residue graph, combined with an explicit biophysical scoring term
(charge complementarity, hydrophobic match, rigidity) in a
**physics-informed neural network (PINN)** blend — 95% biophysics, 5% learned
correction — so the network cannot produce a physically nonsensical ΔG even
before it is fully trained. `train_partner_gnn.py` supports supervised
training against a curated interaction index with **protein-disjoint train /
validation / test splits**, rejecting any accession that leaks across splits
— the single most common way affinity models silently overfit (homolog
leakage), and a documented failure mode in ML-for-binding benchmarking
critiques.

**Honest caveat, stated in the code itself:** `existing_protein_finder.py`
never displays a GNN score as a measured affinity — its docstring states this
directly — and the manifest gate (`model_status`) only allows a trained
checkpoint to be used for candidate *ranking* once it clears ROC-AUC ≥ 0.75,
average precision ≥ 0.70, and precision@20 ≥ 0.50 on the held-out,
protein-disjoint split. Absent a qualifying checkpoint, the model runs with
deterministic Xavier-initialized weights, meaning the neural term contributes
essentially nothing beyond the explicit biophysics term — which is the
intended, conservative fallback, not a hidden failure.

---

## 4. Evidence-gated candidate reporting (no fabricated affinities)

**Files:** `existing_protein_finder.py`, `antidote_finder.py`

**Naive baseline:** report a predicted ΔG or Kd next to a protein name as if
it were a measurement.

**Problem with the baseline:** the CAPRI blind-assessment program (Lensink et
al., *Proteins* 2019) has repeatedly shown that scoring functions rank protein
complexes poorly even *given* the correct bound pose, and Vangone & Bonvin
(*eLife* 2015) note directly that intuitive descriptors like buried surface
area do not correlate with measured affinity. Presenting a computed number as
equivalent to a wet-lab Kd is actively misleading.

**What we built:** every response carries an explicit evidence label
distinguishing "Curated functional evidence" (real database interaction) from
"GNN Candidate Match" (predicted, unmeasured), and the ranking logic in
`antidote_finder.py` goes further — evidence tier (A/B/C/D) is the *primary*
sort key, chemistry score only breaks ties within a tier (`_rank_key`
enforces this structurally, not just in the UI copy). A chemistry score can
never promote a candidate above one with real experimental support.

---

## 5. Real, live-retrieved protein candidates replacing a hardcoded pool — the major advancement

**Files:** `interaction_sources.py`, `antidote_finder.py`,
`existing_protein_finder.py` (rewritten in the follow-up pass described in
section 6)

This is the largest single change in the project, so it's documented in more
depth than the items above.

### 5a. The baseline this replaced

`existing_protein_finder.py` originally shipped a `REFERENCE_CANDIDATE_POOL`
of five hardcoded proteins (p53, EGFR, HSP90, "InhA", FKBP12) that were
returned for *every* target, reordered only by a deterministic hash of
`(target, candidate)` so different targets didn't visibly repeat. **That pool
also contained a real data error found while building the replacement:** the
entry labelled `P9WHE3 — "InhA — Fatty Acid Pathway Enzyme"` was wrong;
P9WHE3 is actually `rplL`, a ribosomal stalk protein. The true UniProt
accession for *M. tuberculosis* InhA is `P9WGR1`. This is exactly the kind of
error a fixed pool accumulates silently and a live-retrieval design cannot,
because the name and accession are re-fetched from UniProt on every call. As
of the cleanup pass in section 6, the pool and this mislabelling no longer
exist anywhere in the codebase (backend or frontend).

### 5b. What replaces it

`antidote_finder.py`'s pipeline retrieves real candidates from four public
evidence channels, ranked strictly by evidence strength:

| Tier | Channel | What it proves |
|---|---|---|
| A | RCSB co-structure transfer | The partner has been physically resolved in contact with this fold in a solved structure |
| B | UniProtKB/IntAct curated interactions | An experimentally reported binary interaction, no solved complex required |
| C | STRING functional association | Probabilistic, possibly indirect pathway co-membership |
| — | RCSB ligand survey | Small molecules crystallised in this fold, used when no protein partner exists |

**Literature basis for retrieval-over-scoring as the primary strategy:**
FINDSITE (Brylinski & Skolnick, *PNAS* 2008) and 3DLigandSite (Wass et al.,
*Nucleic Acids Research* 2010) both established that transferring binder
identity from structurally characterised templates outperforms de novo
docking specifically when the target is a *predicted model* rather than an
experimental structure — which is the AlphaFold case throughout this entire
project. This is why retrieval, not scoring, is the primary mechanism.

### 5c. Three retrieval bugs found and fixed during construction

Each of these was discovered by inspecting real API output rather than
assumed, and each one silently produces an empty or wrong result set if left
unfixed — worth recording because they are the kind of bug that looks like
"no interaction evidence exists" instead of "the code is broken":

1. **Unfiltered PDB ID truncation returns only monomers.** Human lysozyme
   (P61626) has 215 PDB cross-references; only 5 of them are multi-protein
   complexes, scattered arbitrarily through the accession-sorted list. Taking
   the first 60 IDs (a reasonable-looking cap) returned zero partners — not
   because none exist, but because none of the first 60 happened to be a
   complex. Fixed by querying RCSB's search API with a server-side
   `polymer_entity_count_protein >= 2` filter, so the *complex-containing*
   entries are the ones retrieved, regardless of where they sort
   alphabetically.

2. **MHC/TCR epitope-presentation complexes look like protein partners but
   aren't.** For TB InhA, the top "partners" before this fix were T-cell
   receptor and HLA-E chains — because InhA's 9-residue epitope (Mtb44) is
   presented in several solved HLA structures, which satisfies "the target
   appears in a multi-chain complex" without the target's actual fold or
   pocket being present in the structure at all. Fixed with
   `_target_is_folded`, which requires the target's own chain in the complex
   to be at least 40 residues and at least half its full sequence length
   before accepting any other chain in that entry as a partner.

3. **The target reappearing as its own "partner."** PDB entries frequently
   map two copies of the same protein to two different UniProt accessions
   (species variants, redundant entries). Accession comparison alone let the
   target show up as a top hit against itself. Fixed with a fast
   alignment-free identity check (5-mer Jaccard similarity ≥ 0.55) run against
   sequence, not accession.

### 5d. Admitting engineered binders (the biggest single accuracy win)

Nanobodies, designed binder proteins, and therapeutic antibody fragments
carry no UniProt accession, so an accession-keyed pipeline drops them by
construction. These are frequently the *most* actionable real candidates —
for human lysozyme, once admitted (keyed by PDB entity ID instead of
accession), the top three results became the camelid nanobodies **cAbHuL5**,
**cAbHuL6**, and **HL6**, which are documented inhibitors of human lysozyme
amyloid fibril formation (Dumoulin et al., *Nature* 2003) — i.e., the actual
published antidote to hereditary lysozyme amyloidosis, discovered
automatically rather than hand-curated.

### 5e. Handling the "no protein binder exists" case honestly

For TB InhA, the pipeline correctly finds **zero** protein partners across all
three interaction channels — this is a true negative, not a bug (verified:
InhA's real pharmacology is small-molecule, not protein-protein). Rather than
returning an empty, unhelpful result, the RCSB ligand-survey channel reports
what small molecules have actually been crystallised in that pocket, filtered
against a 40-entry crystallisation-artefact list (buffer salts, cryoprotectants,
common ions) and split into native cofactors versus drug-like binders. For
InhA this surfaces **ZID** (the isonicotinic-acyl-NADH adduct — the
actual active metabolite of isoniazid, the frontline TB drug) and
**triclosan**, both found automatically from crystal structures rather than
hardcoded.

### 5f. PRODIGY-based binding free energy, honestly scoped

`_prodigy_delta_g` implements the published PRODIGY contacts-based linear
model (Vangone & Bonvin, *eLife* 2015; Xue et al., *Bioinformatics* 2016)
with its fitted coefficients unmodified. The one adaptation: PRODIGY expects
*observed* interfacial contact counts from a docked or crystallised complex,
which doesn't exist here (no docking is performed). We instead estimate
expected contact counts from the outer product of the pocket's and the
candidate's residue-class composition vectors — an approximation, and the
code labels it as such in every response (`estimate_basis` field) rather than
presenting it as a measured value. This mirrors item 4's evidence-gating
principle applied to a specific number instead of the ranking as a whole.

### 5g. Therapeutic-mode triage (inhibit vs. stabilise)

**Literature basis:** Ringe & Petsko (*J Biol* 2009) established the
pharmacological-chaperone concept — a small molecule or protein binding a
*destabilised* site can restore folding and function rather than blocking
activity, which is the opposite therapeutic goal from a classic inhibitor.

**What we built:** `therapeutic_strategy` distinguishes three cases from
organism and pocket-confidence data already computed upstream: a non-human
target routes to "Inhibition" (occupy and block), a human target with a
weakly-ordered/poorly-enclosed site routes to "Stabilisation
(pharmacological chaperone)" — this is the literal "fix the deformed protein"
case the user asked for — and a human target with a well-ordered site routes
to "Modulation / competitive displacement." No prior stage of the pipeline
made this distinction; candidates were previously always implicitly framed as
inhibitors regardless of target biology.

---

## 6. Removing every remaining hardcoded value (bug-fix pass)

A follow-up pass triggered by an explicit "don't hardcode anything, fix all
bugs" request. Section 5 had already replaced the *primary* candidate pool,
but a full audit found the same fabrication pattern recurring in three more
places, plus real bugs introduced while building section 5's replacement.

**`existing_protein_finder.py`'s pairwise-contact generator was still
fabricating data even after the accession pool was gone in `antidote_finder.py`.**
The old `_generate_pairwise_interactions` cycled through fixed arrays of
invented partner residue numbers (`f"{partner} {148 + idx * 4}"` — an
arithmetic pattern with no relationship to any real structure) to make each
card look varied. It's replaced by `_real_pairwise_interactions`, which scans
the candidate's *actual fetched sequence* (from UniProt or the resolved PDB
entity — the same real sequence retrieval built in section 5) for a genuine
position carrying a chemically complementary residue. Every residue name and
sequence position shown is now real; only the per-contact-class energy
(e.g. −4.3 kcal/mol for a salt bridge, from Barlow & Thornton, *J Mol Biol*
1983) remains a literature constant rather than a per-protein measurement —
disclosed as such in the UI copy rather than presented as computed for that
specific pair.

**`selectivity.py`'s Foldseek-absent fallback was a keyword-matched hardcoded
table**, not a computation. It has been replaced with a live two-signal
UniProt query: Pfam domain-sharing counts among reviewed human proteins (the
literature basis for kinome/protease-family cross-reactivity screening — e.g.
Fabian et al., *Nat Biotechnol* 2005, profiled kinase inhibitors by exactly
this shared-domain logic), and UniRef cluster membership for near-identical
human sequence hits. Verified against real targets: EGFR correctly surfaces
ABL1 and ERBB2 through the shared tyrosine-kinase domain (a real, documented
kinome cross-reactivity concern); TB InhA correctly surfaces human
short-chain dehydrogenase/reductase family members through its real SDR
fold (InhA genuinely is an SDR-family enzyme — this is accurate homology, not
a coincidence); human lysozyme correctly surfaces alpha-lactalbumin, its
textbook evolutionary homolog. All of this is real UniProt data at request
time; none of it is stored in the codebase.

**`reportTemplate.js` (frontend) had its own independent hardcoded pool.**
The "Generate Report" feature carried a client-side `refDb` duplicating the
same five proteins — including the same P9WHE3/InhA mislabelling caught in
section 5a — plus a `matchBinders()` function that invented a ΔG and Kd
number from `((entryHash + pocketHash + idx * 17) % 19) / 20.0`: a hash of
the report's own entry id, presented next to a "kcal/mol" and "nM" label as
if it were a thermodynamic measurement. This fired whenever a user clicked
"Generate Report" without first running pocket discovery — a silent,
easy-to-hit path. It's removed entirely; the report now only renders binder
data actually fetched from the backend's live pipeline, states plainly when
no live evidence lookup has been run yet, and `App.jsx`'s report button now
fetches that data first (against the best-scoring pocket) instead of passing
along whatever was or wasn't already in state.

**Two real bugs surfaced while removing the fabrication, unrelated to
hardcoding but incidentally fixed in the same pass:** an ID-formatting bug in
`interaction_sources.py` where engineered-binder candidates were keyed as
`f"{pdb_id}_{entity_rcsb_id}"` even though `entity_rcsb_id` already contains
the PDB id, producing doubled identifiers like `1OP9_1OP9_1`; and a malformed
`tracker/CURRENT_TASK.json` containing an empty `{}` object followed by a
second, unenclosed block of keys sitting outside any braces — invalid JSON
that no parser could actually read.

---

## Known limitations (stated for completeness, not hidden)

- **`selectivity.py`'s live fallback is a domain-sharing and sequence-identity
  proxy, not a structural alignment.** Sharing a Pfam domain does not
  guarantee the *pocket itself* is conserved, only that a related functional
  region exists somewhere in the protein, and UniRef clustering measures
  sequence identity rather than a true structural superposition score.
  Install `foldseek` with a local human-proteome database for a direct
  structural comparison — the code already prefers that path when it detects
  both are present. A same-protein alternate UniProt accession for the target
  itself can also currently surface as a "near-identical human hit" via
  UniRef if the target is human; a full fix would apply the same real-sequence
  self-rejection check used in `interaction_sources.py` (section 5c, item 3).
- **PRODIGY's contact-count inputs here are estimated, not observed** (see
  5f) — every returned ΔG/Kd is an order-of-magnitude triage prior, never a
  measured affinity.
- **Tier C (STRING) associations can be indirect** — shared pathway
  membership, not necessarily physical contact — and the pipeline never
  admits a Tier C candidate that an accession-keyed channel didn't already
  discover, specifically to avoid treating "functionally associated" as
  equivalent to "binds."
- **No docking is performed anywhere in the pipeline.** No pose, interface,
  or orientation is computed; co-structure evidence proves contact existed in
  one crystal form, not that the interaction is physiological or occurs at
  the analysed pocket specifically.

## Test coverage for the claims above

`backend/test_antidote.py` asserts, against the live public APIs, that: (1)
human lysozyme returns a real structurally-observed partner and never returns
itself as a partner, (2) TB InhA is correctly routed to the small-molecule
modality and the isoniazid adduct or triclosan is recovered, (3) MHC
epitope-presentation complexes are excluded from InhA's candidate list, and
(4) a deliberately low-confidence pocket is rejected before any retrieval
call is made. All checks pass as of this writing.
