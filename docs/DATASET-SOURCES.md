# DATASET-SOURCES.md — Biological connectome dataset sources (companion for Agent B)

**Owner:** Agent E (research only). **Date:** 2026-10-30. All facts verified from the cited
pages on this date; anything not confirmed is marked **unverified**.

Scope: official sources for importing a first biological connectome ("MaleCNS v1.0") into the
Artificial Life project. Agent B implements the import.

---

## 1. Primary source — Janelia FlyEM MaleCNS v1.0

### 1.1 Official pages (all fetched 2026-10-30)

| Page | URL |
|---|---|
| Project landing (errata: see below) | http://male-cns.janelia.org/ |
| Download page | https://male-cns.janelia.org/download/ |
| Release notes | https://male-cns.janelia.org/release/ |
| Janelia overview | https://www.janelia.org/project-team/flyem/male-cns-connectome |
| Neuroglancer scene | https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.json |

News items verified from http://male-cns.janelia.org/:
"2026-06-08 – MaleCNS v1.0 released!"; "2025-10-03 – MaleCNS v0.9 released!"; v1.0 follows a
v0.9 (October 5, 2025) per the Release Notes page.

Note: an alternate documentation URL, `https://janelia-flyem.github.io/male-cns/`, appears in
a third-party write-up but was **not fetched** in this session — **unverified**; prefer the
janelia.org pages above.

### 1.2 Format & files (verified from the Download page)

All bulk files live in Google Storage bucket `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`
and the Download page hotlinks them over HTTPS (`storage.googleapis.com/flyem-male-cns/...`).
Neuron-level tables are **Apache Arrow Feather** (readable with pyarrow/pandas/R arrow).

| File | Size (page) | What it is |
|---|---|---|
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | **1.1 GB** | "segment-to-segment connection strengths for all segments in the dataset (excluding those with no synapses). This is the full connection graph." ✅ matches the expected ~1.1 GB Feather in the directive |
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 13 MB | "Curated neuron annotations (classes, types, sides, etc.), excluding neurotransmitter properties." |
| `body-neurotransmitters-male-cns-v1.0.feather` | 42 MB | aggregate neurotransmitter predictions per neuron |
| `body-stats-male-cns-v1.0-minconf-0.5.feather` | 780 MB | summary statistics (synapse counts) per segment |
| `syn-points-male-cns-v1.0-minconf-0.5.feather` | 12.7 GB | pre/post synapse locations, body IDs, ROIs; `kind` column = `PreSyn`/`PostSyn`; coordinates in 8 nm voxel units |
| `syn-partners-male-cns-v1.0-minconf-0.5.feather` | 6.8 GB | synaptic partner pairs |
| `tbar-neurotransmitters-male-cns-v1.0.feather` | 2.7 GB | NT probability per pre-synapse |

`minconf-0.5` = synapses detected with confidence ≥ 0.5 (release threshold).

Volumes (separately hosted, not needed for v1: import): EM N5
`gs://flyem_cns_z0720_07m_dvidcoords_n5` (8 nm isotropic, uint8), JPEG-encoded CLAHE EM
`gs://flyem-male-cns/em/em-clahe-jpeg`, proofread segmentation
`gs://flyem-male-cns/v1.0/segmentation` (8 nm, uint64), nuclei seg 16 nm, ROI volumes
(`fullbrain-roi-v4` max ROI 96; `malecns-vnc-neuropil-roi-v0` max 27). Skeletons:
`gs://flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/` (SWC, 8 nm units)
and precomputed variants (1 nm units), plus mirrored and JRC2018-unisex-template transforms.
Also: neo4j database export `gs://flyem-male-cns/v1.0/database/neo4j` (built for Neo4j
4.4.16) + `neuprint-inputs` CSVs.

### 1.3 Connectivity columns (verified vs unverified)

- **Verified (official):** the `syn-partners` table columns are stated verbatim on the
  Download page: `x_pre, y_pre, z_pre, body_pre, conf_pre, x_post, y_post, z_post, body_post,
  conf_post, primary_post`.
- **Unverified:** the exact column names of `connectome-weights-male-cns-v1.0-minconf-0.5.feather`.
  The official page describes it only as "segment-to-segment connection strengths". Third-party
  consumers say it holds neuron-to-neuron edges with synapse counts (flymsg project docs:
  "151.9M segment-to-segment edges with synapse counts"; HuggingFace mirror: "from neuron, to
  neuron, connection weight"; both unverified as to names). **Do not hard-code `body_pre`/
  `body_post`/`weight` — Agent B must print the Arrow schema after download and map columns
  programmatically.**
- **Annotations:** official page says the body-annotations table carries "classes, types,
  sides, etc.". Third-party (unverified enumeration): type, instance, class, side, status,
  dimorphism, fru/dsx, synonyms. `somaSide` is the neuPrint/FlyWire naming — **whether the
  MaleCNS table uses `somaSide` or `side` is unverified**; schema inspection required.
- Query-side alternative (verified from the Download page): neuPrint Python API —
  `Client("https://neuprint.janelia.org", dataset='male-cns:v1.0', token=token)`; account +
  token required; R variant `neuprintr`; `navis` for morphology.

### 1.4 Dataset license — VERIFIED

The Download page states, twice, as the dataset's own license statement (footer + body):

> "The Male CNS dataset is licensed under CC-BY." — hyperlinked to
> https://creativecommons.org/licenses/by/4.0/ (i.e. **CC BY 4.0**)

The Release Notes page carries the identical statement. This is the **dataset's own license
page**, not a paper's license — HHMI's paper/publication licensing elsewhere does not govern
this page, and conversely nothing observed here confers a *narrower* grant than CC BY 4.0.
Verdict: **MaleCNS v1.0 dataset license = CC BY 4.0 (verified from the official dataset
pages; the exact creativecommons.org target page itself was not fetched — the link text
resolves to the CC BY 4.0 deed).** Attribution + version citation required in our app.

### 1.5 Reported scale (verified-with-sources; note the variance)

| Figure | Value | Source |
|---|---|---|
| Neuron (annotated body) count | 211,577 annotated bodies | flymsg project docs (third-party, unverified vs neuPrint live) |
| "Neurons" | ~166,700 | Virtual Fly Brain download guide (third-party but neutral) and HuggingFace mirror (third-party) |
| Edges | 151.9M segment-to-segment edges | flymsg (third-party, unverified) |
| "Synaptic connections" | ~25 million | HuggingFace mirror (third-party, unverified) |

The 151.9M vs ~25M discrepancy likely reflects segment-to-segment edges vs curated
neuron-to-neuron connections; **neuPrint live instance statistics were NOT verified
(account/token-gated)**. Agent B should compute and document final numbers from the actual
tables at import time and treat all counts above as estimates.

---

## 2. FlyWire FAFB v783 (comparative candidate — rejected as first choice)

### 2.1 Official pages (fetched 2026-10-30)

| Page | URL |
|---|---|
| Codex portal | https://codex.flywire.ai/ |
| Codex bulk-download portal | https://codex.flywire.ai/api/download?dataset=fafb (requires Google sign-in per page) |
| Citing guidelines | https://flywire.ai/guidelines |
| Community principles | https://edit.flywire.ai/principles.html |
| Zenodo static release record | https://zenodo.org/records/10676866 (verified via https://zenodo.org/api/records/10676866) |
| Annotations repo | https://github.com/flyconnectome/flywire_annotations |

### 2.2 Facts verified

- FlyWire guidelines page: "**FlyWire's public release data is made available under license
  CC BY-NC 4.0**"; latest public release is **version 783**, a **October 2023 snapshot**;
  Codex-visible annotations auto-included.
  → Verdict: **dataset license = CC BY-NC 4.0 (verified from flywire.ai/guidelines).**
  Commercial use is **not permitted** under CC BY-NC 4.0.
- Codex datasets page: "FAFB v783 (CB) Female Adult Fly Brain **139,255 neurons**,
  **3,732,460 connections**" (fetched page content).
- Zenodo record (API metadata): title "FlyWire Whole-brain Connectome Connectivity Data",
  DOI 10.5281/zenodo.10676866, **license `"cc-by-4.0"` on the record**, publication 2024-06-02.
  Files + md5: `flywire_synapses_783.feather` 9,492,998,242 B (md5 f8f1b97c9d4b0ea9b4c8b287f6b99091),
  `proofread_connections_783.feather` 852,022,274 B (md5 f48f972d262323a102aed49af1396b8a),
  `proofread_root_ids_783.npy` 1,114,168 B, `per_neuron_neuropil_count_{pre,post}_783.feather`.
- **Tension flagged honestly:** the Zenodo *connectivity subset record* is marked CC BY 4.0
  in its own metadata, while FlyWire's guidelines mark the *public release data* CC BY-NC
  4.0. We could not verify which grant FlyWire (the licensor) considers authoritative for the
  parquet/feather dump. **Conservative reading: treat all FlyWire-derived data as CC BY-NC
  4.0 unless FlyWire states otherwise.** This unresolved noncommercial risk is the core
  reason it is beaten by MaleCNS.
- Zenodo `proofread_connections_783.feather` columns (verified from record metadata):
  `pre_pt_root_id, post_pt_root_id, neuropil, syn_count, gaba_avg, ach_avg, glut_avg,
  oct_avg, ser_avg, da_avg`.

### 2.3 Why v783 is a poor FIRST candidate vs MaleCNS

1. **License:** CC BY-NC 4.0 (per FlyWire's own guidelines) vs MaleCNS CC BY 4.0 (per the
   dataset's own pages). Our project may want permissive downstream reuse; MaleCNS is clean.
2. **Scope/semantics:** FlyWire = female brain only (139k neurons); MaleCNS = full male CNS
   (brain + VNC) with sex-dimorphism annotations — better fit for male-organism behavior sims.
3. **Volume:** FlyWire's complete synapse table alone is 9.5 GB; the useful proofread edge
   table (852 MB) is comparable in size to MaleCNS's 1.1 GB graph, so size is not the
   differentiator — licensing and biological scope are.

---

## 3. Other official sources worth noting (license state per cited page)

| Dataset | Official route | License per cited page |
|---|---|---|
| MaleCNS v1.0 via neuPrint | neuprint.janelia.org, dataset `male-cns:v1.0` (verified from Download page) | Same CC-BY dataset statement (Download page) |
| MANC (male VNC) v1.2.1 | Janelia MANC page; bucket `flyem-manc-exports` (per Virtual Fly Brain guide) | **CC BY 4.0 per VFB guide's statement** (Janelia MANC page itself not fetched — unverified directly) |
| Male optic lobe v1.1 | Janelia optic lobe page; bucket `flyem-optic-lobe` (per VFB guide) | **CC BY 4.0 per VFB guide** (unverified directly) |
| Hemibrain v1.2.1 (female central brain) | Hemibrain release page; neuPrint dataset `hemibrain:v1.2.1` | **CC BY 4.0 per VFB guide's blanket statement "All sources above that state one use CC BY 4.0"** — hemibrain's own page not fetched in this session (**unverified directly**) |
| BANC v888 (female brain+VNC) | Harvard Dataverse (per VFB guide) | **License not stated** on the page examined (**unverified**) |
| FANC (female VNC) | FANC_auto_recon wiki (per VFB guide) | **"the wiki does not state a licence"** (VFB guide) → treat as unlicensed/unknown |
| FlyWire annotation products for v783 | github.com/flyconnectome/flywire_annotations | Repo license not verified in this session (**unverified**) |

Virtual Fly Brain's aggregate page (https://www.virtualflybrain.org/docs/data/download-connectomes/)
is a neutral index; used above only where marked.
Other useful infra (verified from official pages): `tensorstore` / `cloud-volume` /
`neuroglancer precomputed` format readers (MaleCNS Download page); `navis` +
`navis-flybrains` for skeleton transforms between MaleCNS space, JRC2018 unisex template and
FlyWire frame (MaleCNS Download page).

---

## 4. Acquisition & storage procedure

### 4.1 Manual download (MaleCNS v1.0)

The Download page hotlinks the four flat-connectome Feather files over plain HTTPS, e.g.:

```powershell
New-Item -ItemType Directory -Force -Path "C:\Users\ASUS\Desktop\ArtificialLife\data-external\malecns-v1.0"
$base = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
# core table only (v1.0 graph):
Invoke-WebRequest "$base/connectome-weights-male-cns-v1.0-minconf-0.5.feather" -OutFile "data-external\malecns-v1.0\"
# plus (recommended) annotations + NT:
#   body-annotations-male-cns-v1.0-minconf-0.5.feather (13 MB)
#   body-neurotransmitters-male-cns-v1.0.feather       (42 MB)
```

For whole-directory or volume access use the Google Cloud CLI with the bucket paths
(`gsutil -m cp -r gs://flyem-male-cns/v1.0/... ./malecns-v1.0/`) — the Download page lists
`gs://` paths but gives no CLI commands; HTTPS mirrors are stated for the flat-connectome
folder. Local-only downloads need no neuPrint account/token (token is only for live queries).

Minimum useful set: connectome-weights (1.1 GB) + body-annotations (13 MB) +
body-neurotransmitters (42 MB) ≈ 1.2 GB. Do NOT pull syn-points (12.7 GB), EM volumes or
neo4j export for the first version.

### 4.2 Storage — OUTSIDE the git repository

Store under `C:\Users\ASUS\Desktop\ArtificialLife\data-external\` (sibling of the repo files,
**not** inside `rust\` / `electron\` / wherever the repo root is except as an ignored path),
to keep ~1.2 GB of binaries out of git history:

- The repo agent owns `.gitignore`, so request from that agent (do not edit their files):
  `data-external/` (and `*.feather`, `*.npy`) ignored.
- Never commit the feather blobs; commit only the checksum manifest (below).

### 4.3 sha256 pinning

1. After download, compute hashes locally (PowerShell):
   `Get-FileHash -Algorithm SHA256 <file> | Format-List` and also record file size in bytes.
2. Maintain `docs`-adjacent note (Agent-owned) or give to Agent B a manifest file inside the
   repo, e.g. `data/manifest-malecns-v1.0.csv`, with columns:
   `filename, sha256, size_bytes, source_url, fetched_date, upstream_version` (here
   `male-cns:v1.0`, release date 2026-06-08).
3. Cross-check upstream checksums: the official MaleCNS Download page publishes **sizes but
   no per-file hashes** (no sha256/md5 published — verified page content). The FlyWire Zenodo
   record publishes **md5** per file (not sha256) — verify integrity against md5s from the
   record and pin our own sha256 in the manifest. On any mismatch, re-download and re-check.
4. Pin URLs + release version in import code; reject files whose recorded sha256 differs.

---

## 5. Decision rationale (for Agent B)

Implement **MaleCNS v1.0 import first**:

1. Dataset's own license page states CC BY 4.0 (verified) — cleanest grant found.
2. Officially hotlinked HTTPS Feather tables; sizes match the directive (~1.1 GB graph ✔).
3. Officially documented `body_pre`/`body_post` columns in the partner table, plus curated
   class/type/side annotations and consensus neurotransmitters — exactly the columns the
   simulator needs.
4. FlyWire (CC BY-NC 4.0, female brain, unresolved Zenodo-record license tension) relegated
   to "later candidate, pending confirmation from FlyWire about reuse rights."

Contact surface for Agent B **must execute** at import: read Arrow schema at runtime, print
column names, and map (pre, post, weight) programmatically — exact connectome-weights column
names are unverified.

---

*Citations fetched 2026-10-30. Rows/pages marked **unverified** were not confirmed from a
primary page in this session and must be re-verified before acting on them.*
