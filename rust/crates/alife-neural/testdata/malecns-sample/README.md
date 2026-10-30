SYNTHETIC TEST DATA — NOT REAL DATA
===================================

The files in this directory are a HAND-WRITTEN miniature sample in the
normalized flybrain-connectome TSV format (tools/convert_malecns.py output
schema). Their sole purpose is `cargo test -p alife-neural` importer unit
tests. They are NOT derived from the MaleCNS v1.0 feather files, contain no
real neuron IDs, and must never be presented as source-derived data.

- edges.tsv  : 5 synthetic synapses between 4 synthetic bodies
               (body 4 deliberately absent from nodes.tsv to exercise the
               implicit-node path of connectome_import).
- nodes.tsv  : 3 synthetic bodies; one row has an EMPTY final field
               (missing somaSide) to exercise unannotated-field retention.
- sidecar.json : format-validated sidecar, dataset field explicitly labeled
               synthetic, sha256 blank per policy.

A converter run on real MaleCNS data would overwrite these names inside its
own output directory; this directory is test-only and dataset-neutral.
