# Patient B endpoint supplement

This is a transport-only backup of the completed 420-row patient endpoint, its
prepared inputs, original training receipts and exact source. It performs no
inference, decoding, evaluation, fitting or official submission.

The immutable original raw collection is already on mount
`5a9955da-5240-41cc-9e52-05d3d8da8a3c` at
`sequence-round2/official-901-B/patient-bridge-recovery/patient-complete.zip`:
897210 bytes, SHA-256
`1df686ceaaf69e55913198482494533ede264dad2effa4561dd4ed4da9593e36`.
It is referenced, never copied into this supplement. Weights and the existing
native tokenizer are also references only. Their actual identities remain in
the unchanged original receipts and export plan; the supplement does not claim
to contain optimizer state, weights or a complete standalone model runtime.

`endpoint_supplement.py pack --run-dir D` creates a deterministic ZIP and
`members-manifest.json` beneath `D/endpoint-supplement-v1`. Each member records
its original path, bytes and actual SHA. Original JSON bytes and absolute refs
are preserved; the inventory maps those paths to packaged names. Identical
expected prompts are an explicit alias of the single prepared prompts member.

`endpoint_supplement.py deploy --run-dir D` inventories the new target first,
uploads missing immutable 64 KiB chunks, verifies their SHA, assembles the
original local ZIP bytes, and verifies all safe ZIP members. It publishes the
ZIP, manifest and extracted members under the new owned target
`sequence-round2/official-901-B/patient-endpoint-supplement-v1`, fsyncs files
and their ancestor directories, then reopens every file for SHA verification.
Every tool error is retained. UNKNOWN stops without resubmitting a mutation;
a later invocation inventories existing chunks first. Explicit LUNA_OFFLINE
requires at least 180 seconds before another attempt. No GPU state is queried.

`persistent-receipt.json` in the local supplement directory is authoritative
for actual successful deployment. A prepared local package alone is not a
claim of remote persistence. No credentials, HTTP tokens or ephemeral token
files are packaged.
