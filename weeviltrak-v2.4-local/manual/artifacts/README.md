# Artifact Bundle

Place the immutable release files in this directory before running the client
notebook:

```text
base_model.pkl
termination_model.pkl
config.yml
manifest.json
checksums.sha256
```

Do not commit binary model artifacts to ordinary Git unless the delivery
process explicitly uses Git LFS. The immutable artifact bundle should be
delivered as a controlled archive or release attachment.

The currently published `v2.4-2027-20260803` bundle predates the explicit
`min_history_days: 7` policy field. Customer delivery must use a new frozen
configuration revision that records this field alongside the unchanged pickle
artifacts.
