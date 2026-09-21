# Data rights and provenance

Author-created numerical predictions, analysis tables and regrouping maps are
released under Creative Commons Attribution 4.0 International (CC BY 4.0):
https://creativecommons.org/licenses/by/4.0/

Attribute Chengfei Xu and Yunrui Jiang, ROBird-Bench, version v1.0.0 (2026),
and identify changes in derived distributions. Trained study-specific heads
are distributed with the author-created code; rights in third-party input
data or pretrained backbones are not reassigned by this release.

iNaturalist photographs and source metadata retain their original rights.
Per-photo `license_code`, `attribution`, source URL, observation/photo IDs and
byte SHA-256 are provided. License codes are recorded from the collection
manifest and are not a new grant from the study authors. No missing license
version is invented. Source licenses include CC0, CC BY and CC BY-SA; consult
the linked source for applicable terms. Photographs are retrieved separately,
not redistributed under this repository's MIT or derived-data CC BY license.

Public release manifests remove workstation paths and collection locality/time
fields from the downloadable retrieval tables. Group, observer, photo and taxon
IDs are preserved to permit leakage and dependency checks. Photo attribution
is preserved to support source-license obligations. These IDs can link back
to public observations and must not be described as anonymous identifiers.

Archived machine-generated result records may retain original absolute generic
workstation paths as historical provenance. They are not paths a new user must
possess. No credentials, chat transcripts, contributor profiles or raw API
census payloads are included. Historical hash records remain historical;
transformed public manifests have separate hashes in FILE_MANIFEST.json.
