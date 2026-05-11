"""ClinicalTrials.gov recall-first query universe.

This configuration is intentionally sponsor-agnostic. It does not list known
companies as discovery inputs. The backend discovers sponsors from the trial
payloads returned by ClinicalTrials.gov, then enrichment layers may optionally
resolve tickers/news handles later.

The important design choice is query breadth + provenance:
- run multiple query families per strategic lane;
- use structured ClinicalTrials.gov search areas where possible;
- paginate within bounded caps;
- dedupe by NCT ID after retrieval;
- retain query provenance so the dashboard can explain what was found and why.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClinicalTrialSearchSpec:
    label: str
    query: str
    strategic_lane: str
    priority: int
    query_area: str = "term"  # term | cond | intr | titles | spons
    max_pages: int = 2
    query_family: str = "broad_term"
    retain_any_terms: tuple[str, ...] = ()


# No sponsor names are used below. The discovery layer queries by target,
# disease, modality, program/asset aliases, and trial-title/acronym vocabulary.
CLINICAL_TRIAL_SEARCH_SPECS: tuple[ClinicalTrialSearchSpec, ...] = (
    # CDH6 / cadherin-6 target/program discovery
    ClinicalTrialSearchSpec(
        label="CDH6 / Ovarian ADC",
        query='CDH6 OR "cadherin 6" OR "cadherin-6" OR "DS-6000" OR "DS-6000a" OR "raludotatug deruxtecan" OR "R-DXd"',
        strategic_lane="Direct pipeline / ovarian ADC relevance",
        priority=1,
        query_area="term",
        max_pages=4,
        query_family="target_program_term",
        retain_any_terms=("cdh6", "cadherin", "ds-6000", "raludotatug", "r-dxd", "adc", "antibody-drug", "ovarian", "gynecologic"),
    ),
    ClinicalTrialSearchSpec(
        label="CDH6 / Ovarian ADC",
        query='CDH6 OR "cadherin 6" OR "cadherin-6" OR "DS-6000" OR "raludotatug deruxtecan"',
        strategic_lane="Direct pipeline / ovarian ADC relevance",
        priority=1,
        query_area="intr",
        max_pages=3,
        query_family="intervention_target",
        retain_any_terms=("cdh6", "cadherin", "ds-6000", "raludotatug", "adc"),
    ),
    ClinicalTrialSearchSpec(
        label="CDH6 / Ovarian ADC",
        query='CDH6 OR "cadherin 6" OR "cadherin-6" OR "DS-6000" OR "raludotatug"',
        strategic_lane="Direct pipeline / ovarian ADC relevance",
        priority=1,
        query_area="titles",
        max_pages=2,
        query_family="title_acronym_target",
        retain_any_terms=("cdh6", "cadherin", "ds-6000", "raludotatug"),
    ),

    # B7-H4 adjacent target discovery
    ClinicalTrialSearchSpec(
        label="B7-H4 ADC",
        query='"B7-H4" OR "B7H4" OR "VTCN1" OR "B7x" OR "B7S1"',
        strategic_lane="Direct target-adjacent competitive relevance",
        priority=1,
        query_area="term",
        max_pages=4,
        query_family="target_program_term",
        retain_any_terms=("b7-h4", "b7h4", "vtcn1", "b7x", "b7s1", "adc", "antibody-drug", "ovarian", "gynecologic"),
    ),
    ClinicalTrialSearchSpec(
        label="B7-H4 ADC",
        query='"B7-H4" OR "B7H4" OR "VTCN1" OR "B7x" OR "B7S1"',
        strategic_lane="Direct target-adjacent competitive relevance",
        priority=1,
        query_area="intr",
        max_pages=3,
        query_family="intervention_target",
        retain_any_terms=("b7-h4", "b7h4", "vtcn1", "b7x", "b7s1", "adc", "antibody-drug"),
    ),
    ClinicalTrialSearchSpec(
        label="B7-H4 ADC",
        query='"B7-H4" OR "B7H4" OR "VTCN1"',
        strategic_lane="Direct target-adjacent competitive relevance",
        priority=1,
        query_area="titles",
        max_pages=2,
        query_family="title_acronym_target",
        retain_any_terms=("b7-h4", "b7h4", "vtcn1"),
    ),

    # Ovarian / gynecologic ADC category discovery. These intentionally combine
    # narrower ADC/modality searches with broader disease searches that are
    # filtered downstream by relevance terms, so non-ADC ovarian trials do not
    # silently dominate the executive synthesis.
    ClinicalTrialSearchSpec(
        label="Ovarian ADC",
        query='("ovarian cancer" OR "ovarian carcinoma" OR "platinum-resistant ovarian" OR "fallopian tube cancer" OR "primary peritoneal cancer" OR gynecologic OR gynaecologic) AND (ADC OR "antibody drug conjugate" OR "antibody-drug conjugate" OR "folate receptor" OR "FR alpha" OR "FRα" OR TROP2 OR HER2 OR NaPi2b)',
        strategic_lane="Ovarian ADC category momentum",
        priority=2,
        query_area="term",
        max_pages=4,
        query_family="disease_modality_term",
        retain_any_terms=("ovarian", "fallopian", "peritoneal", "gynecologic", "gynaecologic", "adc", "antibody-drug", "folate", "fr alpha", "frα", "trop2", "her2", "napi2b"),
    ),
    ClinicalTrialSearchSpec(
        label="Ovarian ADC",
        query='ADC OR "antibody drug conjugate" OR "antibody-drug conjugate" OR "folate receptor" OR "FR alpha" OR "FRα" OR TROP2 OR HER2 OR NaPi2b',
        strategic_lane="Ovarian ADC category momentum",
        priority=2,
        query_area="intr",
        max_pages=4,
        query_family="intervention_modality",
        retain_any_terms=("ovarian", "fallopian", "peritoneal", "gynecologic", "gynaecologic", "adc", "antibody-drug", "folate", "fr alpha", "frα", "trop2", "her2", "napi2b"),
    ),
    ClinicalTrialSearchSpec(
        label="Ovarian ADC",
        query='"ovarian cancer" OR "ovarian carcinoma" OR "fallopian tube cancer" OR "primary peritoneal cancer" OR "gynecologic cancer" OR "gynaecologic cancer"',
        strategic_lane="Ovarian ADC category momentum",
        priority=2,
        query_area="cond",
        max_pages=3,
        query_family="condition_broad_relevance_filtered",
        retain_any_terms=("adc", "antibody-drug", "conjugate", "folate", "fr alpha", "frα", "trop2", "her2", "napi2b", "cdh6", "b7-h4", "b7h4"),
    ),

    # Broad ADC oncology gives competitive context but is intentionally lower
    # priority so it never overwrites more specific lane provenance.
    ClinicalTrialSearchSpec(
        label="ADC Oncology",
        query='(oncology OR cancer OR tumor OR tumour OR neoplasm) AND (ADC OR "antibody drug conjugate" OR "antibody-drug conjugate")',
        strategic_lane="Broad ADC category pressure / validation",
        priority=3,
        query_area="term",
        max_pages=3,
        query_family="broad_modality_context",
        retain_any_terms=("adc", "antibody-drug", "conjugate"),
    ),
    ClinicalTrialSearchSpec(
        label="ADC Oncology",
        query='ADC OR "antibody drug conjugate" OR "antibody-drug conjugate"',
        strategic_lane="Broad ADC category pressure / validation",
        priority=3,
        query_area="intr",
        max_pages=3,
        query_family="intervention_modality_context",
        retain_any_terms=("adc", "antibody-drug", "conjugate"),
    ),

    # Side channels are kept, but bounded tightly so they do not crowd out the
    # oncology battlefield.
    ClinicalTrialSearchSpec(
        label="Alzheimer's Side Channel",
        query='Alzheimer AND (antibody OR immunotherapy OR biomarker OR ApoE4)',
        strategic_lane="Side-channel scientific drift",
        priority=4,
        query_area="term",
        max_pages=1,
        query_family="side_channel_term",
        retain_any_terms=("alzheimer", "apoe4", "antibody", "immunotherapy", "biomarker"),
    ),
    ClinicalTrialSearchSpec(
        label="Bone Disease Side Channel",
        query='("bone disease" OR osteoporosis OR osteoarthritis OR "osteogenesis imperfecta") AND (antibody OR biomarker OR biologic OR Siglec-15)',
        strategic_lane="Side-channel scientific drift",
        priority=4,
        query_area="term",
        max_pages=1,
        query_family="side_channel_term",
        retain_any_terms=("bone", "osteoporosis", "osteoarthritis", "osteogenesis", "siglec", "antibody", "biomarker", "biologic"),
    ),
)

# Recall-first but bounded for Streamlit Cloud safety.
CLINICAL_TRIALS_PAGE_SIZE = 50
CLINICAL_TRIALS_MAX_PAGES_PER_SPEC = 5
CLINICAL_TRIALS_TIMEOUT_SECONDS = 10
