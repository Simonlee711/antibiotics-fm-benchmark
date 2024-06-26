---
# Benchmarking Foundation Models on Antibiotic Susceptibility

---
[![Overleaf](https://chianglab.github.io/antibiotics-benchmark/)]([https://chianglab.github.io/antibiotics-benchmark/)

# Abstract

The rise of antibiotic-resistant bacteria has been identified as a critical global healthcare crisis that compromises the efficacy of essential antibiotics. This crisis is largely driven by the inappropriate and excessive use of antibiotics, which leads to increased bacterial resistance. In response, clinical decision support systems integrated with electronic health records (EHRs) have emerged as a promising solution. These systems employ machine learning models to improve antibiotic stewardship by providing actionable, data-driven insights. This study therefore evaluates pre-trained language models for predicting antibiotic susceptibility, using several open-source models available on the Hugging Face platform. Despite the abundance of models and ongoing advancements in the field, a consensus on the most effective model for encoding clinical knowledge remains unclear.

# Data

The MIMIC-IV-ED dataset, part of the extensive MIMIC-IV collection, concentrates on emergency department records from a major hospital. It anonymizes and details patient demographics, triage, vitals, tests, medications, and outcomes, aiding research in emergency care and hospital operations. Access follows strict privacy regulations.

# Method

![Method](../pictures/Method.png)

### Running the code

To replicate our results, run the file benchmark.ipynb

# Foundation Models Benchmarked

| Foundation Model | Source |
|-----------------|-----------------|
| BioBERT  | https://huggingface.co/dmis-lab/biobert-v1.1  |
| ClinicalBERT  | https://huggingface.co/medicalai/ClinicalBERT |
| MedBERT  | https://huggingface.co/Charangan/MedBERT  |
| RadBERT  | https://huggingface.co/StanfordAIMI/RadBERT  |
| Bio-LM   | https://huggingface.co/EMBO/bio-lm |
| Bio-Megatron | https://huggingface.co/EMBO/BioMegatron345mUncased |
| LinkBERT  | https://huggingface.co/michiyasunaga/LinkBERT-large  |
| distil-bert  | https://huggingface.co/docs/transformers/en/model_doc/distilbert  |
| Bluebert | https://huggingface.co/bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12 |
| BioBERT | https://huggingface.co/pritamdeka/BioBert-PubMed200kRCT |
| PubMedBERT | https://huggingface.co/NeuML/pubmedbert-base-embeddings |
| Gatotron | https://huggingface.co/UFNLP/gatortronS |
| BiomedRoBERTa | https://huggingface.co/allenai/biomed_roberta_base |
| Bio+ClinicalBERT | https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT |
| SciBERT | https://huggingface.co/allenai/scibert_scivocab_uncased |
| BioLM | https://huggingface.co/EMBO/bio-lm |
| RadBERT | https://huggingface.co/StanfordAIMI/RadBERT |
| LinkBERT | https://huggingface.co/michiyasunaga/LinkBERT-large |



---
Authors:
- Helio Halperin
- Simon Lee
- Jeffrey Chiang
