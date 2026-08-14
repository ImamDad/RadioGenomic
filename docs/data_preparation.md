

## 11. docs/data_preparation.md

```markdown
# Data Preparation Guide

## Overview

MS-HGNN requires four types of data for each patient:
1. **CT Imaging**: Radiomic features extracted from CT scans
2. **PET Imaging**: Radiomic features extracted from PET scans
3. **Clinical Data**: Patient demographics, stage, histology, treatment
4. **Genomic Data**: GSVA pathway scores from RNA-seq

## TCIA NSCLC Radiogenomics Dataset

The primary dataset is the NSCLC Radiogenomics dataset from TCIA:

**Access:**
- URL: https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/
- DOI: 10.7937/K9/TCIA.2017.7hs46erv

**Data Description:**
- 211 patients with NSCLC
- Pre-treatment CT and PET/CT imaging
- Expert-validated tumor segmentations
- Comprehensive clinical data
- RNA-seq genomic data

## Data Processing Pipeline

### 1. Image Preprocessing

#### CT Processing

```python
from utils.preprocessing import ImagePreprocessor

preprocessor = ImagePreprocessor(ct_min_hu=-1000, ct_max_hu=1000)

# Normalize CT
ct_normalized = preprocessor.normalize_ct(ct_volume)

# Align CT with PET
ct_aligned, pet_aligned = preprocessor.align_ct_pet(ct_volume, pet_volume)
