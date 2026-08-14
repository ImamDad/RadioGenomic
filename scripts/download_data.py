#!/usr/bin/env python
"""
Download data from TCIA and TCGA
"""

import os
import argparse
import subprocess
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile


def download_tcia_data(data_dir: str):
    """
    Download TCIA NSCLC Radiogenomics data
    
    Args:
        data_dir: Directory to save data
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("Downloading TCIA NSCLC Radiogenomics Dataset")
    print("=" * 50)
    
    # TCIA NSCLC Radiogenomics Dataset DOI: 10.7937/K9/TCIA.2017.7hs46erv
    # In practice, this would use the TCIA API or download directly
    
    # For demonstration, we'll create placeholder files
    print("NOTE: This is a demonstration script.")
    print("Please download the actual data from:")
    print("https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/")
    
    # Create placeholder files
    create_placeholder_data(data_dir)
    
    print("\nData download complete!")
    print(f"Files saved to: {data_dir}")


def download_tcga_luad(data_dir: str):
    """
    Download TCGA-LUAD data
    
    Args:
        data_dir: Directory to save data
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("Downloading TCGA-LUAD Dataset")
    print("=" * 50)
    
    print("NOTE: This is a demonstration script.")
    print("Please download the actual data from:")
    print("https://www.cancerimagingarchive.net/collection/tcga-luad/")
    print("https://portal.gdc.cancer.gov/projects/TCGA-LUAD")
    
    # Create placeholder files for TCGA-LUAD
    create_placeholder_tcga_data(data_dir)
    
    print("\nData download complete!")
    print(f"Files saved to: {data_dir}")


def create_placeholder_data(data_dir: Path):
    """
    Create placeholder data files for demonstration
    
    Args:
        data_dir: Directory to save placeholder data
    """
    import numpy as np
    import pandas as pd
    
    # Create radiomics data
    n_patients = 130
    n_features = 131
    
    np.random.seed(42)
    
    # CT radiomics
    ct_data = np.random.normal(0, 1, (n_patients, n_features))
    pd.DataFrame(ct_data).to_csv(data_dir / 'radiomics_ct.csv', index=False)
    
    # PET radiomics
    pet_data = np.random.normal(0, 1, (n_patients, n_features))
    pd.DataFrame(pet_data).to_csv(data_dir / 'radiomics_pet.csv', index=False)
    
    # Clinical data
    clinical_data = pd.DataFrame({
        'patient_id': [f'TCIA_{i:03d}' for i in range(n_patients)],
        'age': np.random.normal(65, 10, n_patients).astype(int),
        'sex': np.random.choice([0, 1], n_patients),
        'stage': np.random.choice([1, 2, 3, 4], n_patients, p=[0.3, 0.3, 0.25, 0.15]),
        'histology': np.random.choice([0, 1], n_patients, p=[0.685, 0.315]),
        'smoking_status': np.random.choice([0, 1, 2], n_patients),
        'year': np.random.choice([1992, 1993, 1994, 1995, 1996], n_patients)
    })
    clinical_data.to_csv(data_dir / 'clinical_data.csv', index=False)
    
    # Genomic data (50 pathways)
    genomic_data = np.random.normal(0, 1, (n_patients, 50))
    pd.DataFrame(genomic_data).to_csv(data_dir / 'genomic_data.csv', index=False)
    
    # Survival data
    survival_data = pd.DataFrame({
        'survival_time': np.random.exponential(30, n_patients),
        'survival_event': np.random.binomial(1, 0.7, n_patients)
    })
    survival_data.to_csv(data_dir / 'survival_data.csv', index=False)
    
    # Recurrence data
    recurrence_data = pd.DataFrame({
        'recurrence': np.random.binomial(1, 0.4, n_patients)
    })
    recurrence_data.to_csv(data_dir / 'recurrence_data.csv', index=False)
    
    print("Placeholder data created:")

def create_placeholder_tcga_data(data_dir: Path):
    """
    Create placeholder data for TCGA-LUAD
    
    Args:
        data_dir: Directory to save placeholder data
    """
    import numpy as np
    import pandas as pd
    
    n_patients = 69
    
    np.random.seed(123)
    
    # CT radiomics
    ct_data = np.random.normal(0, 1, (n_patients, 131))
    pd.DataFrame(ct_data).to_csv(data_dir / 'tcga_radiomics_ct.csv', index=False)
    
    # PET radiomics
    pet_data = np.random.normal(0, 1, (n_patients, 131))
    pd.DataFrame(pet_data).to_csv(data_dir / 'tcga_radiomics_pet.csv', index=False)
    
    # Clinical data
    clinical_data = pd.DataFrame({
        'patient_id': [f'TCGA_{i:03d}' for i in range(n_patients)],
        'age': np.random.normal(65, 10, n_patients).astype(int),
        'sex': np.random.choice([0, 1], n_patients),
        'stage': np.random.choice([1, 2, 3, 4], n_patients, p=[0.3, 0.3, 0.25, 0.15]),
    })
    clinical_data.to_csv(data_dir / 'tcga_clinical_data.csv', index=False)
    
    # Genomic data
    genomic_data = np.random.normal(0, 1, (n_patients, 50))
    pd.DataFrame(genomic_data).to_csv(data_dir / 'tcga_genomic_data.csv', index=False)
    
    # Survival data
    survival_data = pd.DataFrame({
        'survival_time': np.random.exponential(30, n_patients),
        'survival_event': np.random.binomial(1, 0.7, n_patients)
    })
    survival_data.to_csv(data_dir / 'tcga_survival_data.csv', index=False)
    
    # Recurrence data
    recurrence_data = pd.DataFrame({
        'recurrence': np.random.binomial(1, 0.4, n_patients)
    })
    recurrence_data.to_csv(data_dir / 'tcga_recurrence_data.csv', index=False)
    
    print("TCGA-LUAD placeholder data created:")


def main():
    parser = argparse.ArgumentParser(description='Download NSCLC data')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory to save data')
    parser.add_argument('--dataset', type=str, choices=['tcia', 'tcga', 'all'],
                       default='all', help='Dataset to download')
    
    args = parser.parse_args()
    
    if args.dataset in ['tcia', 'all']:
        download_tcia_data(os.path.join(args.data_dir, 'tcia'))
    
    if args.dataset in ['tcga', 'all']:
        download_tcga_luad(os.path.join(args.data_dir, 'tcga_luad'))


if __name__ == "__main__":
    main()
