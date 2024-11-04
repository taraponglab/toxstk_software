import pandas as pd
import numpy as np
import os
from padelpy import padeldescriptor, from_smiles
from joblib import load
from glob import glob

""" 
This program contain 
1. Load data
2. Compute FP
3. Baseline predict
4. Stacked
5. AD
6. Show outcome
"""
def load_data(df_name):
    df = pd.read_csv(df_name+".csv", index_col=0)
    print(df)
    df["Smiles"].to_csv('smile.smi', sep='\t', index=False, header=False)
    return df


def compute_fps(df):
    'AP2DC','AD2D'
    padeldescriptor(mol_dir='smile.smi',
                d_file='AP2DC'+'.csv',
                descriptortypes= "AtomPairs2DFingerprintCount.xml",
                retainorder=True, 
                removesalt=True,
                threads=2,
                detectaromaticity=True,
                standardizetautomers=True,
                standardizenitro=True,
                fingerprints=True
                )
    Fingerprint = pd.read_csv("AP2DC"+'.csv').set_index(df.index)
    Fingerprint = Fingerprint.drop('Name', axis=1)
    Fingerprint.to_csv("AP2DC"+'.csv')
    print("AP2DC"+'.csv', 'done')
    
    padeldescriptor(mol_dir='smile.smi',
                d_file='AD2D'+'.csv',
                descriptortypes= "AtomPairs2DFingerprinter.xml",
                retainorder=True, 
                removesalt=True,
                threads=2,
                detectaromaticity=True,
                standardizetautomers=True,
                standardizenitro=True,
                fingerprints=True
                )
    Fingerprint = pd.read_csv("AD2D"+'.csv').set_index(df.index)
    Fingerprint = Fingerprint.drop('Name', axis=1)
    Fingerprint.to_csv("AD2D"+'.csv')
    print("AD2D"+'.csv', 'done')
    #load at pc
    fp_at = pd.read_csv('AD2D.csv'     ).set_index(df.index)
    fp_ac = pd.read_csv('AP2DC.csv'    ).set_index(df.index)
    fp_at.to_csv('AD2D.csv'     )
    fp_ac.to_csv('AP2DC.csv'    )
    return fp_at, fp_ac

def main():
    df_all = pd.read_csv("pbmcs_smiles.csv", index_col=0)
    print(len(df_all['canonical_smiles'].unique()))
    #df_train = pd.read_csv(os.path.join("classification", "pbmcs", "xat_train.csv"), index_col=0)
    #df_test  = pd.read_csv(os.path.join("classification", "pbmcs", "xat_test.csv"), index_col=0)
    #fp_at = pd.read_csv("AD2D.csv", index_col=0)
    #fp_at = fp_at.drop(columns='molecule_chembl_id.1')
    #fp_at_train = fp_at.loc[df_train.index]
    #fp_at_test  = fp_at.loc[df_test.index]
    #fp_at_train.to_csv("pbmcs_xat_train.csv") 
    #fp_at_test .to_csv("pbmcs_xat_test.csv")
    #
    #fp_ac = pd.read_csv("AP2DC.csv", index_col=0)
    #print(fp_ac)
    #fp_ac = fp_ac.drop(columns='molecule_chembl_id.1')
    #fp_ac_train = fp_ac.loc[df_train.index]
    #fp_ac_test  = fp_ac.loc[df_test.index]
    #fp_ac_train.to_csv("pbmcs_xac_train.csv") 
    #fp_ac_test .to_csv("pbmcs_xac_test.csv") 
    
    #df_train_smiles = df_all.loc[df_train.index]
    #df_all["canonical_smiles"].to_csv('smile.smi', sep='\t', index=False, header=False)
    #fp_at, fp_ac = compute_fps(df_train_smiles)
    #df_test_smiles = df_all.loc[df_train.index]
    #df_test_smiles["canonical_smiles"].to_csv('smile.smi', sep='\t', index=False, header=False)
    #fp_at, fp_ac = compute_fps(df_all)
    
if __name__ == "__main__":
    main()