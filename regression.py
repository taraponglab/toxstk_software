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
    xml_files = glob("*.xml")
    xml_files.sort()
    FP_list = [
    'AP2DC','AD2D','EState','CDKExt','CDK','CDKGraph','KRFPC','KRFP','MACCS','PubChem','SubFPC','SubFP']
    fp = dict(zip(FP_list, xml_files))
    print(fp)
    #Calculate fingerprints
    for i in FP_list:
        padeldescriptor(mol_dir='smile.smi',
                    d_file=i+'.csv',
                    descriptortypes= fp[i],
                    retainorder=True, 
                    removesalt=True,
                    threads=2,
                    detectaromaticity=True,
                    standardizetautomers=True,
                    standardizenitro=True,
                    fingerprints=True
                    )
        Fingerprint = pd.read_csv(i+'.csv').set_index(df.index)
        Fingerprint = Fingerprint.drop('Name', axis=1)
        Fingerprint.to_csv(i+'.csv')
        print(i+'.csv', 'done')
    #load at pc
    fp_at = pd.read_csv('AD2D.csv'     ).set_index(df.index)
    fp_es = pd.read_csv('EState.csv'   ).set_index(df.index)
    fp_ke = pd.read_csv('KRFP.csv'     ).set_index(df.index)
    fp_pc = pd.read_csv('PubChem.csv'  ).set_index(df.index)
    fp_ss = pd.read_csv('SubFP.csv'    ).set_index(df.index)
    fp_cd = pd.read_csv('CDKGraph.csv' ).set_index(df.index)
    fp_cn = pd.read_csv('CDK.csv'      ).set_index(df.index)
    fp_kc = pd.read_csv('KRFPC.csv'    ).set_index(df.index)
    fp_ce = pd.read_csv('CDKExt.csv'   ).set_index(df.index)
    fp_sc = pd.read_csv('SubFPC.csv'   ).set_index(df.index)
    fp_ac = pd.read_csv('AP2DC.csv'    ).set_index(df.index)
    fp_ma = pd.read_csv('MACCS.csv'    ).set_index(df.index)
    return fp_at, fp_es, fp_ke, fp_pc, fp_ss, fp_cd, fp_cn, fp_kc, fp_ce, fp_sc, fp_ac, fp_ma
#helper functions
def select_col(df, columns_list):
    missing_columns = [col for col in columns_list if col not in df.columns]
    if not missing_columns:
        return df[columns_list]
    else:
        return f"Column(s) missing: {', '.join(missing_columns)}"
def ad_measurement(name, df, model, dk, sk, z=0.5):
    distance, index = model.kneighbors(df)
    di = np.mean(distance, axis=1)
    AD_status = ['within_AD' if di[i] < dk + (z * sk) else 'outside_AD' for i in range(len(di))]
    df_ad = pd.DataFrame(AD_status, index=df.index, columns=['AD_'+ name])
    return df_ad
    
def herg_fp_sel(fp_at, fp_es, fp_ke, fp_pc, fp_ss, fp_cd, fp_cn, fp_kc, fp_ce, fp_sc, fp_ac, fp_ma):
    fp_at_sel = pd.read_csv(os.path.join("regression", "herg", "xat_train.csv"), index_col=0)
    fp_es_sel = pd.read_csv(os.path.join("regression", "herg", "xes_train.csv"), index_col=0)
    fp_ke_sel = pd.read_csv(os.path.join("regression", "herg", "xke_train.csv"), index_col=0)
    fp_pc_sel = pd.read_csv(os.path.join("regression", "herg", "xpc_train.csv"), index_col=0)
    fp_ss_sel = pd.read_csv(os.path.join("regression", "herg", "xss_train.csv"), index_col=0)
    fp_cd_sel = pd.read_csv(os.path.join("regression", "herg", "xcd_train.csv"), index_col=0)
    fp_cn_sel = pd.read_csv(os.path.join("regression", "herg", "xcn_train.csv"), index_col=0)
    fp_kc_sel = pd.read_csv(os.path.join("regression", "herg", "xkc_train.csv"), index_col=0)
    fp_ce_sel = pd.read_csv(os.path.join("regression", "herg", "xce_train.csv"), index_col=0)
    fp_sc_sel = pd.read_csv(os.path.join("regression", "herg", "xsc_train.csv"), index_col=0)
    fp_ac_sel = pd.read_csv(os.path.join("regression", "herg", "xac_train.csv"), index_col=0)
    fp_ma_sel = pd.read_csv(os.path.join("regression", "herg", "xma_train.csv"), index_col=0)
    herg_fp_at = select_col(fp_at, fp_at_sel.columns)
    herg_fp_es = select_col(fp_es, fp_es_sel.columns)
    herg_fp_ke = select_col(fp_ke, fp_ke_sel.columns)
    herg_fp_pc = select_col(fp_pc, fp_pc_sel.columns)
    herg_fp_ss = select_col(fp_ss, fp_ss_sel.columns)
    herg_fp_cd = select_col(fp_cd, fp_cd_sel.columns)
    herg_fp_cn = select_col(fp_cn, fp_cn_sel.columns)
    herg_fp_kc = select_col(fp_kc, fp_kc_sel.columns)
    herg_fp_ce = select_col(fp_ce, fp_ce_sel.columns)
    herg_fp_sc = select_col(fp_sc, fp_sc_sel.columns)
    herg_fp_ac = select_col(fp_ac, fp_ac_sel.columns)
    herg_fp_ma = select_col(fp_ma, fp_ma_sel.columns)
    
    return herg_fp_at, herg_fp_es, herg_fp_ke, herg_fp_pc, herg_fp_ss, herg_fp_cd, herg_fp_cn, herg_fp_kc, herg_fp_ce, herg_fp_sc, herg_fp_ac, herg_fp_ma

def mtor_fp_sel(fp_at, fp_es, fp_ke, fp_pc, fp_ss, fp_cd, fp_cn, fp_kc, fp_ce, fp_sc, fp_ac, fp_ma):
    fp_at_sel = pd.read_csv(os.path.join("regression", "mtor", "xat_train.csv"), index_col=0)
    fp_es_sel = pd.read_csv(os.path.join("regression", "mtor", "xes_train.csv"), index_col=0)
    fp_ke_sel = pd.read_csv(os.path.join("regression", "mtor", "xke_train.csv"), index_col=0)
    fp_pc_sel = pd.read_csv(os.path.join("regression", "mtor", "xpc_train.csv"), index_col=0)
    fp_ss_sel = pd.read_csv(os.path.join("regression", "mtor", "xss_train.csv"), index_col=0)
    fp_cd_sel = pd.read_csv(os.path.join("regression", "mtor", "xcd_train.csv"), index_col=0)
    fp_cn_sel = pd.read_csv(os.path.join("regression", "mtor", "xcn_train.csv"), index_col=0)
    fp_kc_sel = pd.read_csv(os.path.join("regression", "mtor", "xkc_train.csv"), index_col=0)
    fp_ce_sel = pd.read_csv(os.path.join("regression", "mtor", "xce_train.csv"), index_col=0)
    fp_sc_sel = pd.read_csv(os.path.join("regression", "mtor", "xsc_train.csv"), index_col=0)
    fp_ac_sel = pd.read_csv(os.path.join("regression", "mtor", "xac_train.csv"), index_col=0)
    fp_ma_sel = pd.read_csv(os.path.join("regression", "mtor", "xma_train.csv"), index_col=0)
    mtor_fp_at = select_col(fp_at, fp_at_sel.columns)
    mtor_fp_es = select_col(fp_es, fp_es_sel.columns)
    mtor_fp_ke = select_col(fp_ke, fp_ke_sel.columns)
    mtor_fp_pc = select_col(fp_pc, fp_pc_sel.columns)
    mtor_fp_ss = select_col(fp_ss, fp_ss_sel.columns)
    mtor_fp_cd = select_col(fp_cd, fp_cd_sel.columns)
    mtor_fp_cn = select_col(fp_cn, fp_cn_sel.columns)
    mtor_fp_kc = select_col(fp_kc, fp_kc_sel.columns)
    mtor_fp_ce = select_col(fp_ce, fp_ce_sel.columns)
    mtor_fp_sc = select_col(fp_sc, fp_sc_sel.columns)
    mtor_fp_ac = select_col(fp_ac, fp_ac_sel.columns)
    mtor_fp_ma = select_col(fp_ma, fp_ma_sel.columns)
    return mtor_fp_at, mtor_fp_es, mtor_fp_ke, mtor_fp_pc, mtor_fp_ss, mtor_fp_cd, mtor_fp_cn, mtor_fp_kc, mtor_fp_ce, mtor_fp_sc, mtor_fp_ac, mtor_fp_ma

def pbmcs_fp_sel(fp_at, fp_es, fp_ke, fp_pc, fp_ss, fp_cd, fp_cn, fp_kc, fp_ce, fp_sc, fp_ac, fp_ma):
    fp_at_sel = pd.read_csv(os.path.join("regression", "pbmcs", "xat_train.csv"), index_col=0)
    fp_es_sel = pd.read_csv(os.path.join("regression", "pbmcs", "xes_train.csv"), index_col=0)
    fp_ke_sel = pd.read_csv(os.path.join("regression", "pbmcs", "xke_train.csv"), index_col=0)
    fp_pc_sel = pd.read_csv(os.path.join("regression", "pbmcs", "xpc_train.csv"), index_col=0)
    fp_ss_sel = pd.read_csv(os.path.join("regression", "pbmcs", "xss_train.csv"), index_col=0)
    fp_cd_sel = pd.read_csv(os.path.join("regression", "pbmcs", "xcd_train.csv"), index_col=0)
    fp_cn_sel = pd.read_csv(os.path.join("regression", "pbmcs", "xcn_train.csv"), index_col=0)
    fp_kc_sel = pd.read_csv(os.path.join("regression", "pbmcs", "xkc_train.csv"), index_col=0)
    fp_ce_sel = pd.read_csv(os.path.join("regression", "pbmcs", "xce_train.csv"), index_col=0)
    fp_sc_sel = pd.read_csv(os.path.join("regression", "pbmcs", "xsc_train.csv"), index_col=0)
    fp_ac_sel = pd.read_csv(os.path.join("regression", "pbmcs", "xac_train.csv"), index_col=0)
    fp_ma_sel = pd.read_csv(os.path.join("regression", "pbmcs", "xma_train.csv"), index_col=0)
    pbmcs_fp_at = select_col(fp_at, fp_at_sel.columns)
    pbmcs_fp_es = select_col(fp_es, fp_es_sel.columns)
    pbmcs_fp_ke = select_col(fp_ke, fp_ke_sel.columns)
    pbmcs_fp_pc = select_col(fp_pc, fp_pc_sel.columns)
    pbmcs_fp_ss = select_col(fp_ss, fp_ss_sel.columns)
    pbmcs_fp_cd = select_col(fp_cd, fp_cd_sel.columns)
    pbmcs_fp_cn = select_col(fp_cn, fp_cn_sel.columns)
    pbmcs_fp_kc = select_col(fp_kc, fp_kc_sel.columns)
    pbmcs_fp_ce = select_col(fp_ce, fp_ce_sel.columns)
    pbmcs_fp_sc = select_col(fp_sc, fp_sc_sel.columns)
    pbmcs_fp_ac = select_col(fp_ac, fp_ac_sel.columns)
    pbmcs_fp_ma = select_col(fp_ma, fp_ma_sel.columns)
    return pbmcs_fp_at, pbmcs_fp_es, pbmcs_fp_ke, pbmcs_fp_pc, pbmcs_fp_ss, pbmcs_fp_cd, pbmcs_fp_cn, pbmcs_fp_kc, pbmcs_fp_ce, pbmcs_fp_sc, pbmcs_fp_ac, pbmcs_fp_ma

def stacked_herg(herg_fp_at, herg_fp_es, herg_fp_ke, herg_fp_pc, herg_fp_ss, herg_fp_cd, herg_fp_cn, herg_fp_kc, herg_fp_ce, herg_fp_sc, herg_fp_ac, herg_fp_ma):
    rf_at = load(os.path.join("regression", "herg", "baseline_model_rf_at.joblib"))
    rf_es = load(os.path.join("regression", "herg", "baseline_model_rf_es.joblib"))
    rf_ke = load(os.path.join("regression", "herg", "baseline_model_rf_ke.joblib"))
    rf_pc = load(os.path.join("regression", "herg", "baseline_model_rf_pc.joblib"))
    rf_ss = load(os.path.join("regression", "herg", "baseline_model_rf_ss.joblib"))
    rf_cd = load(os.path.join("regression", "herg", "baseline_model_rf_cd.joblib"))
    rf_cn = load(os.path.join("regression", "herg", "baseline_model_rf_cn.joblib"))
    rf_kc = load(os.path.join("regression", "herg", "baseline_model_rf_kc.joblib"))
    rf_ce = load(os.path.join("regression", "herg", "baseline_model_rf_ce.joblib"))
    rf_sc = load(os.path.join("regression", "herg", "baseline_model_rf_sc.joblib"))
    rf_ac = load(os.path.join("regression", "herg", "baseline_model_rf_ac.joblib"))
    rf_ma = load(os.path.join("regression", "herg", "baseline_model_rf_ma.joblib"))
    yat_pred_rf_test = pd.DataFrame(rf_at.predict(herg_fp_at), columns=["yat_pred_rf"]).set_index(herg_fp_at.index)
    yes_pred_rf_test = pd.DataFrame(rf_es.predict(herg_fp_es), columns=["yes_pred_rf"]).set_index(herg_fp_es.index)
    yke_pred_rf_test = pd.DataFrame(rf_ke.predict(herg_fp_ke), columns=["yke_pred_rf"]).set_index(herg_fp_ke.index)
    ypc_pred_rf_test = pd.DataFrame(rf_pc.predict(herg_fp_pc), columns=["ypc_pred_rf"]).set_index(herg_fp_pc.index)
    yss_pred_rf_test = pd.DataFrame(rf_ss.predict(herg_fp_ss), columns=["yss_pred_rf"]).set_index(herg_fp_ss.index)
    ycd_pred_rf_test = pd.DataFrame(rf_cd.predict(herg_fp_cd), columns=["ycd_pred_rf"]).set_index(herg_fp_cd.index)
    ycn_pred_rf_test = pd.DataFrame(rf_cn.predict(herg_fp_cn), columns=["ycn_pred_rf"]).set_index(herg_fp_cn.index)
    ykc_pred_rf_test = pd.DataFrame(rf_kc.predict(herg_fp_kc), columns=["ykc_pred_rf"]).set_index(herg_fp_kc.index)
    yce_pred_rf_test = pd.DataFrame(rf_ce.predict(herg_fp_ce), columns=["yce_pred_rf"]).set_index(herg_fp_ce.index)
    ysc_pred_rf_test = pd.DataFrame(rf_sc.predict(herg_fp_sc), columns=["ysc_pred_rf"]).set_index(herg_fp_sc.index)
    yac_pred_rf_test = pd.DataFrame(rf_ac.predict(herg_fp_ac), columns=["yac_pred_rf"]).set_index(herg_fp_ac.index)
    yma_pred_rf_test = pd.DataFrame(rf_ma.predict(herg_fp_ma), columns=["yma_pred_rf"]).set_index(herg_fp_ma.index)
    xgb_at = load(os.path.join("regression", "herg", "baseline_model_xgb_at.joblib"))
    xgb_es = load(os.path.join("regression", "herg", "baseline_model_xgb_es.joblib"))
    xgb_ke = load(os.path.join("regression", "herg", "baseline_model_xgb_ke.joblib"))
    xgb_pc = load(os.path.join("regression", "herg", "baseline_model_xgb_pc.joblib"))
    xgb_ss = load(os.path.join("regression", "herg", "baseline_model_xgb_ss.joblib"))
    xgb_cd = load(os.path.join("regression", "herg", "baseline_model_xgb_cd.joblib"))
    xgb_cn = load(os.path.join("regression", "herg", "baseline_model_xgb_cn.joblib"))
    xgb_kc = load(os.path.join("regression", "herg", "baseline_model_xgb_kc.joblib"))
    xgb_ce = load(os.path.join("regression", "herg", "baseline_model_xgb_ce.joblib"))
    xgb_sc = load(os.path.join("regression", "herg", "baseline_model_xgb_sc.joblib"))
    xgb_ac = load(os.path.join("regression", "herg", "baseline_model_xgb_ac.joblib"))
    xgb_ma = load(os.path.join("regression", "herg", "baseline_model_xgb_ma.joblib"))
    yat_pred_xgb_test = pd.DataFrame(xgb_at.predict(herg_fp_at), columns=["yat_pred_xgb"]).set_index(herg_fp_at.index)
    yes_pred_xgb_test = pd.DataFrame(xgb_es.predict(herg_fp_es), columns=["yes_pred_xgb"]).set_index(herg_fp_es.index)
    yke_pred_xgb_test = pd.DataFrame(xgb_ke.predict(herg_fp_ke), columns=["yke_pred_xgb"]).set_index(herg_fp_ke.index)
    ypc_pred_xgb_test = pd.DataFrame(xgb_pc.predict(herg_fp_pc), columns=["ypc_pred_xgb"]).set_index(herg_fp_pc.index)
    yss_pred_xgb_test = pd.DataFrame(xgb_ss.predict(herg_fp_ss), columns=["yss_pred_xgb"]).set_index(herg_fp_ss.index)
    ycd_pred_xgb_test = pd.DataFrame(xgb_cd.predict(herg_fp_cd), columns=["ycd_pred_xgb"]).set_index(herg_fp_cd.index)
    ycn_pred_xgb_test = pd.DataFrame(xgb_cn.predict(herg_fp_cn), columns=["ycn_pred_xgb"]).set_index(herg_fp_cn.index)
    ykc_pred_xgb_test = pd.DataFrame(xgb_kc.predict(herg_fp_kc), columns=["ykc_pred_xgb"]).set_index(herg_fp_kc.index)
    yce_pred_xgb_test = pd.DataFrame(xgb_ce.predict(herg_fp_ce), columns=["yce_pred_xgb"]).set_index(herg_fp_ce.index)
    ysc_pred_xgb_test = pd.DataFrame(xgb_sc.predict(herg_fp_sc), columns=["ysc_pred_xgb"]).set_index(herg_fp_sc.index)
    yac_pred_xgb_test = pd.DataFrame(xgb_ac.predict(herg_fp_ac), columns=["yac_pred_xgb"]).set_index(herg_fp_ac.index)
    yma_pred_xgb_test = pd.DataFrame(xgb_ma.predict(herg_fp_ma), columns=["yma_pred_xgb"]).set_index(herg_fp_ma.index)
    svm_at = load(os.path.join("regression", "herg", "baseline_model_svc_at.joblib"))
    svm_es = load(os.path.join("regression", "herg", "baseline_model_svc_es.joblib"))
    svm_ke = load(os.path.join("regression", "herg", "baseline_model_svc_ke.joblib"))
    svm_pc = load(os.path.join("regression", "herg", "baseline_model_svc_pc.joblib"))
    svm_ss = load(os.path.join("regression", "herg", "baseline_model_svc_ss.joblib"))
    svm_cd = load(os.path.join("regression", "herg", "baseline_model_svc_cd.joblib"))
    svm_cn = load(os.path.join("regression", "herg", "baseline_model_svc_cn.joblib"))
    svm_kc = load(os.path.join("regression", "herg", "baseline_model_svc_kc.joblib"))
    svm_ce = load(os.path.join("regression", "herg", "baseline_model_svc_ce.joblib"))
    svm_sc = load(os.path.join("regression", "herg", "baseline_model_svc_sc.joblib"))
    svm_ac = load(os.path.join("regression", "herg", "baseline_model_svc_ac.joblib"))
    svm_ma = load(os.path.join("regression", "herg", "baseline_model_svc_ma.joblib"))
    yat_pred_svc_test = pd.DataFrame(svm_at.predict(herg_fp_at), columns=["yat_pred_svr"]).set_index(herg_fp_at.index)
    yes_pred_svc_test = pd.DataFrame(svm_es.predict(herg_fp_es), columns=["yes_pred_svr"]).set_index(herg_fp_es.index)
    yke_pred_svc_test = pd.DataFrame(svm_ke.predict(herg_fp_ke), columns=["yke_pred_svr"]).set_index(herg_fp_ke.index)
    ypc_pred_svc_test = pd.DataFrame(svm_pc.predict(herg_fp_pc), columns=["ypc_pred_svr"]).set_index(herg_fp_pc.index)
    yss_pred_svc_test = pd.DataFrame(svm_ss.predict(herg_fp_ss), columns=["yss_pred_svr"]).set_index(herg_fp_ss.index)
    ycd_pred_svc_test = pd.DataFrame(svm_cd.predict(herg_fp_cd), columns=["ycd_pred_svr"]).set_index(herg_fp_cd.index)
    ycn_pred_svc_test = pd.DataFrame(svm_cn.predict(herg_fp_cn), columns=["ycn_pred_svr"]).set_index(herg_fp_cn.index)
    ykc_pred_svc_test = pd.DataFrame(svm_kc.predict(herg_fp_kc), columns=["ykc_pred_svr"]).set_index(herg_fp_kc.index)
    yce_pred_svc_test = pd.DataFrame(svm_ce.predict(herg_fp_ce), columns=["yce_pred_svr"]).set_index(herg_fp_ce.index)
    ysc_pred_svc_test = pd.DataFrame(svm_sc.predict(herg_fp_sc), columns=["ysc_pred_svr"]).set_index(herg_fp_sc.index)
    yac_pred_svc_test = pd.DataFrame(svm_ac.predict(herg_fp_ac), columns=["yac_pred_svr"]).set_index(herg_fp_ac.index)
    yma_pred_svc_test = pd.DataFrame(svm_ma.predict(herg_fp_ma), columns=["yma_pred_svr"]).set_index(herg_fp_ma.index)
    stack_test  = pd.concat([yat_pred_rf_test, yat_pred_xgb_test, yat_pred_svc_test,
                            yes_pred_rf_test, yes_pred_xgb_test, yes_pred_svc_test,
                            yke_pred_rf_test, yke_pred_xgb_test, yke_pred_svc_test,
                            ypc_pred_rf_test, ypc_pred_xgb_test, ypc_pred_svc_test,
                            yss_pred_rf_test, yss_pred_xgb_test, yss_pred_svc_test,
                            ycd_pred_rf_test, ycd_pred_xgb_test, ycd_pred_svc_test,
                            ycn_pred_rf_test, ycn_pred_xgb_test, ycn_pred_svc_test,
                            ykc_pred_rf_test, ykc_pred_xgb_test, ykc_pred_svc_test,
                            yce_pred_rf_test, yce_pred_xgb_test, yce_pred_svc_test,
                            ysc_pred_rf_test, ysc_pred_xgb_test, ysc_pred_svc_test,
                            yac_pred_rf_test, yac_pred_xgb_test, yac_pred_svc_test,
                            yma_pred_rf_test, yma_pred_xgb_test, yma_pred_svc_test,],  axis=1)
    stacked_model = load(os.path.join("regression", "herg", "stacked_model.joblib"))
    y_pred = pd.DataFrame(stacked_model.predict(stack_test), columns=["hERG_pIC50"]).set_index(herg_fp_at.index)
    ad_model = load(os.path.join("regression", "herg", "ad_1_0.5.joblib"))
    y_ad = ad_measurement("hERG", stack_test, ad_model, 2.0574, 0.6952, z=0.5)
    y_pred = pd.concat([y_pred, y_ad], axis=1)
    return y_pred

def stacked_mtor(mtor_fp_at, mtor_fp_es, mtor_fp_ke, mtor_fp_pc, mtor_fp_ss, mtor_fp_cd, mtor_fp_cn, mtor_fp_kc, mtor_fp_ce, mtor_fp_sc, mtor_fp_ac, mtor_fp_ma):
    rf_at = load(os.path.join("regression", "mtor", "baseline_model_rf_at.joblib"))
    rf_es = load(os.path.join("regression", "mtor", "baseline_model_rf_es.joblib"))
    rf_ke = load(os.path.join("regression", "mtor", "baseline_model_rf_ke.joblib"))
    rf_pc = load(os.path.join("regression", "mtor", "baseline_model_rf_pc.joblib"))
    rf_ss = load(os.path.join("regression", "mtor", "baseline_model_rf_ss.joblib"))
    rf_cd = load(os.path.join("regression", "mtor", "baseline_model_rf_cd.joblib"))
    rf_cn = load(os.path.join("regression", "mtor", "baseline_model_rf_cn.joblib"))
    rf_kc = load(os.path.join("regression", "mtor", "baseline_model_rf_kc.joblib"))
    rf_ce = load(os.path.join("regression", "mtor", "baseline_model_rf_ce.joblib"))
    rf_sc = load(os.path.join("regression", "mtor", "baseline_model_rf_sc.joblib"))
    rf_ac = load(os.path.join("regression", "mtor", "baseline_model_rf_ac.joblib"))
    rf_ma = load(os.path.join("regression", "mtor", "baseline_model_rf_ma.joblib"))
    yat_pred_rf_test = pd.DataFrame(rf_at.predict(mtor_fp_at), columns=["yat_pred_rf"]).set_index(mtor_fp_at.index)
    yes_pred_rf_test = pd.DataFrame(rf_es.predict(mtor_fp_es), columns=["yes_pred_rf"]).set_index(mtor_fp_es.index)
    yke_pred_rf_test = pd.DataFrame(rf_ke.predict(mtor_fp_ke), columns=["yke_pred_rf"]).set_index(mtor_fp_ke.index)
    ypc_pred_rf_test = pd.DataFrame(rf_pc.predict(mtor_fp_pc), columns=["ypc_pred_rf"]).set_index(mtor_fp_pc.index)
    yss_pred_rf_test = pd.DataFrame(rf_ss.predict(mtor_fp_ss), columns=["yss_pred_rf"]).set_index(mtor_fp_ss.index)
    ycd_pred_rf_test = pd.DataFrame(rf_cd.predict(mtor_fp_cd), columns=["ycd_pred_rf"]).set_index(mtor_fp_cd.index)
    ycn_pred_rf_test = pd.DataFrame(rf_cn.predict(mtor_fp_cn), columns=["ycn_pred_rf"]).set_index(mtor_fp_cn.index)
    ykc_pred_rf_test = pd.DataFrame(rf_kc.predict(mtor_fp_kc), columns=["ykc_pred_rf"]).set_index(mtor_fp_kc.index)
    yce_pred_rf_test = pd.DataFrame(rf_ce.predict(mtor_fp_ce), columns=["yce_pred_rf"]).set_index(mtor_fp_ce.index)
    ysc_pred_rf_test = pd.DataFrame(rf_sc.predict(mtor_fp_sc), columns=["ysc_pred_rf"]).set_index(mtor_fp_sc.index)
    yac_pred_rf_test = pd.DataFrame(rf_ac.predict(mtor_fp_ac), columns=["yac_pred_rf"]).set_index(mtor_fp_ac.index)
    yma_pred_rf_test = pd.DataFrame(rf_ma.predict(mtor_fp_ma), columns=["yma_pred_rf"]).set_index(mtor_fp_ma.index)
    xgb_at = load(os.path.join("regression", "mtor", "baseline_model_xgb_at.joblib"))
    xgb_es = load(os.path.join("regression", "mtor", "baseline_model_xgb_es.joblib"))
    xgb_ke = load(os.path.join("regression", "mtor", "baseline_model_xgb_ke.joblib"))
    xgb_pc = load(os.path.join("regression", "mtor", "baseline_model_xgb_pc.joblib"))
    xgb_ss = load(os.path.join("regression", "mtor", "baseline_model_xgb_ss.joblib"))
    xgb_cd = load(os.path.join("regression", "mtor", "baseline_model_xgb_cd.joblib"))
    xgb_cn = load(os.path.join("regression", "mtor", "baseline_model_xgb_cn.joblib"))
    xgb_kc = load(os.path.join("regression", "mtor", "baseline_model_xgb_kc.joblib"))
    xgb_ce = load(os.path.join("regression", "mtor", "baseline_model_xgb_ce.joblib"))
    xgb_sc = load(os.path.join("regression", "mtor", "baseline_model_xgb_sc.joblib"))
    xgb_ac = load(os.path.join("regression", "mtor", "baseline_model_xgb_ac.joblib"))
    xgb_ma = load(os.path.join("regression", "mtor", "baseline_model_xgb_ma.joblib"))
    yat_pred_xgb_test = pd.DataFrame(xgb_at.predict(mtor_fp_at), columns=["yat_pred_xgb"]).set_index(mtor_fp_at.index)
    yes_pred_xgb_test = pd.DataFrame(xgb_es.predict(mtor_fp_es), columns=["yes_pred_xgb"]).set_index(mtor_fp_es.index)
    yke_pred_xgb_test = pd.DataFrame(xgb_ke.predict(mtor_fp_ke), columns=["yke_pred_xgb"]).set_index(mtor_fp_ke.index)
    ypc_pred_xgb_test = pd.DataFrame(xgb_pc.predict(mtor_fp_pc), columns=["ypc_pred_xgb"]).set_index(mtor_fp_pc.index)
    yss_pred_xgb_test = pd.DataFrame(xgb_ss.predict(mtor_fp_ss), columns=["yss_pred_xgb"]).set_index(mtor_fp_ss.index)
    ycd_pred_xgb_test = pd.DataFrame(xgb_cd.predict(mtor_fp_cd), columns=["ycd_pred_xgb"]).set_index(mtor_fp_cd.index)
    ycn_pred_xgb_test = pd.DataFrame(xgb_cn.predict(mtor_fp_cn), columns=["ycn_pred_xgb"]).set_index(mtor_fp_cn.index)
    ykc_pred_xgb_test = pd.DataFrame(xgb_kc.predict(mtor_fp_kc), columns=["ykc_pred_xgb"]).set_index(mtor_fp_kc.index)
    yce_pred_xgb_test = pd.DataFrame(xgb_ce.predict(mtor_fp_ce), columns=["yce_pred_xgb"]).set_index(mtor_fp_ce.index)
    ysc_pred_xgb_test = pd.DataFrame(xgb_sc.predict(mtor_fp_sc), columns=["ysc_pred_xgb"]).set_index(mtor_fp_sc.index)
    yac_pred_xgb_test = pd.DataFrame(xgb_ac.predict(mtor_fp_ac), columns=["yac_pred_xgb"]).set_index(mtor_fp_ac.index)
    yma_pred_xgb_test = pd.DataFrame(xgb_ma.predict(mtor_fp_ma), columns=["yma_pred_xgb"]).set_index(mtor_fp_ma.index)
    svm_at = load(os.path.join("regression", "mtor", "baseline_model_svc_at.joblib"))
    svm_es = load(os.path.join("regression", "mtor", "baseline_model_svc_es.joblib"))
    svm_ke = load(os.path.join("regression", "mtor", "baseline_model_svc_ke.joblib"))
    svm_pc = load(os.path.join("regression", "mtor", "baseline_model_svc_pc.joblib"))
    svm_ss = load(os.path.join("regression", "mtor", "baseline_model_svc_ss.joblib"))
    svm_cd = load(os.path.join("regression", "mtor", "baseline_model_svc_cd.joblib"))
    svm_cn = load(os.path.join("regression", "mtor", "baseline_model_svc_cn.joblib"))
    svm_kc = load(os.path.join("regression", "mtor", "baseline_model_svc_kc.joblib"))
    svm_ce = load(os.path.join("regression", "mtor", "baseline_model_svc_ce.joblib"))
    svm_sc = load(os.path.join("regression", "mtor", "baseline_model_svc_sc.joblib"))
    svm_ac = load(os.path.join("regression", "mtor", "baseline_model_svc_ac.joblib"))
    svm_ma = load(os.path.join("regression", "mtor", "baseline_model_svc_ma.joblib"))
    yat_pred_svc_test = pd.DataFrame(svm_at.predict(mtor_fp_at), columns=["yat_pred_svr"]).set_index(mtor_fp_at.index)
    yes_pred_svc_test = pd.DataFrame(svm_es.predict(mtor_fp_es), columns=["yes_pred_svr"]).set_index(mtor_fp_es.index)
    yke_pred_svc_test = pd.DataFrame(svm_ke.predict(mtor_fp_ke), columns=["yke_pred_svr"]).set_index(mtor_fp_ke.index)
    ypc_pred_svc_test = pd.DataFrame(svm_pc.predict(mtor_fp_pc), columns=["ypc_pred_svr"]).set_index(mtor_fp_pc.index)
    yss_pred_svc_test = pd.DataFrame(svm_ss.predict(mtor_fp_ss), columns=["yss_pred_svr"]).set_index(mtor_fp_ss.index)
    ycd_pred_svc_test = pd.DataFrame(svm_cd.predict(mtor_fp_cd), columns=["ycd_pred_svr"]).set_index(mtor_fp_cd.index)
    ycn_pred_svc_test = pd.DataFrame(svm_cn.predict(mtor_fp_cn), columns=["ycn_pred_svr"]).set_index(mtor_fp_cn.index)
    ykc_pred_svc_test = pd.DataFrame(svm_kc.predict(mtor_fp_kc), columns=["ykc_pred_svr"]).set_index(mtor_fp_kc.index)
    yce_pred_svc_test = pd.DataFrame(svm_ce.predict(mtor_fp_ce), columns=["yce_pred_svr"]).set_index(mtor_fp_ce.index)
    ysc_pred_svc_test = pd.DataFrame(svm_sc.predict(mtor_fp_sc), columns=["ysc_pred_svr"]).set_index(mtor_fp_sc.index)
    yac_pred_svc_test = pd.DataFrame(svm_ac.predict(mtor_fp_ac), columns=["yac_pred_svr"]).set_index(mtor_fp_ac.index)
    yma_pred_svc_test = pd.DataFrame(svm_ma.predict(mtor_fp_ma), columns=["yma_pred_svr"]).set_index(mtor_fp_ma.index)
    stack_test  = pd.concat([yat_pred_rf_test, yat_pred_xgb_test, yat_pred_svc_test,
                            yes_pred_rf_test, yes_pred_xgb_test, yes_pred_svc_test,
                            yke_pred_rf_test, yke_pred_xgb_test, yke_pred_svc_test,
                            ypc_pred_rf_test, ypc_pred_xgb_test, ypc_pred_svc_test,
                            yss_pred_rf_test, yss_pred_xgb_test, yss_pred_svc_test,
                            ycd_pred_rf_test, ycd_pred_xgb_test, ycd_pred_svc_test,
                            ycn_pred_rf_test, ycn_pred_xgb_test, ycn_pred_svc_test,
                            ykc_pred_rf_test, ykc_pred_xgb_test, ykc_pred_svc_test,
                            yce_pred_rf_test, yce_pred_xgb_test, yce_pred_svc_test,
                            ysc_pred_rf_test, ysc_pred_xgb_test, ysc_pred_svc_test,
                            yac_pred_rf_test, yac_pred_xgb_test, yac_pred_svc_test,
                            yma_pred_rf_test, yma_pred_xgb_test, yma_pred_svc_test,],  axis=1)
    stacked_model = load(os.path.join("regression", "mtor", "stacked_model.joblib"))
    y_pred = pd.DataFrame(stacked_model.predict(stack_test), columns=["mTOR_pIC50"]).set_index(mtor_fp_at.index)
    ad_model = load(os.path.join("regression", "mtor", "ad_1_3.joblib"))
    y_ad = ad_measurement("mTOR", stack_test, ad_model, 1.4391, 0.5930, z=3)
    y_pred = pd.concat([y_pred, y_ad], axis=1)
    return y_pred

def stacked_pbmcs(pbmcs_fp_at, pbmcs_fp_es, pbmcs_fp_ke, pbmcs_fp_pc, pbmcs_fp_ss, pbmcs_fp_cd, pbmcs_fp_cn, pbmcs_fp_kc, pbmcs_fp_ce, pbmcs_fp_sc, pbmcs_fp_ac, pbmcs_fp_ma):
    rf_at = load(os.path.join("regression", "pbmcs", "baseline_model_rf_at.joblib"))
    rf_es = load(os.path.join("regression", "pbmcs", "baseline_model_rf_es.joblib"))
    rf_ke = load(os.path.join("regression", "pbmcs", "baseline_model_rf_ke.joblib"))
    rf_pc = load(os.path.join("regression", "pbmcs", "baseline_model_rf_pc.joblib"))
    rf_ss = load(os.path.join("regression", "pbmcs", "baseline_model_rf_ss.joblib"))
    rf_cd = load(os.path.join("regression", "pbmcs", "baseline_model_rf_cd.joblib"))
    rf_cn = load(os.path.join("regression", "pbmcs", "baseline_model_rf_cn.joblib"))
    rf_kc = load(os.path.join("regression", "pbmcs", "baseline_model_rf_kc.joblib"))
    rf_ce = load(os.path.join("regression", "pbmcs", "baseline_model_rf_ce.joblib"))
    rf_sc = load(os.path.join("regression", "pbmcs", "baseline_model_rf_sc.joblib"))
    rf_ac = load(os.path.join("regression", "pbmcs", "baseline_model_rf_ac.joblib"))
    rf_ma = load(os.path.join("regression", "pbmcs", "baseline_model_rf_ma.joblib"))
    yat_pred_rf_test = pd.DataFrame(rf_at.predict(pbmcs_fp_at), columns=["yat_pred_rf"]).set_index(pbmcs_fp_at.index)
    yes_pred_rf_test = pd.DataFrame(rf_es.predict(pbmcs_fp_es), columns=["yes_pred_rf"]).set_index(pbmcs_fp_es.index)
    yke_pred_rf_test = pd.DataFrame(rf_ke.predict(pbmcs_fp_ke), columns=["yke_pred_rf"]).set_index(pbmcs_fp_ke.index)
    ypc_pred_rf_test = pd.DataFrame(rf_pc.predict(pbmcs_fp_pc), columns=["ypc_pred_rf"]).set_index(pbmcs_fp_pc.index)
    yss_pred_rf_test = pd.DataFrame(rf_ss.predict(pbmcs_fp_ss), columns=["yss_pred_rf"]).set_index(pbmcs_fp_ss.index)
    ycd_pred_rf_test = pd.DataFrame(rf_cd.predict(pbmcs_fp_cd), columns=["ycd_pred_rf"]).set_index(pbmcs_fp_cd.index)
    ycn_pred_rf_test = pd.DataFrame(rf_cn.predict(pbmcs_fp_cn), columns=["ycn_pred_rf"]).set_index(pbmcs_fp_cn.index)
    ykc_pred_rf_test = pd.DataFrame(rf_kc.predict(pbmcs_fp_kc), columns=["ykc_pred_rf"]).set_index(pbmcs_fp_kc.index)
    yce_pred_rf_test = pd.DataFrame(rf_ce.predict(pbmcs_fp_ce), columns=["yce_pred_rf"]).set_index(pbmcs_fp_ce.index)
    ysc_pred_rf_test = pd.DataFrame(rf_sc.predict(pbmcs_fp_sc), columns=["ysc_pred_rf"]).set_index(pbmcs_fp_sc.index)
    yac_pred_rf_test = pd.DataFrame(rf_ac.predict(pbmcs_fp_ac), columns=["yac_pred_rf"]).set_index(pbmcs_fp_ac.index)
    yma_pred_rf_test = pd.DataFrame(rf_ma.predict(pbmcs_fp_ma), columns=["yma_pred_rf"]).set_index(pbmcs_fp_ma.index)
    xgb_at = load(os.path.join("regression", "pbmcs", "baseline_model_xgb_at.joblib"))
    xgb_es = load(os.path.join("regression", "pbmcs", "baseline_model_xgb_es.joblib"))
    xgb_ke = load(os.path.join("regression", "pbmcs", "baseline_model_xgb_ke.joblib"))
    xgb_pc = load(os.path.join("regression", "pbmcs", "baseline_model_xgb_pc.joblib"))
    xgb_ss = load(os.path.join("regression", "pbmcs", "baseline_model_xgb_ss.joblib"))
    xgb_cd = load(os.path.join("regression", "pbmcs", "baseline_model_xgb_cd.joblib"))
    xgb_cn = load(os.path.join("regression", "pbmcs", "baseline_model_xgb_cn.joblib"))
    xgb_kc = load(os.path.join("regression", "pbmcs", "baseline_model_xgb_kc.joblib"))
    xgb_ce = load(os.path.join("regression", "pbmcs", "baseline_model_xgb_ce.joblib"))
    xgb_sc = load(os.path.join("regression", "pbmcs", "baseline_model_xgb_sc.joblib"))
    xgb_ac = load(os.path.join("regression", "pbmcs", "baseline_model_xgb_ac.joblib"))
    xgb_ma = load(os.path.join("regression", "pbmcs", "baseline_model_xgb_ma.joblib"))
    yat_pred_xgb_test = pd.DataFrame(xgb_at.predict(pbmcs_fp_at), columns=["yat_pred_xgb"]).set_index(pbmcs_fp_at.index)
    yes_pred_xgb_test = pd.DataFrame(xgb_es.predict(pbmcs_fp_es), columns=["yes_pred_xgb"]).set_index(pbmcs_fp_es.index)
    yke_pred_xgb_test = pd.DataFrame(xgb_ke.predict(pbmcs_fp_ke), columns=["yke_pred_xgb"]).set_index(pbmcs_fp_ke.index)
    ypc_pred_xgb_test = pd.DataFrame(xgb_pc.predict(pbmcs_fp_pc), columns=["ypc_pred_xgb"]).set_index(pbmcs_fp_pc.index)
    yss_pred_xgb_test = pd.DataFrame(xgb_ss.predict(pbmcs_fp_ss), columns=["yss_pred_xgb"]).set_index(pbmcs_fp_ss.index)
    ycd_pred_xgb_test = pd.DataFrame(xgb_cd.predict(pbmcs_fp_cd), columns=["ycd_pred_xgb"]).set_index(pbmcs_fp_cd.index)
    ycn_pred_xgb_test = pd.DataFrame(xgb_cn.predict(pbmcs_fp_cn), columns=["ycn_pred_xgb"]).set_index(pbmcs_fp_cn.index)
    ykc_pred_xgb_test = pd.DataFrame(xgb_kc.predict(pbmcs_fp_kc), columns=["ykc_pred_xgb"]).set_index(pbmcs_fp_kc.index)
    yce_pred_xgb_test = pd.DataFrame(xgb_ce.predict(pbmcs_fp_ce), columns=["yce_pred_xgb"]).set_index(pbmcs_fp_ce.index)
    ysc_pred_xgb_test = pd.DataFrame(xgb_sc.predict(pbmcs_fp_sc), columns=["ysc_pred_xgb"]).set_index(pbmcs_fp_sc.index)
    yac_pred_xgb_test = pd.DataFrame(xgb_ac.predict(pbmcs_fp_ac), columns=["yac_pred_xgb"]).set_index(pbmcs_fp_ac.index)
    yma_pred_xgb_test = pd.DataFrame(xgb_ma.predict(pbmcs_fp_ma), columns=["yma_pred_xgb"]).set_index(pbmcs_fp_ma.index)
    svm_at = load(os.path.join("regression", "pbmcs", "baseline_model_svc_at.joblib"))
    svm_es = load(os.path.join("regression", "pbmcs", "baseline_model_svc_es.joblib"))
    svm_ke = load(os.path.join("regression", "pbmcs", "baseline_model_svc_ke.joblib"))
    svm_pc = load(os.path.join("regression", "pbmcs", "baseline_model_svc_pc.joblib"))
    svm_ss = load(os.path.join("regression", "pbmcs", "baseline_model_svc_ss.joblib"))
    svm_cd = load(os.path.join("regression", "pbmcs", "baseline_model_svc_cd.joblib"))
    svm_cn = load(os.path.join("regression", "pbmcs", "baseline_model_svc_cn.joblib"))
    svm_kc = load(os.path.join("regression", "pbmcs", "baseline_model_svc_kc.joblib"))
    svm_ce = load(os.path.join("regression", "pbmcs", "baseline_model_svc_ce.joblib"))
    svm_sc = load(os.path.join("regression", "pbmcs", "baseline_model_svc_sc.joblib"))
    svm_ac = load(os.path.join("regression", "pbmcs", "baseline_model_svc_ac.joblib"))
    svm_ma = load(os.path.join("regression", "pbmcs", "baseline_model_svc_ma.joblib"))
    yat_pred_svc_test = pd.DataFrame(svm_at.predict(pbmcs_fp_at), columns=["yat_pred_svr"]).set_index(pbmcs_fp_at.index)
    yes_pred_svc_test = pd.DataFrame(svm_es.predict(pbmcs_fp_es), columns=["yes_pred_svr"]).set_index(pbmcs_fp_es.index)
    yke_pred_svc_test = pd.DataFrame(svm_ke.predict(pbmcs_fp_ke), columns=["yke_pred_svr"]).set_index(pbmcs_fp_ke.index)
    ypc_pred_svc_test = pd.DataFrame(svm_pc.predict(pbmcs_fp_pc), columns=["ypc_pred_svr"]).set_index(pbmcs_fp_pc.index)
    yss_pred_svc_test = pd.DataFrame(svm_ss.predict(pbmcs_fp_ss), columns=["yss_pred_svr"]).set_index(pbmcs_fp_ss.index)
    ycd_pred_svc_test = pd.DataFrame(svm_cd.predict(pbmcs_fp_cd), columns=["ycd_pred_svr"]).set_index(pbmcs_fp_cd.index)
    ycn_pred_svc_test = pd.DataFrame(svm_cn.predict(pbmcs_fp_cn), columns=["ycn_pred_svr"]).set_index(pbmcs_fp_cn.index)
    ykc_pred_svc_test = pd.DataFrame(svm_kc.predict(pbmcs_fp_kc), columns=["ykc_pred_svr"]).set_index(pbmcs_fp_kc.index)
    yce_pred_svc_test = pd.DataFrame(svm_ce.predict(pbmcs_fp_ce), columns=["yce_pred_svr"]).set_index(pbmcs_fp_ce.index)
    ysc_pred_svc_test = pd.DataFrame(svm_sc.predict(pbmcs_fp_sc), columns=["ysc_pred_svr"]).set_index(pbmcs_fp_sc.index)
    yac_pred_svc_test = pd.DataFrame(svm_ac.predict(pbmcs_fp_ac), columns=["yac_pred_svr"]).set_index(pbmcs_fp_ac.index)
    yma_pred_svc_test = pd.DataFrame(svm_ma.predict(pbmcs_fp_ma), columns=["yma_pred_svr"]).set_index(pbmcs_fp_ma.index)
    stack_test  = pd.concat([yat_pred_rf_test, yat_pred_xgb_test, yat_pred_svc_test,
                            yes_pred_rf_test, yes_pred_xgb_test, yes_pred_svc_test,
                            yke_pred_rf_test, yke_pred_xgb_test, yke_pred_svc_test,
                            ypc_pred_rf_test, ypc_pred_xgb_test, ypc_pred_svc_test,
                            yss_pred_rf_test, yss_pred_xgb_test, yss_pred_svc_test,
                            ycd_pred_rf_test, ycd_pred_xgb_test, ycd_pred_svc_test,
                            ycn_pred_rf_test, ycn_pred_xgb_test, ycn_pred_svc_test,
                            ykc_pred_rf_test, ykc_pred_xgb_test, ykc_pred_svc_test,
                            yce_pred_rf_test, yce_pred_xgb_test, yce_pred_svc_test,
                            ysc_pred_rf_test, ysc_pred_xgb_test, ysc_pred_svc_test,
                            yac_pred_rf_test, yac_pred_xgb_test, yac_pred_svc_test,
                            yma_pred_rf_test, yma_pred_xgb_test, yma_pred_svc_test,],  axis=1)
    stacked_model = load(os.path.join("regression", "pbmcs", "stacked_model.joblib"))
    y_pred = pd.DataFrame(stacked_model.predict(stack_test), columns=["PBMCs_pIC50"]).set_index(pbmcs_fp_at.index)
    ad_model = load(os.path.join("regression", "pbmcs", "ad_2_2.joblib"))
    y_ad = ad_measurement("PBMCs", stack_test, ad_model, 0.8258, 0.9770, z=2)
    y_pred = pd.concat([y_pred, y_ad], axis=1)
    return y_pred

def main():
    df_name = input("Please type name of your csv file: ")
    df = load_data(df_name)
    fp_at, fp_es, fp_ke, fp_pc, fp_ss, fp_cd, fp_cn, fp_kc, fp_ce, fp_sc, fp_ac, fp_ma = compute_fps(df)
    herg_fp_at, herg_fp_es, herg_fp_ke, herg_fp_pc, herg_fp_ss, herg_fp_cd, herg_fp_cn, herg_fp_kc, herg_fp_ce, herg_fp_sc, herg_fp_ac, herg_fp_ma = herg_fp_sel(fp_at, fp_es, fp_ke, fp_pc, fp_ss, fp_cd, fp_cn, fp_kc, fp_ce, fp_sc, fp_ac, fp_ma)
    mtor_fp_at, mtor_fp_es, mtor_fp_ke, mtor_fp_pc, mtor_fp_ss, mtor_fp_cd, mtor_fp_cn, mtor_fp_kc, mtor_fp_ce, mtor_fp_sc, mtor_fp_ac, mtor_fp_ma = mtor_fp_sel(fp_at, fp_es, fp_ke, fp_pc, fp_ss, fp_cd, fp_cn, fp_kc, fp_ce, fp_sc, fp_ac, fp_ma)
    pbmcs_fp_at, pbmcs_fp_es, pbmcs_fp_ke, pbmcs_fp_pc, pbmcs_fp_ss, pbmcs_fp_cd, pbmcs_fp_cn, pbmcs_fp_kc, pbmcs_fp_ce, pbmcs_fp_sc, pbmcs_fp_ac, pbmcs_fp_ma = pbmcs_fp_sel(fp_at, fp_es, fp_ke, fp_pc, fp_ss, fp_cd, fp_cn, fp_kc, fp_ce, fp_sc, fp_ac, fp_ma)
    
    herg_output = stacked_herg(herg_fp_at, herg_fp_es, herg_fp_ke, herg_fp_pc, herg_fp_ss, herg_fp_cd, herg_fp_cn, herg_fp_kc, herg_fp_ce, herg_fp_sc, herg_fp_ac, herg_fp_ma)
    mtor_output = stacked_mtor(mtor_fp_at, mtor_fp_es, mtor_fp_ke, mtor_fp_pc, mtor_fp_ss, mtor_fp_cd, mtor_fp_cn, mtor_fp_kc, mtor_fp_ce, mtor_fp_sc, mtor_fp_ac, mtor_fp_ma)
    pbmcs_output = stacked_pbmcs(pbmcs_fp_at, pbmcs_fp_es, pbmcs_fp_ke, pbmcs_fp_pc, pbmcs_fp_ss, pbmcs_fp_cd, pbmcs_fp_cn, pbmcs_fp_kc, pbmcs_fp_ce, pbmcs_fp_sc, pbmcs_fp_ac, pbmcs_fp_ma)
    predictions = pd.concat([herg_output, mtor_output, pbmcs_output], axis=1)
    print(predictions)
    
if __name__ == "__main__":
    main()
    

    
    