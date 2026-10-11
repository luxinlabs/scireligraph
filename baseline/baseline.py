#!/usr/bin/env python3
"""SCORE claim-level replication forecasting baseline. No synthetic outcomes."""
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Explicitly whitelist only ORIGINAL-study information; never infer features from replication data.
NUMERIC = ['original_n','original_p','original_effect_size','publication_year']
CATEGORICAL = ['discipline','journal']
REQUIRED = ['claim_id','paper_id','replicated']

def load(path):
    df = pd.read_csv(path)
    missing = set(REQUIRED)-set(df.columns)
    if missing: raise ValueError(f'Missing required columns: {sorted(missing)}. See schema.md; map SCORE columns explicitly.')
    if df.claim_id.isna().any() or df.paper_id.isna().any(): raise ValueError('Missing claim/paper identifiers')
    if df.claim_id.duplicated().any(): raise ValueError('Duplicate claim_id; aggregate repeated attempts before modeling')
    if not set(df.replicated.dropna().unique()).issubset({0,1}): raise ValueError('replicated must be binary 0/1')
    if df.replicated.isna().any(): raise ValueError('Missing outcome labels')
    return df

def evaluate(df, folds=5, seed=42):
    numeric=[c for c in NUMERIC if c in df and df[c].notna().any()]
    categoric=[c for c in CATEGORICAL if c in df and df[c].notna().any()]
    features=numeric+categoric
    if not features: raise ValueError('No usable original-study features; do not train on outcomes alone')
    if len(df.paper_id.unique()) < folds: raise ValueError('Not enough paper groups')
    y=df.replicated.astype(int).to_numpy()
    if len(np.unique(y))<2: raise ValueError('Both outcome classes required')
    transformer=ColumnTransformer([
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),
        ('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),categoric)
    ])
    cv=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=seed)
    rows=[]; predictions=[]
    for fold,(tr,te) in enumerate(cv.split(df[features],y,df.paper_id)):
        if len(np.unique(y[tr]))<2: raise ValueError(f'Fold {fold}: one-class training data')
        models={
            'prevalence':DummyClassifier(strategy='prior'),
            'metadata_logistic':Pipeline([('features',transformer),('model',LogisticRegression(max_iter=3000,C=1.0))])
        }
        for name,model in models.items():
            model.fit(df.iloc[tr][features],y[tr]); p=model.predict_proba(df.iloc[te][features])[:,list(model.classes_).index(1)]
            p=np.clip(p,1e-8,1-1e-8)
            metrics={'fold':fold,'model':name,'n_test':len(te),'n_papers_test':df.iloc[te].paper_id.nunique(),
                     'brier':float(brier_score_loss(y[te],p)),'log_loss':float(log_loss(y[te],p,labels=[0,1])),
                     'auroc':float(roc_auc_score(y[te],p)) if len(np.unique(y[te]))==2 else None,
                     'average_precision':float(average_precision_score(y[te],p)) if len(np.unique(y[te]))==2 else None}
            rows.append(metrics)
            for idx,pr in zip(te,p): predictions.append({'claim_id':str(df.iloc[idx].claim_id),'paper_id':str(df.iloc[idx].paper_id),'fold':fold,'model':name,'y_true':int(y[idx]),'probability':float(pr)})
    return pd.DataFrame(rows),pd.DataFrame(predictions),features

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--csv',required=True);parser.add_argument('--out',default='results');parser.add_argument('--folds',type=int,default=5);args=parser.parse_args()
    df=load(args.csv);metrics,preds,features=evaluate(df,args.folds)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    metrics.to_csv(out/'fold_metrics.csv',index=False);preds.to_csv(out/'out_of_fold_predictions.csv',index=False)
    summary={}
    for name,g in preds.groupby('model'):
        summary[name]={'n':len(g),'auroc':float(roc_auc_score(g.y_true,g.probability)),
                       'brier':float(brier_score_loss(g.y_true,g.probability)),
                       'log_loss':float(log_loss(g.y_true,g.probability,labels=[0,1])),
                       'average_precision':float(average_precision_score(g.y_true,g.probability))}
    (out/'summary.json').write_text(json.dumps({'n_claims':len(df),'n_papers':df.paper_id.nunique(),'features':features,'models':summary},indent=2))
    print(json.dumps({'n_claims':len(df),'n_papers':df.paper_id.nunique(),'features':features,'models':summary},indent=2))
if __name__=='__main__': main()
