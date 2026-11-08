from pathlib import Path
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)
SNAP=pd.Timestamp("2025-10-01")
CUTOFF=pd.Timestamp("2025-09-01")

def run():
    o=pd.read_csv(ROOT/"data/clean/orders.csv",parse_dates=["order_ts"])
    c=pd.read_csv(ROOT/"data/clean/customers.csv",parse_dates=["signup_date"])
    done=o[o.status.eq("completed")].copy()
    rfm=done.groupby("customer_id").agg(last_order_ts=("order_ts","max"),
        frequency=("order_id","nunique"),monetary=("order_amount","sum")).reset_index()
    rfm["recency"]=(SNAP-rfm.last_order_ts).dt.days
    X=rfm[["recency","frequency","monetary"]]
    scaler=StandardScaler(); Z=scaler.fit_transform(X)
    metric=[]
    for k in range(2,9):
        km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(Z)
        metric.append({"k":k,"inertia":km.inertia_,"silhouette":silhouette_score(Z,km.labels_)})
    pd.DataFrame(metric).to_csv(OUT/"kmeans_metrics.csv",index=False)
    k_final=4
    km=KMeans(n_clusters=k_final,n_init=10,random_state=42).fit(Z)
    rfm["segment_id"]=km.labels_
    prof=rfm.groupby("segment_id").agg(size=("customer_id","size"),
        recency=("recency","mean"),frequency=("frequency","mean"),monetary=("monetary","mean")).sort_values("monetary")
    names={sid: ("Premium Champions" if sid==prof.index[-1] else
                 "Frequent Essentials" if row.frequency>=prof.frequency.median() else
                 "At-Risk Occasionals") for sid,row in prof.iterrows()}
    rfm["segment_name"]=rfm.segment_id.map(names)
    rfm["is_premium"]=rfm.segment_id.eq(prof.index[-1])
    rfm[["customer_id","recency","frequency","monetary","segment_id","segment_name","is_premium"]].to_csv(OUT/"customer_segments.csv",index=False)
    pca=PCA(n_components=2).fit(Z)
    pd.DataFrame({"component":["PC1","PC2"],"explained_variance_ratio":pca.explained_variance_ratio_}).to_csv(OUT/"pca_variance.csv",index=False)

    # Leakage-safe pre-cutoff cohort and feature construction.
    pre=o[(o.order_ts<CUTOFF)]
    pre_done=pre[pre.status.eq("completed")]
    cohort=pre_done.groupby("customer_id").agg(last=("order_ts","max"),frequency=("order_id","nunique"),
        monetary=("order_amount","sum"),avg_discount=("discount_pct","mean"),
        avg_delivery=("delivery_minutes","mean")).reset_index()
    all_pre=pre.groupby("customer_id").agg(all_orders=("order_id","nunique"),
        cancelled=("status",lambda x:(x=="cancelled").sum())).reset_index()
    cohort=cohort.merge(all_pre,on="customer_id",how="left").merge(c,on="customer_id",how="left")
    cohort["recency"]=(CUTOFF-cohort["last"]).dt.days
    cohort["cancellation_rate"]=cohort.cancelled/cohort.all_orders
    cohort["tenure_days"]=(CUTOFF-cohort.signup_date).dt.days
    fav=pre_done.merge(pd.read_csv(ROOT/"data/clean/products.csv"),on="product_id").groupby("customer_id")["category"].agg(lambda x:x.mode().iloc[0])
    cohort=cohort.merge(fav.rename("favourite_category"),on="customer_id")
    september=done[(done.order_ts>=CUTOFF)&(done.order_ts<pd.Timestamp("2025-10-01"))].groupby("customer_id").size()
    cohort["lapsed"]=(~cohort.customer_id.isin(september.index)).astype(int)
    cohort=cohort.dropna()
    feature_num=["recency","frequency","monetary","avg_discount","cancellation_rate","avg_delivery","age","tenure_days"]
    feature_cat=["city","gender","favourite_category"]
    X=cohort[feature_num+feature_cat]; y=cohort.lapsed
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    prep=ColumnTransformer([
      ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),feature_num),
      ("cat",OneHotEncoder(drop="first",handle_unknown="ignore"),feature_cat)])
    log=Pipeline([("prep",prep),("model",LogisticRegression(max_iter=1000))])
    rf=Pipeline([("prep",prep),("model",RandomForestClassifier(n_estimators=100,random_state=42))])
    gl=GridSearchCV(log,{"model__C":[.01,.1,1,10,100]},cv=5,scoring="f1").fit(Xtr,ytr)
    gr=GridSearchCV(rf,{"model__max_depth":[3,5,8],"model__min_samples_leaf":[5,10,20]},cv=5,scoring="f1").fit(Xtr,ytr)
    models={"logistic":gl.best_estimator_,"random_forest":gr.best_estimator_}
    report=[]
    for name,m in models.items():
        pred=m.predict(Xte)
        report.append({"model":name,"best_params":str(gl.best_params_ if name=="logistic" else gr.best_params_),
          "cv_f1":gl.best_score_ if name=="logistic" else gr.best_score_,
          "confusion_matrix":confusion_matrix(yte,pred).tolist(),
          "precision":precision_score(yte,pred,zero_division=0),
          "recall":recall_score(yte,pred,zero_division=0),"f1":f1_score(yte,pred,zero_division=0)})
    pd.DataFrame(report).to_csv(OUT/"lapse_model_metrics.csv",index=False)
    selected=models["random_forest"]
    joblib.dump(selected,OUT/"lapse_model.joblib")
    loaded=joblib.load(OUT/"lapse_model.joblib")
    assert np.array_equal(selected.predict(Xte),loaded.predict(Xte))
    # Post-hoc premium label is deliberately used only to create the illustrative outreach list.
    seg=pd.read_csv(OUT/"customer_segments.csv")
    premium=set(seg.loc[seg.is_premium.astype(str).str.lower().eq("true"),"customer_id"])
    te_ids=Xte.index
    prob=selected.predict_proba(Xte)[:,1]
    ret=pd.DataFrame({"customer_id":cohort.loc[te_ids,"customer_id"].values,"lapse_probability":prob,"actually_lapsed":yte.values})
    ret=ret[ret.customer_id.isin(premium)].sort_values("lapse_probability",ascending=False).head(10)
    ret.to_csv(OUT/"retention_list.csv",index=False)
    return rfm,prof,cohort,report

if __name__=="__main__": run()
