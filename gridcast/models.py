"""Four predeclared models; all transformations fitted before the origin."""
import hashlib
import warnings
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .data import STEP

MODELS=('seasonal_naive','ridge_ar','lightgbm','sarima')
VERSION='aemo-3.0.0'
TRAINING_DAYS=365
BASELINE_WEIGHTS={'annual':.40,'monthly':.20,'weekly':.25,'daily':.15}

def features(values,t):
    local=t.tz_convert('Etc/GMT-10');hour=local.hour+local.minute/60
    return [values[-1],values[-4],values[-96],values[-672],np.mean(values[-4:]),np.mean(values[-96:]),
            np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),*[float(local.dayofweek==d) for d in range(7)]]

def clean_training(series,origin,days=TRAINING_DAYS):
    index=pd.date_range(origin-pd.Timedelta(days=days),origin-STEP,freq=STEP)
    train=series.reindex(index);missing=float(train.isna().mean())
    if missing>.02:raise ValueError(f'Training missingness {missing:.1%} exceeds 2%')
    train=train.ffill(limit=4)
    if train.isna().any():raise ValueError(f'Training contains unfillable gaps; need {days} complete days')
    # Keep immutable pre-origin reference for annual baseline; no future observations.
    train.attrs['history']=series.loc[series.index<origin].copy()
    return train,missing

def seasonal_baseline(history,targets):
    cutoff=targets[0]
    hist=history.loc[history.index<cutoff]
    recent=hist.loc[hist.index>=cutoff-pd.Timedelta(days=28)]
    annual_recent=hist.reindex(recent.index-pd.Timedelta(days=364))
    if annual_recent.notna().mean()<.98:raise ValueError('Annual benchmark needs at least 392 days of aligned history')
    prior_mean=float(annual_recent.mean())
    if abs(prior_mean)<1:raise ValueError('Annual adjustment denominator too small')
    factor=float(np.clip(recent.mean()/prior_mean,.5,1.5))
    components={}
    for name,days in {'annual':[357,364,371],'monthly':[7,14,21,28],'weekly':[7,14]}.items():
        values=np.array([hist.reindex(targets-pd.Timedelta(days=d)).to_numpy() for d in days])
        if not np.isfinite(values).all():raise ValueError('Missing seasonal reference values; no silent fallback')
        components[name]=values.mean(axis=0)*(factor if name=='annual' else 1)
    # After the first day, daily lags may refer to earlier predictions from this vintage.
    generated={};result=[]
    for i,t in enumerate(targets):
        daily=[]
        for day in (1,2,3):
            lag=t-pd.Timedelta(days=day)
            value=generated.get(lag) if lag>=cutoff else hist.get(lag)
            if value is None or not np.isfinite(value):raise ValueError('Missing daily reference')
            daily.append(value)
        value=sum(BASELINE_WEIGHTS[k]*v[i] for k,v in components.items())+BASELINE_WEIGHTS['daily']*np.mean(daily)
        result.append(value);generated[t]=value
    result=np.array(result)
    return result,{'weights':BASELINE_WEIGHTS,'annual_level_factor':factor,'annual_alignment_days':[357,364,371],
                   'reference_start':hist.index[0].isoformat(),'reference_end':hist.index[-1].isoformat(),
                   'reference_sha256':hashlib.sha256(hist.to_csv().encode()).hexdigest()}

def predict_all(train,horizon=96,weather=None):
    values=train.to_numpy(dtype=float);targets=pd.date_range(train.index[-1]+STEP,periods=horizon,freq=STEP)
    history=train.attrs.get('history',train)
    if weather is not None:
        required=train.index.append(targets)
        if not weather.index.is_unique or not np.isfinite(weather.reindex(required).to_numpy(dtype=float)).all():
            raise ValueError('Weather must cover every training and forecast timestamp')
    def model_features(hist,t):
        base=features(hist,t)
        return base if weather is None else base+weather.loc[t].to_list()
    timings={};started=time.perf_counter()
    baseline,base_art=seasonal_baseline(history,targets)
    timings['seasonal_naive']={'fit_predict_seconds':time.perf_counter()-started};started=time.perf_counter()
    result={'seasonal_naive':baseline}
    x=np.asarray([model_features(values[:i],train.index[i]) for i in range(672,len(values))]);y=values[672:]
    timings['feature_preparation_seconds']=time.perf_counter()-started;started=time.perf_counter()
    mean,scale=x.mean(axis=0),x.std(axis=0);scale[scale<1e-8]=1
    z=np.column_stack([np.ones(len(x)),(x-mean)/scale]);penalty=np.eye(z.shape[1])*10;penalty[0,0]=0
    coefficients=np.linalg.solve(z.T@z+penalty,z.T@y)
    ridge_fit=time.perf_counter()-started;started=time.perf_counter()
    hist=list(values);pred=[]
    for t in targets:
        value=float(np.r_[1,(np.asarray(model_features(hist,t))-mean)/scale]@coefficients);hist.append(value);pred.append(value)
    result['ridge_ar']=np.array(pred)
    timings['ridge_ar']={'fit_seconds':ridge_fit,'predict_seconds':time.perf_counter()-started};started=time.perf_counter()
    gb=LGBMRegressor(n_estimators=180,learning_rate=.04,num_leaves=15,max_depth=5,min_child_samples=40,
                    reg_lambda=5,random_state=42,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True)
    gb.fit(x,y);gb_fit=time.perf_counter()-started;started=time.perf_counter();hist=list(values);pred=[]
    for t in targets:
        value=float(gb.booster_.predict(np.asarray([model_features(hist,t)]),num_threads=2)[0]);hist.append(value);pred.append(value)
    result['lightgbm']=np.array(pred)
    timings['lightgbm']={'fit_seconds':gb_fit,'predict_seconds':time.perf_counter()-started};started=time.perf_counter()
    # Hourly SARIMA keeps a transparent daily seasonal period of 24; causal shape disaggregation.
    hourly=train.resample('1h',origin=targets[0]).mean().to_numpy()
    with warnings.catch_warnings(record=True) as caught:
        fit=SARIMAX(hourly,order=(1,0,0),seasonal_order=(1,1,0,24),trend='n',simple_differencing=False,
                    enforce_stationarity=True,enforce_invertibility=True).fit(disp=False,maxiter=60,cov_type='none',low_memory=True)
    sarima_fit=time.perf_counter()-started;started=time.perf_counter()
    hourly_pred=np.asarray(fit.forecast(int(np.ceil(horizon/4))))
    shaped=values[-28*96:].reshape(-1,4);offsets=(shaped-shaped.mean(axis=1,keepdims=True)).mean(axis=0)
    result['sarima']=(np.repeat(hourly_pred,4)+np.tile(offsets,len(hourly_pred)))[:horizon]
    if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('Nonfinite model prediction')
    timings['sarima']={'fit_seconds':sarima_fit,'predict_seconds':time.perf_counter()-started}
    artifact={'version':VERSION,'timings':timings,'training_days':len(train)/96,'baseline':base_art,'ridge':{'alpha':10,'coefficients':coefficients.tolist(),
        'feature_mean':mean.tolist(),'feature_scale':scale.tolist()},
        'lightgbm':{'parameters':gb.get_params(),'model_text':gb.booster_.model_to_string()},
        'sarima':{'order':[1,0,0],'seasonal_order':[1,1,0,24],'training_frequency':'hourly',
                  'quarter_hour_offsets':offsets.tolist(),'parameters':fit.params.tolist(),
                  'converged':bool(fit.mle_retvals.get('converged',False)),'warnings':[str(w.message) for w in caught]},
        'features':['lag1','lag4','lag96','lag672','mean4','mean96','sin_hour','cos_hour',*[f'dow{i}' for i in range(7)]]}
    if weather is not None:
        artifact['weather']={**weather.attrs.get('provenance',{}),'features':list(weather.columns),'applied_to':['ridge_ar','lightgbm'],'sarima':'univariate control; no weather exogenous regressors'}
        artifact['features']+=list(weather.columns)
    return targets,result,artifact
