"""Direct demand corrections relative to the seasonal benchmark, using past data only."""
import numpy as np
import pandas as pd
from .data import STEP

PARAMETERS=dict(n_estimators=220,learning_rate=.035,num_leaves=15,max_depth=5,min_child_samples=120,reg_lambda=15,objective='regression_l1',linear_tree=False)
MIN_HISTORY=392*96

def design(series,origins,leads):
    """Origins and zero-based leads broadcast to rows. Target values are never read."""
    origins=np.asarray(origins,dtype=int);leads=np.asarray(leads,dtype=int)
    origins,leads=np.broadcast_arrays(origins,leads);o=origins.ravel();h=leads.ravel();t=o+h
    if (o<MIN_HISTORY).any() or (h<0).any() or (h>=96).any():raise ValueError('Need 392 past days and leads 0..95')
    v=series.to_numpy(dtype=float);prefix=np.r_[0.,np.cumsum(v)]
    def average(end,n):return (prefix[end]-prefix[end-n])/n
    recent=average(o,28*96);prior=average(o-364*96,28*96)
    if (abs(prior)<1).any():raise ValueError('Annual reference too small')
    factor=np.clip(recent/prior,.5,1.5)
    annual=np.mean([v[t-d*96] for d in [357,364,371]],axis=0)*factor
    monthly=np.mean([v[t-d*96] for d in [7,14,21,28]],axis=0)
    weekly=np.mean([v[t-d*96] for d in [7,14]],axis=0)
    daily=np.mean([v[t-d*96] for d in [1,2,3]],axis=0)
    baseline=.4*annual+.2*monthly+.25*weekly+.15*daily
    times=series.index[0]+pd.to_timedelta(t*15,unit='min');local=times.tz_convert('Etc/GMT-10');hour=local.hour+local.minute/60
    x=np.column_stack([baseline,annual,monthly,weekly,daily,*[v[t-d*96] for d in [1,2,3,7,14,28,364]],
                      v[o-1],v[o-4],average(o,4),average(o,96),average(o,672),recent,factor,
                      h/4,np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),
                      np.sin(2*np.pi*local.dayofyear/365.25),np.cos(2*np.pi*local.dayofyear/365.25),
                      *[np.asarray(local.dayofweek==d,dtype=float) for d in range(7)]])
    if not np.isfinite(x).all():raise ValueError('Invalid direct features')
    return x,baseline

def training(series,cutoff):
    """Six-hour training issues, hourly targets plus the final quarter-hour."""
    series=series.loc[series.index<pd.Timestamp(cutoff)]
    expected=pd.date_range(series.index[0],periods=len(series),freq=STEP)
    if not series.index.equals(expected) or not np.isfinite(series.to_numpy()).all():raise ValueError('Complete history required')
    origins=np.arange(MIN_HISTORY,len(series)-95,24)
    leads=np.r_[np.arange(0,96,4),95]
    o=np.repeat(origins,len(leads));h=np.tile(leads,len(origins))
    x,b=design(series,o,h)
    return x,series.to_numpy()[o+h]-b

def predict(model,history,origin,horizon=96,mean=0.,scale=1.,weight=1.):
    history=history.loc[history.index<pd.Timestamp(origin)]
    if len(history)<MIN_HISTORY:raise ValueError('Not enough direct history')
    if not history.index.equals(pd.date_range(history.index[0],pd.Timestamp(origin),freq=STEP,inclusive='left')):raise ValueError('Missing direct history')
    x,b=design(history,np.full(horizon,len(history)),np.arange(horizon))
    return b+weight*model.predict((x-mean)/scale)
