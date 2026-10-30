"""Two-level reconciliation; inputs ordered NEM then the five regions."""
import numpy as np
from sklearn.covariance import LedoitWolf
S=np.vstack([np.ones(5),np.eye(5)])
def bottom_up(base):return np.asarray(base)[:,1:]@S.T

def top_down(base,training):
 training=np.asarray(training,dtype=float)
 if training.shape[1]!=6 or not np.isfinite(training).all():raise ValueError('Expected six complete training series')
 if np.any(abs(training[:,0])<1):raise ValueError('Total too small to estimate proportions')
 shares=(training[:,1:]/training[:,[0]]).mean(axis=0);shares/=shares.sum()
 # Signed dispatch demand permits a negative share. Do not clip or lose coherence.
 return (np.asarray(base)[:,[0]]*shares)@S.T,shares

def mint_shrink(base,past_errors):
 errors=np.asarray(past_errors,dtype=float)
 if errors.ndim!=2 or errors.shape[1]!=6 or len(errors)<12 or not np.isfinite(errors).all():raise ValueError('Need complete pre-test errors for all six nodes')
 scale=errors.std(axis=0);scale=np.maximum(scale,1e-6)
 estimator=LedoitWolf().fit(errors/scale)
 covariance=estimator.covariance_*np.outer(scale,scale)
 covariance+=np.eye(6)*max(np.trace(covariance)/6,1)*1e-10
 winv_s=np.linalg.solve(covariance,S)
 projection=S@np.linalg.solve(S.T@winv_s,winv_s.T)
 reconciled=np.asarray(base)@projection.T
 return reconciled,{'shrinkage':float(estimator.shrinkage_),'covariance':covariance.tolist(),'projection':projection.tolist(),'calibration_rows':len(errors),'method':'MinT with covariance shrunk toward its diagonal; pooled 0–24h errors'}
