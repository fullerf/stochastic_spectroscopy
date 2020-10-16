# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import sys
sys.path.append('/home/dgp_iwvi_gpflow2/')
import gpflow
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, List, Union
import dgp_iwvi_gpflow2.layers as layers
import attr
import tensorflow_probability as tfp

RegressionData = Tuple[tf.Tensor, tf.Tensor]
AuxRegressionData = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
InputData = tf.Tensor
MeanVarKL = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]


class SLGP(gpflow.models.BayesianModel, gpflow.models.ExternalDataTrainingLossMixin):
    """
    This is a "Single Layer Gaussian Process" model, it should be identical in behavior/performance to the
    svgp model
    """

    def __init__(
        self,
        layer: layers.GPLayer,
        likelihood: gpflow.likelihoods.Likelihood,
        *,
        num_data: Optional[int] = None
    ):
        """
        - layer: an instance of GPLayer
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # init the super class, accept args
        super().__init__()
        self.num_data = num_data
        self.layer = layer
        self.likelihood = likelihood

    def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:
        return self.elbo(data)

    def elbo(self, data: RegressionData) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y = data
        f_mean, f_var, kl = self.predict_f(X, full_cov=False)
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl

    def predict_f(self, Xnew: InputData, full_cov=False) -> MeanVarKL:
        mu, var, kl = self.layer.components(
            Xnew,
            full_cov=full_cov,
        )
        return mu, var, kl


# +
GPMMData = Tuple[tf.Tensor, tf.Tensor]

class SLGPMM(gpflow.models.BayesianModel, gpflow.models.ExternalDataTrainingLossMixin):
    """
    This is a "Single Layer Gaussian Process Mixture of Measurements" model, wherein the
    observed variable is linear mixed by a known mixing matrix W.
    """

    def __init__(
        self,
        layer: layers.GPLayer,
        likelihood: gpflow.likelihoods.Likelihood,
        reduction_axis_len: int,
        *,
        num_data: Optional[int] = None
    ):
        """
        - layer: an instance of GPLayer
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # init the super class, accept args
        super().__init__()
        self.num_data = num_data
        self.layer = layer
        self.likelihood = likelihood
        self.reduction_axis_len = int(reduction_axis_len)
        reduction_axis = tf.convert_to_tensor(np.linspace(-np.pi,np.pi,self.reduction_axis_len)[:,None],
                                                  gpflow.config.default_float())
        self.raxis = gpflow.Parameter(reduction_axis, transform=None)
        gpflow.utilities.set_trainable(self.raxis, False)

    def maximum_log_likelihood_objective(self, data: GPMMData) -> tf.Tensor:
        return self.elbo(data)

    def elbo(self, data: GPMMData) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        W, Y = data
        f_mean, f_var, kl = self.predict_f(self.raxis)  # R,1, R,R
        f_mean_reduced = tf.squeeze(tf.einsum('nr,...rd->...n',W,f_mean))[:,None] # N -> N,1
        f_var_reduced = tf.squeeze(tf.einsum('nr,...rq,nq->...n',W,f_var,W))[:,None] # N -> N,1
        var_exp = self.likelihood.variational_expectations(f_mean_reduced, f_var_reduced, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(W)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl

    def predict_f(self, Xnew: tf.Tensor, full_cov=True) -> MeanVarKL:
        mu, var, kl = self.layer.components(
            Xnew,
            full_cov=full_cov,
        )
        return mu, var, kl


# -

class LVNGP(gpflow.models.BayesianModel, gpflow.models.ExternalDataTrainingLossMixin):
    """
    This is the basic model mentioned in Hugh Salembini's DGP IWVI paper, with N GP layers
    beginning with a single amortized latent layer.
    """
    def __init__(self,
                 likelihood: gpflow.likelihoods.Likelihood,
                 latent_layer: layers.AmortizedLatentVariableLayer,
                 gp_layers: List[layers.GPLayer],
                 *,
                 num_samples: int = 1,
                 num_data: Optional[int] = None,
                 local_kl_scale: Optional[float] = None,
                ):
        super().__init__()
        self.likelihood = likelihood
        self.num_samples = num_samples
        self.num_data = num_data
        self.layers = [latent_layer] + gp_layers
        if local_kl_scale is not None:
            self.local_kl_scale = tf.convert_to_tensor(float(local_kl_scale), dtype=gpflow.config.default_float())
        
    def propagate(self, X: tf.Tensor,
                  full_cov: bool = False,
                  inference_amortization_inputs: Optional[tf.Tensor] = None,
                  is_sampled_local_regularizer: bool = False):
        samples, means, covs, kls, kl_types = [X, ], [], [], [], []

        for i,layer in enumerate(self.layers):
#             full_cov_local = False if i < len(self.layers)-1 else full_cov
            full_cov_local = full_cov
            sample, mean, cov, kl = layer.propagate(samples[-1],
                                                    full_cov=full_cov_local,
                                                    inference_amortization_inputs=inference_amortization_inputs,
                                                    is_sampled_local_regularizer=is_sampled_local_regularizer)
            samples.append(sample)
            means.append(mean)
            covs.append(cov)
            kls.append(kl)
            kl_types.append(layer.regularizer_type)

        return samples[1:], means, covs, kls, kl_types
        
    def maximum_log_likelihood_objective(self, 
                                         data: AuxRegressionData) -> tf.Tensor:
        return self.elbo(data)
    
    def elbo(self, 
             data: AuxRegressionData) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y, XY = data
        X_tiled = tf.tile(X, [self.num_samples, 1])  # SN, Dx
        Y_tiled = tf.tile(Y, [self.num_samples, 1])  # SN, Dy
        XY_tiled = tf.tile(XY, [self.num_samples, 1])  # SN, Dy
        samples, means, covs, kls, kl_types = self.propagate(X_tiled,
                                                             full_cov=False,
                                                             inference_amortization_inputs=XY_tiled,
                                                             is_sampled_local_regularizer=False)
        if self.local_kl_scale is not None:
            local_kls = [kl*self.local_kl_scale for kl, t in zip(kls, kl_types) if t is layers.RegularizerType.LOCAL]
        else:
            local_kls = [kl for kl, t in zip(kls, kl_types) if t is layers.RegularizerType.LOCAL]
        global_kls = [kl for kl, t in zip(kls, kl_types) if t is layers.RegularizerType.GLOBAL]
        L_SN = self.likelihood.variational_expectations(means[-1], covs[-1], Y_tiled)  # SN
       
        # separate out repeated samples
        shape_S_N = tf.concat([tf.convert_to_tensor([self.num_samples], dtype=tf.int32),
                               tf.convert_to_tensor([tf.shape(X)[0]], dtype=tf.int32)],0)
        L_S_N = tf.reshape(L_SN, shape_S_N)
        
        if len(local_kls) > 0:
            local_kls_SN_D = tf.concat(local_kls, -1)  # SN, sum(W_dims)
            local_kls_SN = tf.reduce_sum(local_kls_SN_D, -1)
            local_kls_S_N = tf.reshape(local_kls_SN, shape_S_N)
            L_S_N -= local_kls_S_N  # SN
            
        global_kl = tf.reduce_sum(global_kls)
            
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, global_kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], global_kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, global_kl.dtype)

        # This line is replaced with tf.reduce_logsumexp in the IW case
        logp = tf.reduce_mean(L_S_N, 0)

        return tf.reduce_sum(logp) * scale - global_kl
    
    def predict_f_multisample(self, X, S):
        X_tiled = tf.tile(X[None, :, :], [S, 1, 1])
        _, means, covs, _, _ = self.propagate(X_tiled)
        return means[-1], covs[-1]

    def predict_y_samples(self, X, S):
        X_tiled = tf.tile(X[None, :, :], [S, 1, 1])
        _, means, covs, _, _ = self.propagate(X_tiled)
        m, v = self.likelihood.predict_mean_and_var(means[-1], covs[-1])
        z = tf.random.normal(tf.shape(means[-1]), dtype=X.dtype)
        return m + z * v**0.5


# +
   
class LVNGP_IWVI(LVNGP):
    def __init__(self,
                 likelihood: gpflow.likelihoods.Likelihood,
                 latent_layer: layers.AmortizedLatentVariableLayer,
                 gp_layers: List[layers.GPLayer],
                 *,
                 num_samples: int = 1,
                 num_data: Optional[int] = None):
        super().__init__(likelihood, latent_layer, gp_layers, num_samples=num_samples, num_data=num_data)
        
    def elbo(self, 
             data: AuxRegressionData) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y, XY = data
        X_tiled = tf.tile(X[:, None, :], [1, self.num_samples, 1])  # N, S, Dx
        Y_tiled = tf.tile(Y[:, None, :], [1, self.num_samples, 1])  # N, S, Dy
        XY_tiled = tf.tile(XY[:, None, :], [1, self.num_samples, 1])  # N, S, Dxy
        samples, means, covs, kls, kl_types = self.propagate(X_tiled,
                                                             full_cov=True,  # full_cov is over the S dim
                                                             inference_amortization_inputs=XY_tiled,
                                                             is_sampled_local_regularizer=True)

        local_kls = [kl for kl, t in zip(kls, kl_types) if t is layers.RegularizerType.LOCAL]
        global_kls = [kl for kl, t in zip(kls, kl_types) if t is layers.RegularizerType.GLOBAL]

        # This could be made slightly more efficient by making the last layer full_cov=False,
        # but this seems a small price to pay for cleaner code. NB this is only a SxS matrix, not
        # an NxN matrix.
        cov_diag = tf.transpose(tf.linalg.diag_part(covs[-1]), [0, 2, 1])  # N,Dy,S,S -> N,S,Dy
        L_N_S = self.likelihood.variational_expectations(means[-1], cov_diag, Y_tiled)  # N, S
        
        if len(local_kls) > 0:
            local_kls_N_S_D = tf.concat(local_kls, -1)  # N, S, sum(W_dims)
            local_kls_N_S = tf.reduce_sum(local_kls_N_S_D, -1)
            L_N_S -= local_kls_N_S  # N,S
            
        global_kl = tf.reduce_sum(global_kls)
            
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, global_kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], global_kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, global_kl.dtype)

        # This line is replaced with tf.reduce_logsumexp in the IW case
        logp = tf.reduce_logsumexp(L_N_S, -1) - np.log(self.num_samples)
        return tf.reduce_sum(logp) * scale - global_kl


# -

@attr.s(auto_attribs=True)
class LVNGP_Config(object):
    latent_layer: Union[layers.LatentLayer_Config, layers.EmbeddingLatentLayer_Config]
    gplayers: List[layers.GPLayer_Config]
    var_init: float
    latent_samples: int
    inference_type: str
    local_kl_scale: Optional[float]


class DGP_GPMM(gpflow.models.BayesianModel, gpflow.models.ExternalDataTrainingLossMixin):
    """
    Garden variety doubly stochastic DGP with GPMM likelihood
    """
    def __init__(self,
                 likelihood: gpflow.likelihoods.Likelihood,
                 gp_layers: List[layers.GPLayer],
                 reduction_axis_len: int,
                 *,
                 num_samples: int = 1,
                 num_data: Optional[int] = None,
                ):
        super().__init__()
        self.likelihood = likelihood
        self.num_samples = num_samples
        self.num_data = num_data
        self.reduction_axis_len = int(reduction_axis_len)
        reduction_axis = tf.convert_to_tensor(np.linspace(-np.pi,np.pi,self.reduction_axis_len)[:,None],
                                                  gpflow.config.default_float())
        self.raxis = gpflow.Parameter(reduction_axis, transform=None)
        gpflow.utilities.set_trainable(self.raxis, False)
        self.layers = gp_layers
        
    def propagate(self, X: tf.Tensor,
                  full_cov: bool = False):
        samples, means, covs, kls, kl_types = [X, ], [], [], [], []

        for i,layer in enumerate(self.layers):
            sample, mean, cov, kl = layer.propagate(samples[-1],
                                                    full_cov=full_cov)
            samples.append(sample)
            means.append(mean)
            covs.append(cov)
            kls.append(kl)
            kl_types.append(layer.regularizer_type)

        return samples[1:], means, covs, kls, kl_types
        
    def maximum_log_likelihood_objective(self, data: AuxRegressionData) -> tf.Tensor:
        return self.elbo(data)
    
    def elbo(self, 
             data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        W, Y = data
        X_tiled = tf.tile(self.raxis[None, :, :], [self.num_samples, 1, 1])  # SRD
        Y_tiled = tf.tile(Y[None, :, :], [self.num_samples, 1, 1])  # SND
        samples, means, covs, kls, kl_types = self.propagate(X_tiled,
                                                             full_cov=True)
        global_kls = [kl for kl, t in zip(kls, kl_types) if t is layers.RegularizerType.GLOBAL]
        f_mean = means[-1] # SRD
        f_cov = covs[-1] # SDRR
        f_mean_reduced =tf.einsum('nr,...rd->...nd',W,f_mean) # SRD -> SND
        f_var_reduced = tf.einsum('nr,...drq,nq->...nd',W,f_cov,W) # SDRR -> SND
        L_S_N = self.likelihood.variational_expectations(f_mean_reduced, f_var_reduced, Y_tiled)  # SND
        global_kl = tf.reduce_sum(global_kls)
            
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, global_kl.dtype)
            minibatch_size = tf.cast(tf.shape(W)[0], global_kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, global_kl.dtype)

        # This line is replaced with tf.reduce_logsumexp in the IW case
        logp = tf.reduce_mean(L_S_N, 0)

        return tf.reduce_sum(logp) * scale - global_kl
    
    def predict_f_multisample(self, X, S):
        X_tiled = tf.tile(X[None, :, :], [S, 1, 1])
        _, means, covs, _, _ = self.propagate(X_tiled)
        return means[-1], covs[-1]

    def predict_y_samples(self, X, S):
        X_tiled = tf.tile(X[None, :, :], [S, 1, 1])
        _, means, covs, _, _ = self.propagate(X_tiled)
        m, v = self.likelihood.predict_mean_and_var(means[-1], covs[-1])
        z = tf.random.normal(tf.shape(means[-1]), dtype=X.dtype)
        return m + z * v**0.5


@attr.s(auto_attribs=True)
class DGP_GPMM_Config(object):
    gplayers: List[layers.GPLayer_Config]
    var_init: float
    num_samples: int


class LVNDGP_GPMM_IWVI_MarginalApprox(gpflow.models.BayesianModel, gpflow.models.ExternalDataTrainingLossMixin):
    """
    Garden variety doubly stochastic DGP with GPMM likelihood, latent inputs, and importance sampling. We
    approximate the full GPMM distribution by marginalizing before carrying out reduction rather than after.
    """
    def __init__(self,
                 likelihood: gpflow.likelihoods.Likelihood,
                 latent_layer: layers.AmortizedSASELatentVariableLayer,
                 gp_layers: List[layers.GPLayer],
                 reduction_axis_len: int,
                 *,
                 num_samples: int = 1,
                 num_data: Optional[int] = None,
                ):
        super().__init__()
        self.likelihood = likelihood
        self.num_samples = num_samples
        self.num_data = num_data
        self.reduction_axis_len = int(reduction_axis_len)
        reduction_axis = tf.convert_to_tensor(np.linspace(-np.pi,np.pi,self.reduction_axis_len)[:,None],
                                                  gpflow.config.default_float())
        self.raxis = gpflow.Parameter(reduction_axis, transform=None)
        gpflow.utilities.set_trainable(self.raxis, False)
        self.latent_layer = latent_layer
        self.layers = gp_layers
        
    def propagate(self, X: tf.Tensor,
                  full_cov: bool = False,
                 is_sampled_local_regularizer: bool = False):
        samples, means, covs, kls, kl_types = [X, ], [], [], [], []

        for i,layer in enumerate(self.layers):
            sample, mean, cov, kl = layer.propagate(samples[-1],
                                                    full_cov=full_cov,
                                                   is_sampled_local_regularizer=is_sampled_local_regularizer)
            samples.append(sample)
            means.append(mean)
            covs.append(cov)
            kls.append(kl)
            kl_types.append(layer.regularizer_type)

        return samples[1:], means, covs, kls, kl_types
        
    def maximum_log_likelihood_objective(self, data: AuxRegressionData) -> tf.Tensor:
        return self.elbo(data)
    
    def elbo(self, 
             data: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        W, Y, Wstd = data
        N = tf.shape(W)[0] # expecting NR shape
        R = tf.shape(self.raxis)[0] # expecting R1 shape
        X_tiled = tf.tile(self.raxis[None, :, :], [self.num_samples, N, 1])  # S, N*R, Din
        W_tiled = tf.tile(Wstd[None, :, :], [self.num_samples, 1, 1]) # S, N, R
        Y_tiled = tf.tile(Y[:, None, :], [1, self.num_samples, 1]) # N, S, Dout
        # get the latent sample
        lsamp, lmean, lcov, lkl = self.latent_layer.propagate(X_tiled, inference_amortization_inputs=W_tiled,
                                                             is_sampled_local_regularizer=True)
        
        #lsamp has shape ..., S, N*R, Din + Dlatent. Want to reshape to ..., N*R, S, Din + Dlatent
        lsamp_reshaped = tf.einsum('...ijk->...jik',lsamp)  # N*R, S, Din + Dlatent
        lmean_reshaped = tf.einsum('...ijk->...jik',lmean)  # N*R, S, Din + Dlatent
        lcov_reshaped = tf.einsum('...ijk->...jik',lcov)  # N*R, S, Din + Dlatent
        lkl_reshaped = tf.einsum('...ijk->...jik',lkl)  # N*R, S, Dlatent
        
        samples, means, covs, kls, kl_types = self.propagate(lsamp_reshaped,
                                                             full_cov=True,
                                                            is_sampled_local_regularizer=False)
        samples = [lsamp_reshaped, *samples]
        means = [lmean_reshaped, *means]
        lcov_reshaped = [lcov_reshaped, *covs]
        kls = [lkl_reshaped, *kls]
        kl_types = [self.latent_layer.regularizer_type, *kl_types]
        
        global_kls = [kl for kl, t in zip(kls, kl_types) if t is layers.RegularizerType.GLOBAL]
        local_kls = [kl for kl, t in zip(kls, kl_types) if t is layers.RegularizerType.LOCAL]
        
        f_mean = means[-1]  # N*R, S, Dout
        Dout = tf.shape(f_mean)[-1]
        f_var = tf.linalg.diag_part(covs[-1])  # N*R, Dout, S, S -> N*R, Dout, S
        
        
        f_mean_reshaped = tf.reshape(f_mean, [N, R, self.num_samples, Dout])  # N, R, S, Dout
        f_var_reshaped = tf.reshape(f_var, [N, R, Dout, self.num_samples])  # N, R, Dout, S
        f_mean_reduced =tf.einsum('nr,nrsd->nsd',W,f_mean_reshaped) # N, R, S, Dout -> N, S, Dout
        f_var_reduced = tf.einsum('nr,nrds->nsd',W*W,f_var_reshaped)  # N, R, Dout, S  -> N, S, Dout
        #variational_expectations reduces over -1 dim, leaving only S, N
        
        L_N_S = self.likelihood.variational_expectations(f_mean_reduced, f_var_reduced, Y_tiled)
        
        if len(local_kls) > 0:
            local_kls_NR_S_D = tf.concat(local_kls, -1)  # N*R, S, sum(Dlatents)
            Dl = tf.shape(local_kls_NR_S_D)[-1]
            local_kls_N_R_S_D = tf.reshape(local_kls_NR_S_D, [N, R, self.num_samples, Dl])
            local_kls_N_R_S = tf.reduce_sum(local_kls_N_R_S_D, -1)
            local_kls_N_S = tf.reduce_sum(local_kls_N_R_S,-2)
            L_N_S -= local_kls_N_S  # N,S
        
        global_kl = tf.reduce_sum(global_kls) # a scalar
            
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, global_kl.dtype)
            minibatch_size = tf.cast(tf.shape(W)[0], global_kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, global_kl.dtype)

        # This line is replaced with tf.reduce_logsumexp in the IW case
        logp = tf.reduce_logsumexp(L_N_S, -1)

        return tf.reduce_sum(logp) * scale - global_kl
    
    def predict_f_multisample(self, X, S):
        X_tiled = tf.tile(X[None, :, :], [S, 1, 1])
        _, means, covs, _, _ = self.propagate(X_tiled)
        return means[-1], covs[-1]

    def predict_y_samples(self, X, S):
        X_tiled = tf.tile(X[None, :, :], [S, 1, 1])
        _, means, covs, _, _ = self.propagate(X_tiled)
        m, v = self.likelihood.predict_mean_and_var(means[-1], covs[-1])
        z = tf.random.normal(tf.shape(means[-1]), dtype=X.dtype)
        return m + z * v**0.5


class DGPMM_IWVI_MITM(gpflow.models.BayesianModel, gpflow.models.ExternalDataTrainingLossMixin):
    """
    it's a DGP -> GPMM reduction -> DGP IWVI, so we have layers for the spectrum: spectrum_layers
    (these get reduced by sase), then we enter the reduced layers.
    """
    def __init__(self,
                 likelihood: gpflow.likelihoods.Likelihood,
                 spectrum_layers: List[layers.GPLayer],
                 reduced_layers: List[Union[layers.GPLayer, layers.AmortizedSASELatentVariableLayer]],
                 reduction_axis_len: int,
                 *,
                 num_samples: int = 1,
                 num_data: Optional[int] = None,
                ):
        super().__init__()
        self.likelihood = likelihood
        self.num_samples = num_samples
        self.num_data = num_data
        self.reduction_axis_len = int(reduction_axis_len)
        reduction_axis = tf.convert_to_tensor(np.linspace(-np.pi,np.pi,self.reduction_axis_len)[:,None],
                                                  gpflow.config.default_float())
        self.raxis = gpflow.Parameter(reduction_axis, transform=None)
        gpflow.utilities.set_trainable(self.raxis, False)
        self.latent_layer = latent_layer
        self.spectrum_layers = spectrum_layers
        self.reduced_layers = reduced_layers
        
    def propagate(self, X: tf.Tensor, 
                  W: Optional[tf.Tensor] = None,
                  full_cov: bool = False,
                  inference_amortization_inputs: Optional[tf.Tensor] = None,
                  is_sampled_local_regularizer: bool = False):
        samples, means, covs, kls, kl_types = [X, ], [], [], [], []

        for layer in self.spectrum_layers:
            sample, mean, cov, kl = layer.propagate(samples[-1],
                                                    full_cov=True,
                                                    is_sampled_local_regularizer=is_sampled_local_regularizer,
                                                    inference_amortization_inputs=inference_amortization_inputs)
            samples.append(sample)
            means.append(mean)
            covs.append(cov)
            kls.append(kl)
            kl_types.append(layer.regularizer_type)
            
        # reduce the latent space
        if W is None:
            # this case used for prediction of spectra
            N = tf.shape(X)[-2]
            W = tf.eye(N,N, dtype=gpflow.config.default_float())
        
        # we assume that W is always NxR
        # below we produce the reduced mean and marginal variance
        rmean = tf.einsum('nr,...rd->...nd',W,means[-1])
        rvar = tf.einsum('nr,...drq,nq->...nd')
        
        # sample from the marginalized mean/var, producing a single sample
        eps = tf.random.normal(tf.shape(rmean), dtype=gpflow.config.default_float())
        rsample = rmean + eps * tf.sqrt(rvar)
        
        # now we tile the latent space to prepare for importance sampling
        bdims = tf.shape(rsample)[:-2]
        rsample_tiled = tf.tile(rsample[...,None,:], bdims*[1] + [self.num_samples, 1])
        
        means.append(rmean)
        covs.append(rcov)
        samples.append(rsample)
        
        for layer in self.reduced_layers:
            sample, mean, cov, kl = layer.propagate(samples[-1],
                                                    full_cov=True,
                                                    is_sampled_local_regularizer=is_sampled_local_regularizer,
                                                    inference_amortization_inputs=inference_amortization_inputs)
            samples.append(sample)
            means.append(mean)
            covs.append(cov)
            kls.append(kl)
            kl_types.append(layer.regularizer_type)
        

        return samples[1:], means, covs, kls, kl_types
        
    def maximum_log_likelihood_objective(self, data: AuxRegressionData) -> tf.Tensor:
        return self.elbo(data)
    
    def elbo(self, 
             data: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        W, Y, Wstd = data
        Wstd_tiled = tf.tile(Wstd[:, None, :], [1, self.num_samples, 1]) # N, S, R
        Y_tiled = tf.tile(Y[:, None, :], [1, self.num_samples, 1]) # N, S, Dout
        # get the latent sample
        lsamp, lmean, lcov, lkl = self.latent_layer.propagate(self.raxis,
                                                             inference_amortization_inputs=Wstd,
                                                             is_sampled_local_regularizer=True)
        
        global_kls = [kl for kl, t in zip(kls, kl_types) if t is layers.RegularizerType.GLOBAL]
        local_kls = [kl for kl, t in zip(kls, kl_types) if t is layers.RegularizerType.LOCAL]
        
        f_mean = means[-1]  # N, S, Dout
        f_var = tf.transpose(tf.linalg.diag_part(covs[-1]),[0,2,1])  # N, Dout, S, S -> N, Dout, S -> N, S, Dout
        L_N_S = self.likelihood.variational_expectations(f_mean, f_var, Y_tiled)
        
        if len(local_kls) > 0:
            local_kls_N_S_D = tf.concat(local_kls, -1)  # N, S, sum(Dlatents)
            local_kls_N_S = tf.reduce_sum(local_kls_N_S_D, -1)
            L_N_S -= local_kls_N_S  # N,S
        
        global_kl = tf.reduce_sum(global_kls) # a scalar
            
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, global_kl.dtype)
            minibatch_size = tf.cast(tf.shape(W)[0], global_kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, global_kl.dtype)

        # This line is replaced with tf.reduce_logsumexp in the IW case
        logp = tf.reduce_logsumexp(L_N_S, -1)

        return tf.reduce_sum(logp) * scale - global_kl
    
    def predict_f_multisample(self, X, S):
        X_tiled = tf.tile(X[None, :, :], [S, 1, 1])
        _, means, covs, _, _ = self.propagate(X_tiled)
        return means[-1], covs[-1]

    def predict_y_samples(self, X, S):
        X_tiled = tf.tile(X[None, :, :], [S, 1, 1])
        _, means, covs, _, _ = self.propagate(X_tiled)
        m, v = self.likelihood.predict_mean_and_var(means[-1], covs[-1])
        z = tf.random.normal(tf.shape(means[-1]), dtype=X.dtype)
        return m + z * v**0.5


class LVNGP_GPMM_IWVI_Tiled(gpflow.models.BayesianModel, gpflow.models.ExternalDataTrainingLossMixin):
    """
    LVNGP with GPMM likelihood. We prepare a latent input based on SASE spectrum and tile that input over
    the reduction axis. This is then fed to the GP that produces a latent spectrum, which will be reduced
    by the standard GPMM approach and compared to the output. We refer to latent inputs as being the result
    of a function w, which behaves like:
    
    w(N x R) -> N x Dlatent === l
    
    l is combined with the energy axis with something like a Kronecker product, where Dlatent and Dinput are
    treated as single elements and unpacked via concatenation after the kronecker product is 
    carried out. This is implemented by tiling and concatenation.
    
    e_axis = R x Din
    X = pseudo_kron(l,e_axis) = N x R x Din + Dlatent
      = tf.concatenate([tf.tile( l[:,None,:], [1,R,1]),
                        tf.tile( e_axis[None,:,:], [N,1,1])], -1)
                        
    To ease implementation, we approximate tiling of the amortized output by instead tiling input to 
    the amortization network, so:
    tf.tile( l[:,None,:], [1,R,1]) -> w(tf.tile(W[:,None,:], [1,R,1]))
    
    
    feeding this into the gp gives:
    
    F = f(X) = N x R x Dout
    
    To include multi-sampling, we must have full covariance over R and S, so we do:
    
    l = w(tf.tile(W[:,None,:], [1,S*R,1])) = N x SR x Dlatent
    e_axis = R x Din
    X = pseudo_kron(l,e_axis) = N x SR x Din + Dlatent
      = tf.concatenate([tf.tile( l, [1,R,1]),
                        tf.tile( e_axis[None,:,:], [N,S,1])], -1)
    
    F = f(X) = N x SR x Dout
    
    To carry out the reduction, must reshape the mean/cov to isolate R:
    
    F_mean_reshaped = tf.reshape(*, [N,S,R,Dout])
    F_cov_reshaped = tf.reshape(*, [N,Dout,S,R,S,R])
    
    
    F_mean_reduced = tf.einsum('nr,nsrd->nsd',W,F_mean_reshaped)
    F_cov_reduced = tf.einsum('nr,ndsrsq,nq->nsd'W,F_cov_reshaped,W)
    """
    def __init__(self,
                 likelihood: gpflow.likelihoods.Likelihood,
                 layers: List[layers.GPLayer],
                 reduction_axis_len: int,
                 *,
                 num_samples: int = 1,
                 num_data: Optional[int] = None,
                ):
        super().__init__()
        self.likelihood = likelihood
        self.num_samples = num_samples
        self.num_data = num_data
        self.reduction_axis_len = int(reduction_axis_len)
        reduction_axis = tf.convert_to_tensor(np.linspace(-np.pi,np.pi,self.reduction_axis_len)[:,None],
                                                  gpflow.config.default_float())
        self.raxis = gpflow.Parameter(reduction_axis, transform=None)
        gpflow.utilities.set_trainable(self.raxis, False)
        self.layers = layers
        
    def propagate(self, X: tf.Tensor,
                  full_cov: bool = False,
                  inference_amortization_inputs: Optional[tf.Tensor] = None,
                  is_sampled_local_regularizer: bool = False):
        samples, means, covs, kls, kl_types = [X, ], [], [], [], []

        for layer in self.layers:
            full_cov_local = full_cov
            sample, mean, cov, kl = layer.propagate(samples[-1],
                                                    full_cov=full_cov_local,
                                                    inference_amortization_inputs=inference_amortization_inputs,
                                                    is_sampled_local_regularizer=is_sampled_local_regularizer)
            samples.append(sample)
            means.append(mean)
            covs.append(cov)
            kls.append(kl)
            kl_types.append(layer.regularizer_type)

        return samples[1:], means, covs, kls, kl_types
        
    def maximum_log_likelihood_objective(self, data: AuxRegressionData) -> tf.Tensor:
        return self.elbo(data)
    
    def elbo(self, 
             data: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        We accept a triplet of inputs: SASE, Fluorescence, and Standardized SASE
        
        the last is used in the amortized latent inference network
        """
        W, Y, Wstd = data
        N = tf.shape(W)[0]
        R = tf.shape(W)[1]
        Dout = tf.shape(Y)[-1]
        S = self.num_samples
        Wstd_tiled = tf.tile(Wstd[:,None,:], [1,S*R,1])  # N, SR, R
        Y_tiled = tf.tile(Y[:,None,:], [1,S,1])  # N,S,Dout
        X = self.raxis
        X_tiled = tf.tile(X[None,:,:], [N,S,1])  # N,SR,Din
        
        # get the latent sample
        samples, means, covs, kls, kl_types = self.propagate(X_tiled,
                                                             full_cov=True,  # full_cov is over the S dim
                                                             inference_amortization_inputs=Wstd_tiled,
                                                             is_sampled_local_regularizer=True)

        local_kls = [kl for kl, t in zip(kls, kl_types) if t is layers.RegularizerType.LOCAL]
        global_kls = [kl for kl, t in zip(kls, kl_types) if t is layers.RegularizerType.GLOBAL]
        f_mean = tf.reshape(means[-1],[N,S,R,Dout])  # N, SR, Dout -> N, S, R, Dout
        f_cov = tf.reshape(covs[-1],[N,Dout,S,R,S,R])  # N, Dout, SR, SR - > N, Dout, S, R, S, R
        f_mean_reduced = tf.einsum('nr,nsrd->nsd',W,f_mean)
        f_var_reduced = tf.einsum('nr,ndsrsq,nq->nsd',W,f_cov,W)
        
        
        L_N_S = self.likelihood.variational_expectations(f_mean_reduced, f_var_reduced, Y_tiled)
        
        if len(local_kls) > 0:
            local_kls_N_SR_D = tf.concat(local_kls, -1)  # N, SR, sum(Dlatents)
            Dlatent = tf.shape(local_kls_N_SR_D)[-1]
            local_kls_N_S_R_D = tf.reshape(local_kls_N_SR_D, [N, S, R, Dlatent])
            local_kls_N_S = tf.reduce_sum(local_kls_N_S_R_D, [-1,-2])
            L_N_S -= local_kls_N_S  # N,S
        
        global_kl = tf.reduce_sum(global_kls) # a scalar
            
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, global_kl.dtype)
            minibatch_size = tf.cast(tf.shape(W)[0], global_kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, global_kl.dtype)

        # This line is replaced with tf.reduce_logsumexp in the IW case
        logp = tf.reduce_logsumexp(L_N_S, -1)

        return tf.reduce_sum(logp) * scale - global_kl
    
    def predict_f_multisample(self, X, S):
        X_tiled = tf.tile(X[None, :, :], [S, 1, 1])
        _, means, covs, _, _ = self.propagate(X_tiled)
        return means[-1], covs[-1]

    def predict_y_samples(self, X, S):
        X_tiled = tf.tile(X[None, :, :], [S, 1, 1])
        _, means, covs, _, _ = self.propagate(X_tiled)
        m, v = self.likelihood.predict_mean_and_var(means[-1], covs[-1])
        z = tf.random.normal(tf.shape(means[-1]), dtype=X.dtype)
        return m + z * v**0.5


class LVNGP_GPMM_IWVI_Tiled2(gpflow.models.BayesianModel, gpflow.models.ExternalDataTrainingLossMixin):
    """
    LVNGP with GPMM likelihood. We prepare a latent input based on SASE spectrum and tile that input over
    the reduction axis. This is then fed to the GP that produces a latent spectrum, which will be reduced
    by the standard GPMM approach and compared to the output. We refer to latent inputs as being the result
    of a function w, which behaves like:
    
    w(N x R) -> N x Dlatent === l
    
    l is combined with the energy axis with something like a Kronecker product, where Dlatent and Dinput are
    treated as single elements and unpacked via concatenation after the kronecker product is 
    carried out. This is implemented by tiling and concatenation.
    
    e_axis = R x Din
    X = pseudo_kron(l,e_axis) = N x R x Din + Dlatent
      = tf.concatenate([tf.tile( l[:,None,:], [1,R,1]),
                        tf.tile( e_axis[None,:,:], [N,1,1])], -1)
                        
    To ease implementation, we approximate tiling of the amortized output by instead tiling input to 
    the amortization network, so:
    tf.tile( l[:,None,:], [1,R,1]) -> w(tf.tile(W[:,None,:], [1,R,1]))
    
    
    feeding this into the gp gives:
    
    F = f(X) = N x R x Dout
    
    To include multi-sampling, we must have full covariance over R and S, so we do:
    
    l = w(tf.tile(W[:,None,:], [1,S*R,1])) = N x SR x Dlatent
    e_axis = R x Din
    X = pseudo_kron(l,e_axis) = N x SR x Din + Dlatent
      = tf.concatenate([tf.tile( l, [1,R,1]),
                        tf.tile( e_axis[None,:,:], [N,S,1])], -1)
    
    F = f(X) = N x SR x Dout
    
    To carry out the reduction, must reshape the mean/cov to isolate R:
    
    F_mean_reshaped = tf.reshape(*, [N,S,R,Dout])
    F_cov_reshaped = tf.reshape(*, [N,Dout,S,R,S,R])
    
    
    F_mean_reduced = tf.einsum('nr,nsrd->nsd',W,F_mean_reshaped)
    F_cov_reduced = tf.einsum('nr,ndsrsq,nq->nsd'W,F_cov_reshaped,W)
    """
    def __init__(self,
                 likelihood: gpflow.likelihoods.Likelihood,
                 layers: List[layers.GPLayer],
                 reduction_axis_len: int,
                 *,
                 num_samples: int = 1,
                 num_data: Optional[int] = None,
                ):
        super().__init__()
        self.likelihood = likelihood
        self.num_samples = num_samples
        self.num_data = num_data
        self.reduction_axis_len = int(reduction_axis_len)
        reduction_axis = tf.convert_to_tensor(np.linspace(-np.pi,np.pi,self.reduction_axis_len)[:,None],
                                                  gpflow.config.default_float())
        self.raxis = gpflow.Parameter(reduction_axis, transform=None)
        gpflow.utilities.set_trainable(self.raxis, False)
        self.layers = layers
        
    def propagate(self, X: tf.Tensor,
                  full_cov: bool = False,
                  inference_amortization_inputs: Optional[tf.Tensor] = None,
                  is_sampled_local_regularizer: bool = False):
        samples, means, covs, kls, kl_types = [X, ], [], [], [], []

        for layer in self.layers:
            full_cov_local = full_cov
            sample, mean, cov, kl = layer.propagate(samples[-1],
                                                    full_cov=full_cov_local,
                                                    inference_amortization_inputs=inference_amortization_inputs,
                                                    is_sampled_local_regularizer=is_sampled_local_regularizer)
            samples.append(sample)
            means.append(mean)
            covs.append(cov)
            kls.append(kl)
            kl_types.append(layer.regularizer_type)

        return samples[1:], means, covs, kls, kl_types
        
    def maximum_log_likelihood_objective(self, data: AuxRegressionData) -> tf.Tensor:
        return self.elbo(data)
    
    def elbo(self, 
             data: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        We accept a triplet of inputs: SASE, Fluorescence, and Standardized SASE
        
        the last is used in the amortized latent inference network
        """
        W, Y, Wstd = data
        N = tf.shape(W)[0]
        R = tf.shape(W)[1]
        Dout = tf.shape(Y)[-1]
        S = self.num_samples
        Wstd_tiled = tf.tile(Wstd[:,None,:], [1,S,1])  # N, S, R
        Y_tiled = tf.tile(Y[:,None,:], [1,S,1])  # N,S,Dout
        X = self.raxis
        X_tiled = tf.tile(X[None,:,:], [N,S,1])  # N,SR,Din
        
        # get the latent sample
        samples, means, covs, kls, kl_types = self.propagate(X_tiled,
                                                             full_cov=True,  # full_cov is over the S dim
                                                             inference_amortization_inputs=Wstd_tiled,
                                                             is_sampled_local_regularizer=True)

        local_kls = [kl for kl, t in zip(kls, kl_types) if t is layers.RegularizerType.LOCAL]
        global_kls = [kl for kl, t in zip(kls, kl_types) if t is layers.RegularizerType.GLOBAL]
        f_mean = tf.reshape(means[-1],[N,S,R,Dout])  # N, SR, Dout -> N, S, R, Dout
        f_cov = tf.reshape(covs[-1],[N,Dout,S,R,S,R])  # N, Dout, SR, SR - > N, Dout, S, R, S, R
        f_mean_reduced = tf.einsum('nr,nsrd->nsd',W,f_mean)
        f_var_reduced = tf.einsum('nr,ndsrsq,nq->nsd',W,f_cov,W)
        
        
        L_N_S = self.likelihood.variational_expectations(f_mean_reduced, f_var_reduced, Y_tiled)
        
        if len(local_kls) > 0:
            local_kls_N_SR_D = tf.concat(local_kls, -1)  # N, SR, sum(Dlatents)
            Dlatent = tf.shape(local_kls_N_SR_D)[-1]
            local_kls_N_S_R_D = tf.reshape(local_kls_N_SR_D, [N, S, R, Dlatent])
            local_kls_N_S = tf.reduce_sum(local_kls_N_S_R_D, [-1,-2])
            L_N_S -= local_kls_N_S  # N,S
        
        global_kl = tf.reduce_sum(global_kls) # a scalar
            
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, global_kl.dtype)
            minibatch_size = tf.cast(tf.shape(W)[0], global_kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, global_kl.dtype)

        # This line is replaced with tf.reduce_logsumexp in the IW case
        logp = tf.reduce_logsumexp(L_N_S, -1)

        return tf.reduce_sum(logp) * scale - global_kl
    
    def predict_f_multisample(self, X, S):
        X_tiled = tf.tile(X[None, :, :], [S, 1, 1])
        _, means, covs, _, _ = self.propagate(X_tiled)
        return means[-1], covs[-1]

    def predict_y_samples(self, X, S):
        X_tiled = tf.tile(X[None, :, :], [S, 1, 1])
        _, means, covs, _, _ = self.propagate(X_tiled)
        m, v = self.likelihood.predict_mean_and_var(means[-1], covs[-1])
        z = tf.random.normal(tf.shape(means[-1]), dtype=X.dtype)
        return m + z * v**0.5
