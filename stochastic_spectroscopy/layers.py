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
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gpflow
import enum
import collections
from typing import Callable, Optional, Tuple, TypeVar, Union, List
from gpflow.kernels import Kernel, MultioutputKernel
from gpflow.mean_functions import MeanFunction, Zero
from gpflow.inducing_variables import SeparateIndependentInducingVariables, SharedIndependentInducingVariables
from gpflow.kullback_leiblers import gauss_kl as gauss_kl_gpflow
import attr


# avoiding use of defaults kwarg, to keep compatibility with Python3.6
class RegularizerType(enum.Enum):
    LOCAL = 0
    GLOBAL = 1

def gauss_kl(q_mu, q_sqrt, K=None):
    """
    Wrapper for gauss_kl from gpflow that returns the negative log prob if q_sqrt is None. This can be  
    for use in HMC: all that is required is to set q_sqrt to None and this function substitues the
    negative log prob instead of the KL (so no need to set q_mu.prior = gpflow.priors.Gaussian(0, 1)). 
    Also, this allows the use of HMC in the unwhitened case. 
    """
    if q_sqrt is None:
        # return negative log prob with q_mu as 'x', with mean 0 and cov K (or I, if None)
        M, D = tf.shape(q_mu)[0], tf.shape(q_mu)[1]
        I = tf.eye(M, dtype=q_mu.dtype)

        if K is None:
            L = I
        else:
            L = tf.cholesky(K + I * gpflow.default_jitter())

        return -tf.reduce_sum(gpflow.logdensities.multivariate_normal(q_mu, tf.zeros_like(q_mu), L))

    else:
        # return kl
        return gauss_kl_gpflow(q_mu, q_sqrt, K=K)


class GPLayer(gpflow.Module):
    regularizer_type = RegularizerType.GLOBAL
    def __init__(self, 
                 kernel: gpflow.kernels.Kernel,
                 inducing: gpflow.inducing_variables.InducingVariables,
                 mean_func: gpflow.mean_functions.MeanFunction,
                 **kwargs
                ):
        super().__init__()
        """
        The range of supported options for sample conditional is not complete. The following do not
        work:
        
        LinearCoregionalization: in order to get the proper behavior, you must have a separate kernel
        for each independent GP. While some things evaluate with a single shared kernel, Kuu and Kuf
        do not work properly and treat the number of latent gps as the number of separate kernels.
        
        LinearCoregionalization / SharedIndependentInducingVariable: In this case, you can only eval-
        uate the propagation when full_cov = False. However, full_cov is possible if one uses
        SeparateIndependentInducingVariable.
        """
        
        assert issubclass(type(kernel), gpflow.kernels.Kernel)
        assert issubclass(type(inducing), gpflow.inducing_variables.InducingVariables)
        if not issubclass(type(kernel), gpflow.kernels.MultioutputKernel):
            self.num_latent_gps = 1
            self.output_dim = 1
        else:
            self.num_latent_gps = kernel.num_latent_gps
            if not len(kernel.kernels) == self.num_latent_gps:
                # we want to catch the error with LinearCoregionalization mentioned above
                raise ValueError(
                    f"number of kernels should match number of latent gps " \
                    f"({self.num_latent_gps}) got {len(kernel.kernels)} kernels"
                )
            if issubclass(type(kernel), gpflow.kernels.LinearCoregionalization):
                self.output_dim = kernel.W.shape[-2]
            else:
                self.output_dim = self.num_latent_gps
        self.kernel = kernel
        self.inducing = inducing
        if hasattr(self.inducing, 'inducing_variable_list'):
            # case for separate independent
                assert len(self.inducing.inducing_variable_list) == self.num_latent_gps, \
                            f"Got {len(self.inducing.inducing_variable_list)} inducing variables, " \
                            f"but expected {self.num_latent_gps} gps from kernel. These should match."
                self.in_features = self.inducing.inducing_variable_list[0].Z.shape[-1]
                self.num_inducing = self.inducing.inducing_variable_list[0].Z.shape[-2]
        elif hasattr(self.inducing, 'inducing_variable'):
            # case for shared independent
            self.in_features = self.inducing.inducing_variable.Z.shape[-1]
            self.num_inducing = self.inducing.inducing_variable.Z.shape[-2]
        else:
            self.in_features = self.inducing.Z.shape[-1]
            self.num_inducing = self.inducing.Z.shape[-2]
        assert issubclass(type(mean_func), gpflow.mean_functions.MeanFunction)
        self.mean = mean_func
        if type(mean_func) is gpflow.mean_functions.Linear:
            # more consistency checking
            assert self.mean.A.shape[-1] == self.output_dim
        
        # Now for the storage of variational parameters
        self.q_mu = gpflow.Parameter(np.zeros((self.num_inducing, self.num_latent_gps)), transform=None)
        init_sqrt = np.tile(np.eye(self.num_inducing)[None, :, :], [self.num_latent_gps, 1, 1])
        if 'scale_init_q_sqrt' in kwargs:
            init_sqrt *= kwargs['scale_init_q_sqrt']
        self.q_sqrt = gpflow.Parameter(init_sqrt, transform=gpflow.utilities.triangular())
    
    def propagate(self, F, num_samples=None, full_cov=False, **kwargs):
        # In Hugh's code, he forces one to use full_cov = False for the case of a MoK. This has the effect
        # that only the final layer uses full covariance (as his code uses a single output kernel for the
        # final layer). This is inspite of the fact that he passes full_cov=True to all layers in the IWVI
        # case. Since I don't want to hack GPFlow's conditional system, I will let the full_cov pass through
        # and manually implement his behavior in the model.
        samples, mean, cov = gpflow.conditionals.sample_conditional(F,
                                                self.inducing,
                                                self.kernel,
                                                self.q_mu,
                                                full_cov=full_cov,
                                                q_sqrt=self.q_sqrt,
                                                white=True,
                                                num_samples=num_samples,
                                               )
        kl = gauss_kl(self.q_mu, self.q_sqrt)
        mf = self.mean(F)
        if num_samples is not None:
            samples = samples + mf[...,None,:,:]
        else:
            samples = samples + mf
        mean = mean + mf
        return samples, mean, cov, kl
    
    def components(self, F, num_samples=None, full_cov=False, **kwargs):
        # In Hugh's code, he forces one to use full_cov = False for the case of a MoK. This has the effect
        # that only the final layer uses full covariance (as his code uses a single output kernel for the
        # final layer). This is inspite of the fact that he passes full_cov=True to all layers in the IWVI
        # case. Since I don't want to hack GPFlow's conditional system, I will let the full_cov pass through
        # and manually implement his behavior in the model.
        mean, cov = gpflow.conditionals.conditional(F,
                                                self.inducing,
                                                self.kernel,
                                                self.q_mu,
                                                full_cov=full_cov,
                                                q_sqrt=self.q_sqrt,
                                                white=True,
                                               )
        kl = gauss_kl(self.q_mu, self.q_sqrt)
        mf = self.mean(F)
        mean = mean + mf
        return mean, cov, kl


class Encoder(gpflow.Module):
    def __init__(self, latent_dim: int,
                 input_dim: int,
                 network_dims: int,
                 activation_func: Optional[Callable] = None):
        """
        Encoder that uses GPflow params to encode the features.
        Creates an MLP with input dimensions `input_dim` and produces
        2 * `latent_dim` outputs.
        :param latent_dim: dimension of the latent variable
        :param input_dim: the MLP acts on data of `input_dim` dimensions
        :param network_dims: dimensions of inner MLPs, e.g. [10, 20, 10]
        :param activation_func: TensorFlow operation that can be used
            as non-linearity between the layers (default: tanh).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.activation_func = activation_func or tf.nn.tanh

        self.layer_dims = [input_dim, *network_dims, latent_dim * 2]

        Ws, bs = [], []

        for input_dim, output_dim in zip(self.layer_dims[:-1], self.layer_dims[1:]):
            xavier_std = (2. / (input_dim + output_dim)) ** 0.5
            W = np.random.randn(input_dim, output_dim) * xavier_std
            Ws.append(gpflow.Parameter(W, dtype=gpflow.config.default_float()))
            bs.append(gpflow.Parameter(np.zeros(output_dim), dtype=gpflow.config.default_float()))

        self.Ws, self.bs = Ws, bs

    def __call__(self, Z) -> Tuple[tf.Tensor, tf.Tensor]:
        o = tf.ones_like(Z)[..., :1, :1]  # for correct broadcasting
        for i, (W, b, dim_in, dim_out) in enumerate(zip(self.Ws, self.bs, self.layer_dims[:-1], self.layer_dims[1:])):
            Z0 = tf.identity(Z)
            Z = tf.matmul(Z, o * W) + o * b

            if i < len(self.bs) - 1:
                Z = self.activation_func(Z)

            if dim_out == dim_in:  # skip connection
                Z += Z0

        means, log_chol_diag = tf.split(Z, 2, axis=-1)
        q_sqrt = tf.nn.softplus(log_chol_diag - 3.)  # bias it towards small vals at first
        q_mu = means
        return q_mu, q_sqrt


class SASEEncoder(gpflow.Module):
    def __init__(self, latent_dim: int,
                 input_dim: int,
                 network_dims: int,
                 activation_func: Optional[Callable] = None):
        """
        Encoder that uses GPflow params to encode the features.
        Creates an MLP with input dimensions `input_dim` and produces
        2 * `latent_dim` outputs. Unlike the standard encoder, this 
        expects an input of NR shape, and converts that to an output which is
        (N*R)L, where L is the latent dim.
        
        :param latent_dim: dimension of the latent variable, i.e L
        :param input_dim: the MLP acts on data of `input_dim` dimensions, i.e. R
        :param network_dims: dimensions of inner MLPs, e.g. [10, 20, 10]
        :param activation_func: TensorFlow operation that can be used
            as non-linearity between the layers (default: tanh).
        """
        super().__init__()
        self.latent_dim = tf.convert_to_tensor([latent_dim], tf.int32)
        self.activation_func = activation_func or tf.nn.tanh

        self.layer_dims = [input_dim, *network_dims, input_dim * latent_dim * 2]

        Ws, bs = [], []

        for input_dim, output_dim in zip(self.layer_dims[:-1], self.layer_dims[1:]):
            xavier_std = (2. / (input_dim + output_dim)) ** 0.5
            W = np.random.randn(input_dim, output_dim) * xavier_std
            Ws.append(gpflow.Parameter(W, dtype=gpflow.config.default_float()))
            bs.append(gpflow.Parameter(np.zeros(output_dim), dtype=gpflow.config.default_float()))

        self.Ws, self.bs = Ws, bs

    def __call__(self, Z) -> Tuple[tf.Tensor, tf.Tensor]:
        N = tf.convert_to_tensor([tf.shape(Z)[-2]], dtype=tf.int32)
        R = tf.convert_to_tensor([tf.shape(Z)[-1]], dtype=tf.int32)
        batch_shape = tf.convert_to_tensor(tf.shape(Z)[:-2], dtype=tf.int32)
        o = tf.ones_like(Z)[..., :1, :1]  # for correct broadcasting
        for i, (W, b, dim_in, dim_out) in enumerate(zip(self.Ws, self.bs, self.layer_dims[:-1], self.layer_dims[1:])):
            Z0 = tf.identity(Z)
            Z = tf.matmul(Z, o * W) + o * b

            if i < len(self.bs) - 1:
                Z = self.activation_func(Z)

            if dim_out == dim_in:  # skip connection
                Z += Z0

        means, log_chol_diag = tf.split(Z, 2, axis=-1)
        q_sqrt = tf.nn.softplus(log_chol_diag - 3.)  # bias it towards small vals at first
        q_mu = means #...N(L*R)
        q_mu_reshaped = tf.reshape(q_mu, tf.concat([batch_shape, N*R, self.latent_dim],0))  # ...(N*R)L
        q_sqrt_reshaped = tf.reshape(q_sqrt, tf.concat([batch_shape, N*R, self.latent_dim],0))
        return q_mu_reshaped, q_sqrt_reshaped


class EmbeddingEncoder(gpflow.Module):
    def __init__(self, latent_dim: int,
                 nwords: int):
        """
        Here we simply pass a shot index to an embedding lookup. This allows us to 
        randomly access a tensor more easily, but basically we're just storing latent
        values for each shot.
        """
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.embedding = gpflow.Parameter(np.random.randn(nwords, 2*latent_dim),
                                         dtype=gpflow.config.default_float())
    @tf.function
    def __call__(self, Z: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        Zenc = tf.nn.embedding_lookup(self.embedding, tf.squeeze(Z,-1))
        means, log_chol_diag = tf.split(Zenc, 2, axis=-1)
        q_sqrt = tf.nn.softplus(log_chol_diag - 3.)  # bias it towards small vals at first
        q_mu = means
        return q_mu, q_sqrt


class AmortizedLatentVariableLayer(gpflow.Module):
    regularizer_type = RegularizerType.LOCAL
    def __init__(self, latent_dim: int,
                 XY_dim: Optional[int] = None,
                 encoder: Optional[Callable] = None):
        super().__init__()
        self.latent_dim = latent_dim
        if encoder is None:
            assert XY_dim, 'must pass XY_dim or else an encoder'
            encoder = Encoder(latent_dim, XY_dim, [20, 20])
        self.encoder = encoder

    def propagate(self, F: tf.Tensor,
                  inference_amortization_inputs: Optional[tf.Tensor] = None,
                  is_sampled_local_regularizer: bool = False,
                  **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        if inference_amortization_inputs is None:
            """
            If there isn't an X and Y passed for the recognition model, this samples from the prior.
            Optionally, q_mu and q_sqrt can be fed via a placeholder (e.g. for plotting purposes)
            """
            shape = tf.concat([F.shape[:-1], tf.TensorShape([self.latent_dim])], 0)
            q_mu = tf.zeros(shape, dtype=gpflow.config.default_float())
            q_sqrt = tf.ones(shape, dtype=gpflow.config.default_float())
        else:
            q_mu, q_sqrt = self.encoder(inference_amortization_inputs)

        # reparameterization trick to take a sample for W
        eps = tf.random.normal(tf.shape(q_mu), dtype=gpflow.config.default_float())
        W = q_mu + eps * q_sqrt

        samples = tf.concat([F, W], -1)
        mean = tf.concat([F, q_mu], -1)
        cov = tf.concat([tf.zeros_like(F), q_sqrt ** 2], -1)

        # the prior regularization
        p = p = tfp.distributions.Normal(loc=tf.zeros(1,dtype=gpflow.config.default_float()),
                             scale=tf.ones(1,dtype=gpflow.config.default_float()))
        q = tfp.distributions.Normal(loc=q_mu, scale=q_sqrt)

        if is_sampled_local_regularizer:
            # for the IW models, we need to return a log q/p for each sample W
            kl = q.log_prob(W) - p.log_prob(W)
        else:
            # for the VI models, we want E_q log q/p, which is closed form for Gaussians
            kl = tfp.distributions.kl_divergence(q, p)

        return samples, mean, cov, kl


class AmortizedSASELatentVariableLayer(gpflow.Module):
    regularizer_type = RegularizerType.LOCAL
    def __init__(self, latent_dim: int,
                 sase_dim: int,
                 encoder: Optional[Callable] = None):
        super().__init__()
        if encoder is None:
            encoder = SASEEncoder(latent_dim, sase_dim, [50, 10, 50])
        self.latent_dim = tf.convert_to_tensor([latent_dim], dtype=tf.int32)
        self.sase_dim = tf.convert_to_tensor([sase_dim], dtype=tf.int32)
        self.encoder = encoder

    def propagate(self, F: tf.Tensor,
                  inference_amortization_inputs: Optional[tf.Tensor] = None,
                  is_sampled_local_regularizer: bool = False,
                  **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        if inference_amortization_inputs is None:
            """
            If there isn't a SASE spec passed for the recognition model, this samples from the prior.
            Optionally, q_mu and q_sqrt can be fed via a placeholder (e.g. for plotting purposes)
            """
            batch_shape = tf.convert_to_tensor(tf.shape(F)[:-2], dtype=tf.int32) # ...
            N = tf.convert_to_tensor([tf.shape(F)[-2]], dtype=tf.int32)
            shape = tf.concat([batch_shape, N, self.latent_dim], 0) # ...(N)L
            q_mu = tf.zeros(shape, dtype=gpflow.config.default_float())
            q_sqrt = tf.ones(shape, dtype=gpflow.config.default_float())
        else:
            q_mu, q_sqrt = self.encoder(inference_amortization_inputs)

        # reparameterization trick to take a sample for W
        eps = tf.random.normal(tf.shape(q_mu), dtype=gpflow.config.default_float())
        W = q_mu + eps * q_sqrt

        samples = tf.concat([F, W], -1)
        mean = tf.concat([F, q_mu], -1)
        cov = tf.concat([tf.zeros_like(F), q_sqrt ** 2], -1)

        # the prior regularization
        p = p = tfp.distributions.Normal(loc=tf.zeros(1,dtype=gpflow.config.default_float()),
                             scale=tf.ones(1,dtype=gpflow.config.default_float()))
        q = tfp.distributions.Normal(loc=q_mu, scale=q_sqrt)

        if is_sampled_local_regularizer:
            # for the IW models, we need to return a log q/p for each sample W
            kl = q.log_prob(W) - p.log_prob(W)
        else:
            # for the VI models, we want E_q log q/p, which is closed form for Gaussians
            kl = tfp.distributions.kl_divergence(q, p)

        return samples, mean, cov, kl


class AmortizedEmbeddingLatentVariableLayer(gpflow.Module):
    regularizer_type = RegularizerType.LOCAL
    def __init__(self, latent_dim: int, nwords: int, nembed: int = 10, nhidden: int = 20):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = EmbeddingEncoder(latent_dim, nwords)

    def propagate(self, F: tf.Tensor,
                  inference_amortization_inputs: Optional[tf.Tensor] = None,
                  is_sampled_local_regularizer: bool = False,
                  **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        if inference_amortization_inputs is None:
            """
            If there isn't an X and Y passed for the recognition model, this samples from the prior.
            Optionally, q_mu and q_sqrt can be fed via a placeholder (e.g. for plotting purposes)
            """
            shape = tf.concat([F.shape[:-1], tf.TensorShape([self.latent_dim])], 0)
            q_mu = tf.zeros(shape, dtype=gpflow.config.default_float())
            q_sqrt = tf.ones(shape, dtype=gpflow.config.default_float())
        else:
            q_mu, q_sqrt = self.encoder(inference_amortization_inputs)

        # reparameterization trick to take a sample for W
        eps = tf.random.normal(tf.shape(q_mu), dtype=gpflow.config.default_float())
        W = q_mu + eps * q_sqrt

        samples = tf.concat([F, W], -1)
        mean = tf.concat([F, q_mu], -1)
        cov = tf.concat([tf.zeros_like(F), q_sqrt ** 2], -1)

        # the prior regularization
        p = p = tfp.distributions.Normal(loc=tf.zeros(1,dtype=gpflow.config.default_float()),
                             scale=tf.ones(1,dtype=gpflow.config.default_float()))
        q = tfp.distributions.Normal(loc=q_mu, scale=q_sqrt)

        if is_sampled_local_regularizer:
            # for the IW models, we need to return a log q/p for each sample W
            kl = q.log_prob(W) - p.log_prob(W)
        else:
            # for the VI models, we want E_q log q/p, which is closed form for Gaussians
            kl = tfp.distributions.kl_divergence(q, p)

        return samples, mean, cov, kl


class SASEReducedEncoder(gpflow.Module):
    def __init__(self, latent_dim: int,
                 input_dim: int,
                 network_dims: int,
                 activation_func: Optional[Callable] = None):
        """
        Encoder that uses GPflow params to encode the features.
        Creates an MLP with input dimensions `input_dim` and produces
        2 * `latent_dim` outputs. Unlike the standard encoder, this 
        expects an input of NR shape, and converts that to an output which is
        (N*R)L, where L is the latent dim.
        
        :param latent_dim: dimension of the latent variable, i.e L
        :param input_dim: the MLP acts on data of `input_dim` dimensions, i.e. R
        :param network_dims: dimensions of inner MLPs, e.g. [10, 20, 10]
        :param activation_func: TensorFlow operation that can be used
            as non-linearity between the layers (default: tanh).
        """
        super().__init__()
        self.latent_dim = tf.convert_to_tensor([latent_dim], tf.int32)
        self.activation_func = activation_func or tf.nn.tanh

        self.layer_dims = [input_dim, *network_dims, latent_dim * 2]

        Ws, bs = [], []

        for input_dim, output_dim in zip(self.layer_dims[:-1], self.layer_dims[1:]):
            xavier_std = (2. / (input_dim + output_dim)) ** 0.5
            W = np.random.randn(input_dim, output_dim) * xavier_std
            Ws.append(gpflow.Parameter(W, dtype=gpflow.config.default_float()))
            bs.append(gpflow.Parameter(np.zeros(output_dim), dtype=gpflow.config.default_float()))

        self.Ws, self.bs = Ws, bs

    def __call__(self, Z) -> Tuple[tf.Tensor, tf.Tensor]:
        N = tf.convert_to_tensor([tf.shape(Z)[-2]], dtype=tf.int32)
        R = tf.convert_to_tensor([tf.shape(Z)[-1]], dtype=tf.int32)
        batch_shape = tf.convert_to_tensor(tf.shape(Z)[:-2], dtype=tf.int32)
        o = tf.ones_like(Z)[..., :1, :1]  # for correct broadcasting
        for i, (W, b, dim_in, dim_out) in enumerate(zip(self.Ws, self.bs, self.layer_dims[:-1], self.layer_dims[1:])):
            Z0 = tf.identity(Z)
            Z = tf.matmul(Z, o * W) + o * b

            if i < len(self.bs) - 1:
                Z = self.activation_func(Z)

            if dim_out == dim_in:  # skip connection
                Z += Z0

        means, log_chol_diag = tf.split(Z, 2, axis=-1)
        q_sqrt = tf.nn.softplus(log_chol_diag - 3.)  # bias it towards small vals at first
        q_mu = means #...N(L*R)
        return q_mu, q_sqrt


class AmortizedSASEReducedLatentVariableLayer(gpflow.Module):
    regularizer_type = RegularizerType.LOCAL
    def __init__(self, latent_dim: int,
                 sase_dim: int,
                 encoder: Optional[Callable] = None):
        super().__init__()
        if encoder is None:
            encoder = SASEReducedEncoder(latent_dim, sase_dim, [50, 10, 10])
        self.latent_dim = tf.convert_to_tensor([latent_dim], dtype=tf.int32)
        self.sase_dim = tf.convert_to_tensor([sase_dim], dtype=tf.int32)
        self.encoder = encoder

    def propagate(self, F: tf.Tensor,
                  inference_amortization_inputs: Optional[tf.Tensor] = None,
                  is_sampled_local_regularizer: bool = False,
                  **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        if inference_amortization_inputs is None:
            """
            If there isn't a SASE spec passed for the recognition model, this samples from the prior.
            Optionally, q_mu and q_sqrt can be fed via a placeholder (e.g. for plotting purposes)
            """
            batch_shape = tf.convert_to_tensor(tf.shape(F)[:-2], dtype=tf.int32) # ...
            N = tf.convert_to_tensor([tf.shape(F)[-2]], dtype=tf.int32)
            shape = tf.concat([batch_shape, N, self.latent_dim], 0) # ...(N)L
            q_mu = tf.zeros(shape, dtype=gpflow.config.default_float())
            q_sqrt = tf.ones(shape, dtype=gpflow.config.default_float())
        else:
            q_mu, q_sqrt = self.encoder(inference_amortization_inputs)

        # reparameterization trick to take a sample for W
        eps = tf.random.normal(tf.shape(q_mu), dtype=gpflow.config.default_float())
        W = q_mu + eps * q_sqrt
        samples = tf.concat([F, W], -1)
        mean = tf.concat([F, q_mu], -1)
        cov = tf.concat([tf.zeros_like(F), q_sqrt ** 2], -1)

        # the prior regularization
        p = p = tfp.distributions.Normal(loc=tf.zeros(1,dtype=gpflow.config.default_float()),
                             scale=tf.ones(1,dtype=gpflow.config.default_float()))
        q = tfp.distributions.Normal(loc=q_mu, scale=q_sqrt)

        if is_sampled_local_regularizer:
            # for the IW models, we need to return a log q/p for each sample W
            kl = q.log_prob(W) - p.log_prob(W)
        else:
            # for the VI models, we want E_q log q/p, which is closed form for Gaussians
            kl = tfp.distributions.kl_divergence(q, p)

        return samples, mean, cov, kl


class AmortizedLatentVariableLayer2(gpflow.Module):
    regularizer_type = RegularizerType.LOCAL
    def __init__(self, latent_dim: int,
                 sase_dim: int,
                 encoder_dims: Optional[List[int]] = None):
        super().__init__()
        if encoder_dims is None:
            encoder = Encoder(latent_dim, sase_dim, [50, 10, 10])
        else:
            encoder = Encoder(latent_dim, sase_dim, encoder_dims)
        self.latent_dim = tf.convert_to_tensor([latent_dim], dtype=tf.int32)
        self.sase_dim = tf.convert_to_tensor([sase_dim], dtype=tf.int32)
        self.encoder = encoder

    def propagate(self, F: tf.Tensor,
                  inference_amortization_inputs: Optional[tf.Tensor] = None,
                  is_sampled_local_regularizer: bool = False,
                  **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        if inference_amortization_inputs is None:
            """
            If there isn't a SASE spec passed for the recognition model, this samples from the prior.
            Optionally, q_mu and q_sqrt can be fed via a placeholder (e.g. for plotting purposes)
            """
            batch_shape = tf.convert_to_tensor(tf.shape(F)[:-2], dtype=tf.int32) # ...
            N = tf.convert_to_tensor([tf.shape(F)[-2]], dtype=tf.int32)
            shape = tf.concat([batch_shape, N, self.latent_dim], 0) # ...(N)L
            q_mu = tf.zeros(shape, dtype=gpflow.config.default_float())
            q_sqrt = tf.ones(shape, dtype=gpflow.config.default_float())
        else:
            q_mu, q_sqrt = self.encoder(inference_amortization_inputs)

        # reparameterization trick to take a sample for W
        eps = tf.random.normal(tf.shape(q_mu), dtype=gpflow.config.default_float())
        W = q_mu + eps * q_sqrt
        samples = tf.concat([F, W], -1)
        mean = tf.concat([F, q_mu], -1)
        cov = tf.concat([tf.zeros_like(F), q_sqrt ** 2], -1)
        
        #### HAHAHASDFDDADSF
        ##### AGGHHH NOTICE THE SCALE ON THE KL
        ##### ITS NOT 1!!!!!

        # the prior regularization
        p = p = tfp.distributions.Normal(loc=tf.zeros(1,dtype=gpflow.config.default_float()),
                             scale=tf.ones(1,dtype=gpflow.config.default_float()))
        q = tfp.distributions.Normal(loc=q_mu, scale=q_sqrt)

        if is_sampled_local_regularizer:
            # for the IW models, we need to return a log q/p for each sample W
            kl = q.log_prob(W) - p.log_prob(W)
        else:
            # for the VI models, we want E_q log q/p, which is closed form for Gaussians
            kl = tfp.distributions.kl_divergence(q, p)

        return samples, mean, cov, kl


class AmortizedLatentVariableLayerTiled(gpflow.Module):
    regularizer_type = RegularizerType.LOCAL
    def __init__(self, latent_dim: int,
                 sase_dim: int,
                 encoder_dims: Optional[List[int]] = None):
        super().__init__()
        if encoder_dims is None:
            encoder = Encoder(latent_dim, sase_dim, [50, 10, 10])
        else:
            encoder = Encoder(latent_dim, sase_dim, encoder_dims)
        self.latent_dim = tf.convert_to_tensor([latent_dim], dtype=tf.int32)
        self.sase_dim = tf.convert_to_tensor([sase_dim], dtype=tf.int32)
        self.encoder = encoder

    def propagate(self, F: tf.Tensor,
                  inference_amortization_inputs: Optional[tf.Tensor] = None,
                  is_sampled_local_regularizer: bool = False,
                  **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        if inference_amortization_inputs is None:
            """
            If there isn't a SASE spec passed for the recognition model, this samples from the prior.
            Optionally, q_mu and q_sqrt can be fed via a placeholder (e.g. for plotting purposes)
            """
            batch_shape = tf.convert_to_tensor(tf.shape(F)[:-3], dtype=tf.int32) # ...
            N = tf.convert_to_tensor([tf.shape(F)[-3]], dtype=tf.int32)
            S = tf.convert_to_tensor([tf.shape(F)[-2]//self.sase_dim[0]], dtype=tf.int32)
            shape = tf.concat([batch_shape, N, S, self.latent_dim], 0) # ...(N)L
            q_mu = tf.zeros(shape, dtype=gpflow.config.default_float())
            q_sqrt = tf.ones(shape, dtype=gpflow.config.default_float())
        else:
            q_mu, q_sqrt = self.encoder(inference_amortization_inputs)

        # reparameterization trick to take a sample for W
        eps = tf.random.normal(tf.shape(q_mu), dtype=gpflow.config.default_float())
        W = q_mu + eps * q_sqrt
        tile_vec = tf.concat([tf.convert_to_tensor([1], dtype=tf.int32),
                              self.sase_dim, tf.convert_to_tensor([1], dtype=tf.int32)],0)
        TW = tf.tile(W,tile_vec)
        Tmu = tf.tile(q_mu, tile_vec)
        Tsqrt = tf.tile(q_sqrt, tile_vec)
        samples = tf.concat([F, TW], -1)
        mean = tf.concat([F, Tmu], -1)
        cov = tf.concat([tf.zeros_like(F), Tsqrt ** 2], -1)

        # the prior regularization
        p = p = tfp.distributions.Normal(loc=tf.zeros(1,dtype=gpflow.config.default_float()),
                             scale=0.1*tf.ones(1,dtype=gpflow.config.default_float()))
        q = tfp.distributions.Normal(loc=Tmu, scale=Tsqrt)

        if is_sampled_local_regularizer:
            # for the IW models, we need to return a log q/p for each sample W
            kl = q.log_prob(TW) - p.log_prob(TW)
        else:
            # for the VI models, we want E_q log q/p, which is closed form for Gaussians
            kl = tfp.distributions.kl_divergence(q, p)

        return samples, mean, cov, kl


# +
@attr.s(auto_attribs=True)
class GPLayer_Config(object):
    ngps: int
    ninducing: int
        
@attr.s(auto_attribs=True)
class LatentLayer_Config(object):
    latent_features: int
    xy_dim: int
        
@attr.s(auto_attribs=True)
class EmbeddingLatentLayer_Config(object):
    latent_features: int
    nwords: int
# -


