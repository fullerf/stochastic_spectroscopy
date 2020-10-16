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
sys.path.append('/home/fdfuller/work/dgp_iwvi_gpflow2/')
import gpflow
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, List, Union
import dgp_iwvi_gpflow2.layers as layers
from dgp_iwvi_gpflow2.reference_spectra import *
import attr
import tensorflow_probability as tfp
from matplotlib.pyplot import *
from gpflow.utilities import print_summary
import h5py
from sklearn.neighbors import KernelDensity
# %matplotlib notebook

__all__ = ['GPMM1D_Exact', 'GPMM2D_Exact']

RegressionData = Tuple[tf.Tensor, tf.Tensor]
AuxRegressionData = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
InputData = tf.Tensor
MeanVarKL = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]


class GPMM1D_Exact(gpflow.models.BayesianModel, gpflow.models.InternalDataTrainingLossMixin):
    r"""
    Gaussian Process Regression with GPMM Likelihood, implemented in a way to exploit linear 
    identities that enable efficient evaluation.
    """

    def __init__(
        self,
        data: Tuple[tf.Tensor, tf.Tensor],
        kernel: gpflow.kernels.Kernel,
        mean_function: Optional[gpflow.mean_functions.MeanFunction] = None,
        noise_variance: float = 1.0,
        jitter: float = 1e-8,
    ):
        super().__init__()
        self.likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        self.kernel = kernel
        self._pred_jitter_kernel = gpflow.kernels.White(variance=jitter) # needed in prediction only
        gpflow.set_trainable(self._pred_jitter_kernel.variance, False)
        W, y = data
        R = W.shape[-1]
        raxis = np.linspace(-1.0, 1.0, R)[:,None]
        self.raxis = tf.convert_to_tensor(raxis, dtype=gpflow.default_float())
        self.data = tuple([tf.convert_to_tensor(d, dtype=gpflow.config.default_float()) for d in data])
        # reassign for convenience now that we're in tensorflow mode
        W, Y = self.data
        self._WTW = tf.matmul(W,W, transpose_a=True)
        self._WTy = tf.matmul(W,y, transpose_a=True)
        self._yTy = tf.matmul(y,y, transpose_a=True)
        self.N = tf.cast(tf.shape(W)[0], dtype=gpflow.default_float())
        if mean_function is None:
            self.prior_mean_func = gpflow.mean_functions.Zero()
            self.zpm = True
        else:
            self.prior_mean_func = mean_function
            self.zpm = False
        self.jitter = tf.cast(jitter, dtype=gpflow.default_float())
        self.log2pi = tf.cast(np.log(np.pi*2), dtype=gpflow.default_float())
        
    @tf.function
    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_marginal_likelihood()

    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.
        .. math::
            \log p(Y | \theta).
        """
        R = tf.shape(self.raxis)[0]
        prior_mean = self.prior_mean_func(self.raxis)
        Krr = self.kernel(self.raxis, full_cov=True)
        Kdiag = tf.linalg.diag_part(Krr)
        jitter_vec = tf.fill([tf.shape(Kdiag)[-1]], self.jitter)
        K = tf.linalg.set_diag(Krr, Kdiag + jitter_vec)
        L = tf.linalg.cholesky(K)
        inv_var = 1/self.likelihood.variance
        inv_var_squared = tf.square(inv_var)
        sigma = tf.sqrt(self.likelihood.variance)
        B = tf.linalg.eye(R, dtype=gpflow.default_float()) + inv_var*tf.matmul(tf.matmul(L, self._WTW, transpose_a=True),L)
        LB = tf.linalg.cholesky(B)
        A = tf.linalg.triangular_solve(LB, tf.matmul(L, self._WTW, transpose_a=True), lower=True, adjoint=False)
        yB = tf.linalg.triangular_solve(LB, tf.matmul(L, self._WTy, transpose_a=True), lower=True, adjoint=False)
        if not self.zpm:
            t1a = -0.5*inv_var*( self._yTy - 
                                    2.0*tf.matmul(prior_mean, self._WTy, transpose_a=True) + 
                                    tf.matmul(tf.matmul(prior_mean, self._WTW, transpose_a=True), prior_mean)
                               )
            Am = tf.matmul(A, prior_mean)
            t1b = 0.5*inv_var_squared*( tf.matmul(yB, yB, transpose_a=True) - 
                                       2.0*tf.matmul(Am, yB, transpose_a=True) + 
                                       tf.matmul(Am, Am, transpose_a=True) )
        else:
            t1a = -0.5*inv_var*( self._yTy)
            t1b = 0.5*inv_var_squared*(tf.matmul(yB, yB, transpose_a=True))
        t1 = tf.reduce_sum(t1a + t1b)
        t2 = self.N*tf.math.log(sigma) + tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
#         t3 = 0.5*self.N*self.log2pi # constant wrt to parameters, so can be dropped.
        return (t1 - t2)# + t3
    def predict_f(self, raxis: Optional[tf.Tensor] = None,
                        full_cov: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        This method computes predictions at the requested axes if supplied or at the training axes if not
        """
        if raxis is None:
            raxis = self.raxis
        R = tf.shape(raxis)[0]
        prior_mean = self.prior_mean_func(raxis)
        prior_mean_train = self.prior_mean_func(self.raxis)
        inv_var = 1/self.likelihood.variance
        inv_var_squared = tf.square(inv_var)
        log_var = tf.math.log(self.likelihood.variance)
        # kernels needed for prediction to potentially new axes
        # read Ksfr as reduced_kernel(prediction_points, training_points)
        # read Ksfq as observed_kernel(prediction_points, training_points)
        Ksfr = self.kernel(raxis, self.raxis, full_cov=True) + \
                self._pred_jitter_kernel(raxis, self.raxis, full_cov=True)
        Kssr = self.kernel(raxis, full_cov=full_cov)
        # kernel bits from training
        Krr = self.kernel(self.raxis, full_cov=True)
#         Krr += self.jitter * tf.eye(tf.shape(Krr)[-1], dtype=gpflow.config.default_float())
#         Kqq += self.jitter * tf.eye(tf.shape(Kqq)[-1], dtype=gpflow.config.default_float())
#         Lr = tf.linalg.cholesky(Krr)
#         Lq = tf.linalg.cholesky(Kqq)
        Krr_diag = tf.linalg.diag_part(Krr)
        jitter_vec_r = tf.fill([tf.shape(Krr_diag)[-1]], self.jitter)
        Krr_full_rank = tf.linalg.set_diag(Krr, Krr_diag + jitter_vec_r)
        Lr = tf.linalg.cholesky(Krr_full_rank)

        B0 = inv_var * tf.matmul(tf.matmul(Lr, self._WTW, transpose_a=True), Lr)
        B0_diag = tf.linalg.diag_part(B0)
        ones_r = tf.fill([tf.shape(B0_diag)[-1]], tf.convert_to_tensor(1., dtype=gpflow.default_float()))
        B = tf.linalg.set_diag(B0, B0_diag + ones_r)
        Lb = tf.linalg.cholesky(B)
        yb = tf.linalg.triangular_solve(Lb, tf.matmul(Lr, self._WTy, transpose_a=True), lower=True, adjoint=False)
        Lbinvt_yb = tf.linalg.triangular_solve(Lb, yb, lower=True, adjoint=True)        
        tm1 = inv_var * tf.matmul(Ksfr, self._WTy)
        tm2 = inv_var_squared * tf.matmul(Ksfr, tf.matmul(self._WTW, tf.matmul(Lr, Lbinvt_yb)))
        tm3 = prior_mean - inv_var*tf.matmul(Ksfr, tf.matmul(self._WTW, prior_mean_train))
        mb = tf.linalg.triangular_solve(Lb, tf.matmul(tf.matmul(Lr, self._WTW, transpose_a=True), prior_mean_train), 
                                        lower=True, adjoint=False)
        Lbinvt_mb = tf.linalg.triangular_solve(Lb, mb, lower=True, adjoint=True)
        tm4 = inv_var_squared * tf.matmul(Ksfr, tf.matmul(self._WTW, tf.matmul(Lr, Lbinvt_mb)))
        m = tm1 - tm2 + tm3 + tm4
        A = tf.linalg.triangular_solve(Lb, tf.matmul(Lr, self._WTW, transpose_a=True), lower=True, adjoint=False)
        V = tf.matmul(A, Ksfr, transpose_b=True)
        if not full_cov:
            v = Kssr - \
                inv_var*tf.linalg.diag_part(tf.matmul(Ksfr, tf.matmul(self._WTW, Ksfr, transpose_b=True))) + \
                inv_var_squared*tf.linalg.diag_part(tf.matmul(V, V, transpose_a=True))
        else:
            v = Kssr - \
                inv_var*tf.matmul(Ksfr, tf.matmul(self._WTW, Ksfr, transpose_b=True)) + \
                inv_var_squared*tf.matmul(V, V, transpose_a=True)
        return m, v


def batched_vec_identity(A: tf.Tensor,B: tf.Tensor,C: tf.Tensor) -> tf.Tensor:
    """
    For matrices, A, B, and C (i.e. 2D tensors):
    
    implements kron(A,B) @ C, where C is interpreted as a "batched vector" with
    the batch dim being the columns. C must have rows = A.shape[-1]*B.shape[-1], 
    but can have any number of columns.
    
    The normal vec identity vec(is B @ vec^-1(C) @ A.T) = kron(A,B) @ C
    for row major vec identity it is vec(is A @ vec^-1(C) @ B.T) = kron(A,B) @ C
    as tf is row major, we use the latter.
    
    to batch this, we reshape C like: (C.T).reshape(C.shape[-1], A.shape[-1], B.shape[-1])
    The matrix multiplication is carried out with broadcasting via einsum:
    
    result = tf.einsum('...ij,zjk,...lk->ilz').reshape(A.shape[0]*B.shape[0],C.shape[-1])
    """
    L = tf.shape(B)[0]
    K = tf.shape(B)[1]
    I = tf.shape(A)[0]
    J = tf.shape(A)[1]
    Z = tf.shape(C)[1]
    Cr = tf.reshape(tf.transpose(C), [Z, J, K])
    A = tf.tile(A[None,:,:], [Z,1,1])
    B = tf.tile(B[None,:,:], [Z,1,1])
    return tf.reshape(tf.einsum('aij,ajk,alk->ila',A,Cr,B), [I*L,Z])


def kron(op1: tf.Tensor, op2: tf.Tensor) -> tf.Tensor:
    """
    This version uses the LinearOperatorKronecker function built into TensorFlow
    """
    lop1 = tf.linalg.LinearOperatorFullMatrix(op1)
    lop2 = tf.linalg.LinearOperatorFullMatrix(op2)
    return tf.linalg.LinearOperatorKronecker([lop1,lop2]).to_dense()


def cartesian_prod(*ops):
    mops = tf.meshgrid(*ops,indexing='ij')
    rmops = [tf.reshape(op,[-1,1]) for op in mops]
    return tf.concat(rmops,-1)


class GPMM2D_Exact(gpflow.models.BayesianModel, gpflow.models.InternalDataTrainingLossMixin):
    r"""
    Gaussian Process Regression with GPMM Likelihood, for RIXS (2D)
    """

    def __init__(
        self,
        data: Tuple[tf.Tensor, tf.Tensor],
        reduced_kernel: gpflow.kernels.Kernel,
        observed_kernel: gpflow.kernels.Kernel,
        mean_function: Optional[gpflow.mean_functions.MeanFunction] = None,
        noise_variance: float = 1.0,
        jitter: float = 1e-8,
    ):
        super().__init__()
        self.likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        self.reduced_kernel = reduced_kernel # this will be the kernel for absorption axis
        self.observed_kernel = observed_kernel # this is the kernel for fluorescence axis
        self._pred_jitter_kernel = gpflow.kernels.White(variance=jitter) # needed in prediction only
        gpflow.set_trainable(self._pred_jitter_kernel.variance, False)
        W, Y = data
        R = W.shape[-1]  # the reduced axis length
        Q = Y.shape[-1]  # the observed axis length
        raxis = np.linspace(-1.0, 1.0, R)[:,None]  # reduced axis
        qaxis = np.linspace(-1.0, 1.0, Q)[:,None]  # observed axis
        self.raxis = tf.convert_to_tensor(raxis, dtype=gpflow.default_float())
        self.qaxis = tf.convert_to_tensor(qaxis, dtype=gpflow.default_float())
        # full_axis = 2D axis ordered to match kron(reduced_axis, observed_axis)
        self.full_axis = cartesian_prod(self.raxis, self.qaxis)  
        self.data = tuple([tf.convert_to_tensor(d, dtype=gpflow.config.default_float()) for d in data])
        # reassign for convenience now that we're in tensorflow mode
        W, Y = self.data
        # we need to store W for prediction, but not training
        self._W = W
        # make some cached quantities
        self._WTW = tf.matmul(W,W, transpose_a=True)
        self._YTW = tf.matmul(Y,W, transpose_a=True)
        self._yTy = tf.reduce_sum(Y*Y) # equivalent to fvec(Y).T @ fvec(Y)
        self._N = tf.cast(tf.shape(W)[0], dtype=gpflow.default_float())
        self._R = tf.cast(tf.shape(W)[-1], dtype=gpflow.default_float())
        self._Q = tf.cast(tf.shape(Y)[-1], dtype=gpflow.default_float())
        if mean_function is None:
            self.prior_mean_func = gpflow.mean_functions.Zero()
            self.zpm = True
        else:
            self.prior_mean_func = mean_function
            self.zpm = False
        self.jitter = tf.cast(jitter, dtype=gpflow.default_float())
        self.log2pi = tf.cast(np.log(np.pi*2), dtype=gpflow.default_float())
        
    @tf.function
    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_marginal_likelihood()

    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.
        .. math::
            \log p(Y | \theta).
        """
        R = tf.shape(self.raxis)[0]
        Q = tf.shape(self.qaxis)[0]
        prior_mean = self.prior_mean_func(self.full_axis)
        inv_var = 1/self.likelihood.variance
        inv_var_squared = tf.square(inv_var)
        log_var = tf.math.log(self.likelihood.variance)
        Krr = self.reduced_kernel(self.raxis, full_cov=True)
        Kqq = self.observed_kernel(self.qaxis, full_cov=True)
#         Krr += self.jitter * tf.eye(tf.shape(Krr)[-1], dtype=gpflow.config.default_float())
#         Kqq += self.jitter * tf.eye(tf.shape(Kqq)[-1], dtype=gpflow.config.default_float())
#         Lr = tf.linalg.cholesky(Krr)
#         Lq = tf.linalg.cholesky(Kqq)
        Krr_diag = tf.linalg.diag_part(Krr)
        Kqq_diag = tf.linalg.diag_part(Kqq)
        jitter_vec_r = tf.fill([tf.shape(Krr_diag)[-1]], self.jitter)
        jitter_vec_q = tf.fill([tf.shape(Kqq_diag)[-1]], self.jitter)
        Krr_full_rank = tf.linalg.set_diag(Krr, Krr_diag + jitter_vec_r)
        Kqq_full_rank = tf.linalg.set_diag(Kqq, Kqq_diag + jitter_vec_q)
        Lr = tf.linalg.cholesky(Krr_full_rank)
        Lq = tf.linalg.cholesky(Kqq_full_rank)

        
        B0 = kron(inv_var * tf.matmul(tf.matmul(Lr, self._WTW, transpose_a=True), Lr),
                  tf.matmul(Lq, Lq, transpose_a=True))
        B0_diag = tf.linalg.diag_part(B0)
        ones_rq = tf.fill([tf.shape(B0_diag)[-1]], tf.convert_to_tensor(1., dtype=gpflow.default_float()))
        B = tf.linalg.set_diag(B0, B0_diag + ones_rq)
        Lb = tf.linalg.cholesky(B)
        
        A = tf.linalg.triangular_solve(Lb, kron(tf.matmul(Lr, self._WTW, transpose_a=True),
                                                tf.transpose(Lq)),lower=True, adjoint=False)
        
        LtWty = tf.reshape(tf.transpose(tf.matmul(tf.matmul(Lq, self._YTW, transpose_a=True), Lr)),[-1,1])
        vecYtW = tf.reshape(tf.transpose(self._YTW),[-1,1])
        Wm = batched_vec_identity(self._WTW, tf.linalg.eye(Q, dtype=gpflow.config.default_float()),prior_mean)
        yb = tf.linalg.triangular_solve(Lb, LtWty, lower=True, adjoint=False)
        t1b1 = tf.reduce_sum(yb*yb)
        if self.zpm:
            t1a = 0.5*inv_var*(self._yTy)
            t1b = -0.5*inv_var_squared*(t1b1)
        else:
            Am = tf.matmul(A, prior_mean)
            t1a1 = 2.0*tf.matmul(prior_mean, vecYtW, transpose_a=True)
            t1a2 = tf.matmul(prior_mean, Wm, transpose_a=True)
            t1a = 0.5*inv_var*(self._yTy - t1a1 + t1a2)
            t1b2 = 2.0*tf.matmul(Am, yb, transpose_a=True)
            t1b3 = tf.reduce_sum(Am*Am)
            t1b = -0.5*inv_var_squared*(t1b1 - t1b2 + t1b3)
        t1 = t1a + t1b
        t2 = 0.5 * self._N * self._Q * log_var + tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Lb)))
        t3 = 0.5 * self._N * self._Q * self.log2pi
        loss = -t1 - t2# -t3 # t3 is constant wrt parameters, removing for numerical stability of gradients
        return tf.reduce_sum(loss)

    def predict_f(self, raxis: Optional[tf.Tensor] = None,
                        qaxis: Optional[tf.Tensor] = None,
                        full_cov: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""
        This method computes predictions at the requested axes if supplied or at the training axes if not
        """
        if raxis is None:
            raxis = self.raxis
        if qaxis is None:
            qaxis = self.qaxis
        full_axis = cartesian_prod(raxis, qaxis)
        R = tf.shape(raxis)[0]
        Q = tf.shape(qaxis)[0]
        prior_mean = self.prior_mean_func(full_axis)
        prior_mean_train = self.prior_mean_func(cartesian_prod(self.raxis, self.qaxis))
        inv_var = 1/self.likelihood.variance
        inv_var_squared = tf.square(inv_var)
        log_var = tf.math.log(self.likelihood.variance)
        # kernels needed for prediction to potentially new axes
        # read Ksfr as reduced_kernel(prediction_points, training_points)
        # read Ksfq as observed_kernel(prediction_points, training_points)
        Ksfr = self.reduced_kernel(raxis, self.raxis, full_cov=True) + \
                self._pred_jitter_kernel(raxis, self.raxis, full_cov=True)
        Kssr = self.reduced_kernel(raxis, full_cov=full_cov)  # diagonal
        Ksfq = self.observed_kernel(qaxis, self.qaxis, full_cov=True) + \
                self._pred_jitter_kernel(qaxis, self.qaxis, full_cov=True)
        Kssq = self.observed_kernel(qaxis, full_cov=full_cov)  # diagonal
        # kernel bits from training
        Krr = self.reduced_kernel(self.raxis, full_cov=True)
        Kqq = self.observed_kernel(self.qaxis, full_cov=True)
#         Krr += self.jitter * tf.eye(tf.shape(Krr)[-1], dtype=gpflow.config.default_float())
#         Kqq += self.jitter * tf.eye(tf.shape(Kqq)[-1], dtype=gpflow.config.default_float())
#         Lr = tf.linalg.cholesky(Krr)
#         Lq = tf.linalg.cholesky(Kqq)
        Krr_diag = tf.linalg.diag_part(Krr)
        Kqq_diag = tf.linalg.diag_part(Kqq)
        jitter_vec_r = tf.fill([tf.shape(Krr_diag)[-1]], self.jitter)
        jitter_vec_q = tf.fill([tf.shape(Kqq_diag)[-1]], self.jitter)
        Krr_full_rank = tf.linalg.set_diag(Krr, Krr_diag + jitter_vec_r)
        Kqq_full_rank = tf.linalg.set_diag(Kqq, Kqq_diag + jitter_vec_q)
        Lr = tf.linalg.cholesky(Krr_full_rank)
        Lq = tf.linalg.cholesky(Kqq_full_rank)

        B0 = kron(inv_var * tf.matmul(tf.matmul(Lr, self._WTW, transpose_a=True), Lr),
                  tf.matmul(Lq, Lq, transpose_a=True))
        B0_diag = tf.linalg.diag_part(B0)
        ones_rq = tf.fill([tf.shape(B0_diag)[-1]], tf.convert_to_tensor(1., dtype=gpflow.default_float()))
        B = tf.linalg.set_diag(B0, B0_diag + ones_rq)
        Lb = tf.linalg.cholesky(B)
        LtWty = tf.reshape(tf.transpose(tf.matmul(tf.matmul(Lq, self._YTW, transpose_a=True), Lr)),[-1,1])
        vecYtW = tf.reshape(tf.transpose(self._YTW),[-1,1])
        yb = tf.linalg.triangular_solve(Lb, LtWty, lower=True, adjoint=False)
        Lbinvt_yb = tf.linalg.triangular_solve(Lb, yb, lower=True, adjoint=True)        
        tm1 = inv_var * batched_vec_identity(Ksfr, Ksfq, vecYtW)
        tm2 = inv_var_squared * batched_vec_identity(tf.matmul(tf.matmul(Ksfr, self._WTW), Lr),
                                                    tf.matmul(Ksfq, Lq),
                                                    Lbinvt_yb)
        tm3 = prior_mean - inv_var * batched_vec_identity(tf.matmul(Ksfr, self._WTW), Ksfq, prior_mean_train)
        mb = tf.linalg.triangular_solve(Lb, batched_vec_identity(tf.matmul(Lr, self._WTW, transpose_a=True),
                                                          tf.transpose(Lq),
                                                          prior_mean_train), lower=True, adjoint=False)
        Lbinvt_mb = tf.linalg.triangular_solve(Lb, mb, lower=True, adjoint=True)
        tm4 = inv_var_squared * batched_vec_identity(tf.matmul(tf.matmul(Ksfr, self._WTW), Lr),
                                                    tf.matmul(Ksfq, Lq),
                                                    Lbinvt_mb)
        m = tm1 - tm2 + tm3 + tm4
        D1 = kron(tf.transpose(tf.matmul(tf.matmul(Ksfr, self._WTW), Lr)),
                  tf.transpose(tf.matmul(Ksfq, Lq)))
        D = tf.linalg.triangular_solve(Lb, D1, lower=True, adjoint=False)
        
        if not full_cov:
            v1a = tf.matmul(Ksfr, self._W, transpose_b=True)
            v1 = tf.reduce_sum(v1a * v1a, -1)
            v2 = tf.reduce_sum(Ksfq * Ksfq, -1)
            va = kron(Kssr[:,None], Kssq[:,None])
            vb = inv_var*kron(v1[:,None],v2[:,None])
            vc = inv_var_squared*(tf.reduce_sum(D*D,-2))[:,None]
            v = va - vb + vc
    #         v = -vb + vc
    #         v = (kron(Kssr[:,None], Kssq[:,None]) - \
    #              inv_var*kron(v1[:,None],v2[:,None]) + \
    #              inv_var_squared*(tf.reduce_sum(D*D,-2))[:,None])
            m = tf.transpose(tf.reshape(m,[R,Q]))
            v = tf.transpose(tf.reshape(v,[R,Q]))
        else:
            v1a = tf.matmul(Ksfr, self._W, transpose_b=True)
            v1 = v1a @ tf.transpose(v1a)
            v2 = Ksfq @ tf.transpose(Ksfq)
            va = kron(Kssr, Kssq)
            vb = inv_var*kron(v1,v2)
            vc = inv_var_squared*(tf.transpose(D) @ D)
            v = va - vb + vc
    #         v = -vb + vc
    #         v = (kron(Kssr[:,None], Kssq[:,None]) - \
    #              inv_var*kron(v1[:,None],v2[:,None]) + \
    #              inv_var_squared*(tf.reduce_sum(D*D,-2))[:,None])
            m = tf.transpose(tf.reshape(m,[R,Q]))
            v = tf.transpose(tf.reshape(v,[R*Q,R*Q]))
        if Q == 1:
            m = tf.squeeze(m)
            v = tf.squeeze(v)
        return m, v


