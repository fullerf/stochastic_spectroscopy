# -*- coding: utf-8 -*-
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

import h5py
import gpflow
from pathlib import Path
import tensorflow as tf

__all__ = ['model_to_h5', 'h5_to_model', 'make_batch_iter', 'setup_logger']


# +
def none_in(t, F):
    flag = True
    if F is None:
        return flag
    for f in F:
        if f in t:
            flag = False
        else:
            continue
    return flag

def model_to_h5(fname, save_dir, model):
    param_dict = gpflow.utilities.parameter_dict(model)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    full_path = str((Path(save_dir) / Path(fname)).absolute())
    # clean the dictionary, converting all parameters to constrained values in numpy form
    # we lose information on the transform or prior. But if the model can be reproduced with
    # blank parameters, we just want to 
    with h5py.File(full_path, 'w') as fid:
        for k, v in param_dict.items():
            fid.create_dataset(k, data=v.numpy())
            
def h5_to_model(fname, save_dir, model, filter_keys=None):
    # a modle must be instantiated with the same structure as indicated by the keys in the h5py file
    full_path = Path(save_dir) / Path(fname)
    assert full_path.exists(), f"could not find {str(full_path.absolute())}"
    param_dict = {}
    with h5py.File(str(full_path.absolute()), 'r') as fid:
        keys = fid.keys()
        for key in keys:
            if none_in(key, filter_keys):
                param_dict[key] = tf.convert_to_tensor(fid[key])
        gpflow.utilities.multiple_assign(model, param_dict)        
    return model


# +
def make_batch_iter(data, batch_size, shuffle=True):
    if shuffle:
        data_minibatch = (
        tf.data.Dataset.from_tensor_slices(data)
                        .prefetch(tf.data.experimental.AUTOTUNE)
                        .repeat()
                        .shuffle(data[0].shape[0])
                        .batch(batch_size)
                        )
    else:
        data_minibatch = (
        tf.data.Dataset.from_tensor_slices(data)
                        .prefetch(tf.data.experimental.AUTOTUNE)
                        .repeat()
                        .batch(batch_size)
                        )
    data_minibatch_it = iter(data_minibatch)
    return data_minibatch_it

def setup_logger(model, name, loss_fn, log_dir='./logs'):
    log_dir = Path(log_dir) / Path(name)
    log_dir.mkdir(parents=True, exist_ok=True)
    fast_tasks = gpflow.monitor.MonitorTaskGroup([gpflow.monitor.ModelToTensorBoard(str(log_dir), model),
                                                 gpflow.monitor.ScalarToTensorBoard(str(log_dir), loss_fn, "loss")],
                                                period=1)
    monitor = gpflow.monitor.Monitor(fast_tasks)
    return monitor
# -


