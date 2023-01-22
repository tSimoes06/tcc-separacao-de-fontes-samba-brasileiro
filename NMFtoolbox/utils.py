"""
    Name: utils
    Date: Jun 2019
    Programmer: Yiğitcan Özer

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    If you use the 'NMF toolbox' please refer to:
    [1] Patricio López-Serrano, Christian Dittmar, Yiğitcan Özer, and Meinard
        Müller
        NMF Toolbox: Music Processing Applications of Nonnegative Matrix
        Factorization
        In Proceedings of the International Conference on Digital Audio Effects
        (DAFx), 2019.

    License:
    This file is part of 'NMF toolbox'.
    https://www.audiolabs-erlangen.de/resources/MIR/NMFtoolbox/
    'NMF toolbox' is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    the Free Software Foundation, either version 3 of the License, or (at
    your option) any later version.

    'NMF toolbox' is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
    Public License for more details.

    You should have received a copy of the GNU General Public License along
    with 'NMF toolbox'. If not, see http://www.gnu.org/licenses/.
"""

import numpy as np
from scipy.ndimage.filters import convolve
import scipy.io
import os

EPS = 2.0 ** -52
MAX_WAV_VALUE = 2.0 ** 15
PATH_TO_MATRICES = 'matrices/'


def make_monaural(audio):
    """Converts multichannel audio to mono-channel output

    Parameters
    ----------
    audio: array-like
        Audio input in numpy format

    Returns
    -------
    audio: array-like
        Monochannel audio

    """
    audio = np.mean(audio, axis=1) if len(audio.shape) == 2 else audio

    return audio


def load_matlab_dict(filepath, field):
    """Loads .mat file from the directory

    Parameters
    ----------
    filepath: str
        Path to the .mat file

    field: str
        Name of the MATLAB matrix, which is the key of the dictionary

    Returns
    -------
    mat[field]: array-like
        MATLAB matrix in python
    """
    mat = scipy.io.loadmat(filepath)
    return mat[field]


def pcmInt16ToFloat32Numpy(audio):
    """Converts the data type of the input from int16 to float32

    Parameters
    ----------
    audio: array-like
        Numpy array in int16 type

    Returns
    -------
    res: array-like
        Numpy array in float32 type

    """
    res = np.array(audio, dtype=np.float32) / MAX_WAV_VALUE
    res[res > 1] = 1
    res[res < -1] = -1

    return res


def conv2(x, y, mode='same'):
    """Emulate the function conv2 from Mathworks.

    Usage:

    z = conv2(x,y,mode='same')

    """
    # We must provide an offset for each non-singleton dimension to reproduce the results of Matlab's conv2.
    # A simple implementation supporting the 'same' option, only, could be made like below
    # source: https://stackoverflow.com/questions/3731093/is-there-a-python-equivalent-of-matlabs-conv2-function

    if not mode == 'same':
        raise NotImplementedError("Mode not supported")

    # Add singleton dimensions
    if len(x.shape) < len(y.shape):
        dim = x.shape
        for i in range(len(x.shape), len(y.shape)):
            dim = (1,) + dim
        x = x.reshape(dim)

    elif len(y.shape) < len(x.shape):
        dim = y.shape
        for i in range(len(y.shape), len(x.shape)):
            dim = (1,) + dim
        y = y.reshape(dim)

    origin = ()

    # Apparently, the origin must be set in a special way to reproduce
    # the results of scipy.signal.convolve and Matlab
    for i in range(len(x.shape)):
        if ((x.shape[i] - y.shape[i]) % 2 == 0 and
                x.shape[i] > 1 and
                y.shape[i] > 1):
            origin = origin + (-1,)
        else:
            origin = origin + (0,)

    z = convolve(x, y, mode='constant', origin=origin)

    return z


def run_unit_test(res_python, mat_matlab, decimal_precision=5):
    """Runs the unit test for one of the functions in unit_tests folder

    Parameters
    ----------
    res_python: array-like
        Python result

    mat_matlab: array-like
        MATLAB matrix

    decimal_precision: int
        Desired precision, default is 5.
    """
    # If res_python is a list, convert it into numpy array format
    if isinstance(res_python, list):
        arr_python = np.concatenate(res_python, axis=1)
    else:
        arr_python = res_python

    np.testing.assert_almost_equal(arr_python, mat_matlab, decimal=decimal_precision, err_msg='', verbose=True)

    print('Test successfully passed. Precision: {} significant digits'.format(decimal_precision))


def get_matlab_matrices(function_name):
    """Loads the matrices generated by MATLAB for unit tests

    Parameters
    ----------
    function_name: str
        Function name, e.g. NMFD

    Returns
    -------
    matrix_dict: dict
        MATLAB matrix in dict format
    """
    base_dir = os.path.join(PATH_TO_MATRICES, function_name)
    filename_list = os.listdir(base_dir)

    matrix_dict = dict()

    for filename in filename_list:
        filepath = os.path.join(base_dir, filename)
        matrix_id = filename.split('.')[0]
        matlab_matrix = load_matlab_dict(filepath, matrix_id)
        matrix_dict[matrix_id] = matlab_matrix

    return matrix_dict


def run_matlab_script(function_name, path_to_matlab_bin):
    """Runs the corresponding MATLAB script for unit testing

    Parameters
    ----------
    function_name: str
        Function name, e.g. NMFD

    path_to_matlab_bin: str
        Path to the binary file of MATLAB
    """
    os.system('cat test_{}.m | {} - nodesktop - nosplash'.format(function_name, path_to_matlab_bin))