"""Diagnostics for absolute observation/action limits."""
import logging

import numpy as np

log = logging.getLogger(__name__)


def check_bounds(values, low, high, labels, message):
    values = np.asarray(values)
    low, high = np.broadcast_to(low, values.shape), np.broadcast_to(high, values.shape)
    outside = (values < low) | (values > high)
    if not np.any(outside):
        return
    details = []
    for index in zip(*np.nonzero(outside)):
        prefix = f'action_index={index[0]} ' if values.ndim == 2 else ''
        kind = 'target' if values.ndim == 2 else 'measured'
        details.append(f'{prefix}{labels[index[-1]]}: {kind}={values[index]:.6f}, '
                       f'low={low[index]:.6f}, high={high[index]:.6f}')
    error = message + ': ' + '; '.join(details)
    log.error('%s', error)
    raise ValueError(error)
