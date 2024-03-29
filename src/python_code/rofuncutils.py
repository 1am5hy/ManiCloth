import os
import pathlib
import shutil
import platform
import socket
from contextlib import contextmanager

import rofunc as rf

def create_dir(path, local_verbose=False):
    """
    Create the directory if it does not exist.

    Example::

        >>> import rofunc as rf
        >>> rf.oslab.create_dir('/home/ubuntu/Github/Rofunc/examples/data/felt/trial_1', local_verbose=True)

    :param path: the path of the directory
    :param local_verbose: if True, print the message
    :return:
    """
    if not pathlib.Path(path).exists():
        if local_verbose:
            rf.logger.beauty_print('{} not exist, created.'.format(path), type='info')
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)\

def get_so_reuseport():
    """
    Get the port with ``SO_REUSEPORT`` flag set.

    :return: port number or None
    """
    try:
        return socket.SO_REUSEPORT
    except AttributeError:
        if platform.system() == "Linux":
            major, minor, *_ = platform.release().split(".")
            if (int(major), int(minor)) > (3, 9):
                # The interpreter must have been compiled on Linux <3.9.
                return 15
    return None


def reserve_sock_addr():
    """
    Reserve an available TCP port to listen on.

    The reservation is done by binding a TCP socket to port 0 with
    ``SO_REUSEPORT`` flag set (requires Linux >=3.9). The socket is
    then kept open until the generator is closed.

    To reduce probability of 'hijacking' port, socket should stay open
    and should be closed _just before_ starting of ``tf.train.Server``
    """
    so_reuseport = get_so_reuseport()
    if so_reuseport is None:
        raise RuntimeError("SO_REUSEPORT is not supported by the operating system") from None

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, so_reuseport, 1)
        sock.bind(("", 0))
        _ipaddr, port = sock.getsockname()
        yield (socket.getfqdn(), port)