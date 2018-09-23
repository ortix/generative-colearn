# Module imports
import glob as _glob
import importlib as _importlib
import json as _json
import os as _os
from datetime import datetime

import numpy as _np
import pandas as _pd
from munch import munchify as _munchify

from settings import settings as cfg


def create_run_dir(path, model, clean):
    dirname = "{}_{}_{}".format(
        model,
        clean,
        datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    )
    fullpath = _os.path.join(path, "run", dirname)
    try:
        _os.makedirs(fullpath)
    except FileExistsError:
        print("{} already exists. Continueing ".format(fullpath))
        pass
    return fullpath


def get_model(model_name):
    """
    Returns the model class. Not an instantiation!
    """
    m = _importlib.import_module("models.{}".format(model_name))
    model_class = getattr(m, model_name.upper())
    return model_class


def get_trainer(trainer):
    """
    Returns the requested trainer class. Not an instantiation!
    the trainer argument is the model name (cvae, cgan, cwgan)
    """
    m = _importlib.import_module("trainers.{}_trainer".format(trainer))
    modelname = trainer.upper()
    trainer = getattr(m, "{}Trainer".format(modelname))
    return trainer


def load_latest(path, ext=["*.*"]):
    """
    Loads the latest file in a path
    path: Absolute directory to search in
    ext: Array containing extensions. Use *.* for wildcard
    """
    list_of_files = []
    path = _os.path.join(path, "")  # Ensure we have trailing slash
    for e in ext:
        list_of_files.extend(_glob.glob(path + e))

    if not list_of_files:
        return False

    return max(list_of_files, key=_os.path.getctime)


def get_settings_path(file_name):
    current_dir = _os.path.dirname(_os.path.realpath(__file__))
    return _os.path.join(current_dir, file_name)


def save_settings(settings, settings_path=None):

    if not settings_path:
        # Settings are probably stored one folder up
        settings_path = _os.path.dirname(_os.path.dirname(_os.path.realpath(__file__)))

    filename = _os.path.join(settings_path, 'settings.json')
    with open(filename, 'w') as outfile:
        _json.dump(settings, outfile, indent=4)
    return None

    # def load_settings(settings_path):
    #     # Mutate paths to be system absolute paths
    #     current_dir = _os.path.dirname(_os.path.realpath(__file__))
    #     settings_path_full = _os.path.join(current_dir, settings_path)

    #     # Import settings as dict and munchify to allow dot access
    #     with open(settings_path_full) as f:
    #         settings = _munchify(_json.load(f))

    #     for d in settings.paths:
    #         settings.paths[d] = _os.path.join(current_dir, settings.paths[d])

    #     return settings
