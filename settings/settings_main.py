import collections
import json
import os

from jsonmerge import merge
from munch import munchify

__instance = None
__current_dir = os.path.dirname(os.path.realpath(__file__))
__main_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def settings(custom=None):
    global __instance
    if __instance != None:
        return __instance

    # Import settings as dict and munchify to allow dot access
    with open(get_settings_path("base")) as f:
        settings = json.load(f)

    # Merge custom settings
    if custom is not None:
        try:
            with open(get_settings_path(custom)) as f:
                custom_settings = json.load(f)
            settings = merge(settings, custom_settings)
            print("Found {} settings. Merging with base".format(custom))
        except Exception as e:
            print(repr(e))
            print("Can't find {} settings. Opening base settings".format(custom))

    settings = munchify(settings)
    # Ensure full system path is inside the paths directive
    for d in settings.paths:
        if __main_dir in settings.paths[d]:
            continue
        settings.paths[d] = os.path.join(__main_dir, settings.paths[d])
    __instance = settings
    return __instance


def get_settings_path(name):
    filename = "{}_settings.json".format(name)
    return os.path.join(__current_dir, filename)
