# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import importlib

def get_all_models():
    # Discover without importing every optional method (e.g. GEM's quadprog).
    return sorted(model[:-3] for model in os.listdir(os.path.dirname(__file__))
                  if model.endswith('.py') and not model.startswith('_'))

names = {}

def get_model(args, backbone, loss, transform):
    if args.model not in get_all_models():
        raise ValueError('Unknown model: ' + args.model)
    if args.model not in names:
        mod = importlib.import_module('models.' + args.model)
        class_name = {x.lower(): x for x in dir(mod)}[args.model.replace('_', '')]
        names[args.model] = getattr(mod, class_name)
    return names[args.model](backbone, loss, args, transform)
