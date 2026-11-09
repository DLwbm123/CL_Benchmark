"""Load selected torch-only definitions from private, pinned author checkouts.
No upstream code is copied into this adapter or its public distribution.
The training/data/inference framework is replaced explicitly by the pilot runner.
"""
import ast
import copy
from pathlib import Path
from typing import Optional, Union, Tuple, List
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class InitWeightsHe:
    def __init__(self, slope=1e-2):
        self.slope = slope

    def __call__(self, module):
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, a=self.slope)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


def namespace():
    return dict(torch=torch, nn=nn, F=F, np=np, Optional=Optional,
                Union=Union, Tuple=Tuple, List=List, deepcopy=copy.deepcopy,
                InitWeights_He=InitWeightsHe, softmax_helper=lambda x: x.softmax(1),
                SegmentationNetwork=nn.Module,
                tqdm=lambda iterable, **kwargs: iterable)


def definitions(path, env, names=None, constants=False):
    tree = ast.parse(Path(path).read_text())
    body = [node for node in tree.body
            if (isinstance(node, (ast.ClassDef, ast.FunctionDef)) and
                (names is None or node.name in names)) or
               (constants and isinstance(node, ast.Assign))]
    if names is not None:
        found = {n.name for n in body if hasattr(n, 'name')}
        if found != set(names):
            raise RuntimeError(f'missing source definitions: {set(names) - found}')
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), env)


def focus_components(root):
    root = Path(root)
    env = namespace()
    definitions(root/'nnunet_ext/network_architecture/GenericUNet_.py', env)
    # Original constructor, CCD, freezing, forward, and first-feature replay.
    definitions(root/'nnunet_ext/network_architecture/generic_UNet_sparse.py', env,
                ['Generic_UNet_sparse'])
    definitions(root/'nnunet_ext/utilities/helpful_functions.py', env,
                ['GINGroupConv2D', 'GradlessGCReplayNonlinBlock2D'])
    return env['Generic_UNet_sparse'], env['GINGroupConv2D']


def cauaug_components(root):
    root = Path(root)
    env = namespace()
    definitions(root/'mp/models/segmentation/segmentation_model.py', env,
                ['SegmentationModel'])
    definitions(root/'mp/models/segmentation/model_utils.py', env, constants=True)
    definitions(root/'mp/models/segmentation/unet_fepegar.py', env,
                ['UNet', 'UNet2D'])
    tree = ast.parse((root/'mp/agents/cauaug_agent.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CauAugAgent')
    keep = {'cross_kernel_network', 'fourier_transformer_generator',
            'domain_knowledge_archive', 'multi_class_cross_entropy_no_softmax'}
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in keep]
    assert {n.name for n in cls.body} == keep
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 str(root/'mp/agents/cauaug_agent.py'), 'exec'), env)
    return env['UNet2D'], env['CauAugAgent']
