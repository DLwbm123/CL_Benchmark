"""Neutral process entry: method/configuration is read from RUN_CONFIG, not argv.

Run from this directory as `python run.py`. `--validate-only` checks parser
compatibility without constructing a dataset/model or starting training.
"""
import argparse
import importlib
import json
import os


def read_configuration(path):
    from models import get_all_models
    with open(path) as source:
        config = json.load(source)
    if not isinstance(config, dict) or config.get('model') not in get_all_models():
        raise ValueError('Configuration must select a registered model.')
    # A long-lived scheduler may hold an older submitted config. A versioned,
    # per-run protocol file updates only explicitly permitted optimizer fields.
    # Identity, data, seed, epoch budget and train/test phase remain unchanged.
    root, name = os.environ.get('SESSION_ROOT'), config.get('name')
    if root and isinstance(name, str) and name == os.path.basename(name):
        protocol_path = os.path.join(root, 'protocols', name + '.json')
        if os.path.isfile(protocol_path):
            with open(protocol_path) as source:
                protocol = json.load(source)
            overrides = protocol['overrides']
            allowed = {
                's3r': {'lr', 'lr_decay_rate', 's3r_optimizer', 's3r_weight_decay', 's3r_epoch_gamma'},
                'ldka_organ': {'ldka_organ_grad_clip'},
                'ldka': {'ldka_grad_clip'},
                'pcdd': {'pcdd_grad_clip'},
                'rebac': {'rebac_grad_clip'},
                'cit': {'lr','cit_optimizer','cit_weight_decay','cit_head_lr_mult'},
                'cit_organ': {'lr','cit_optimizer','cit_weight_decay','cit_head_lr_mult'},
            }
            if (config['model'] not in allowed or protocol['model'] != config['model']
                    or set(overrides) - allowed[config['model']]):
                raise ValueError('Protocol override cannot alter run identity, budget or phase.')
            config.update(overrides)
            print('Resolved protocol:', protocol['protocol_id'], flush=True)
    parser = importlib.import_module('models.' + config['model']).get_parser()
    argv = []
    for key, value in config.items():
        option = '--' + key
        action = parser._option_string_actions.get(option)
        if action is None:
            raise ValueError('Unknown configuration option: ' + key)
        if value is None:
            continue
        if action.nargs == 0:
            if not isinstance(value, bool):
                raise ValueError('Flag must have a boolean value: ' + key)
            if value == action.const:
                argv.append(option)
            elif value != action.default:
                opposite = next((a for a in parser._actions
                                 if a.dest == action.dest and a.nargs == 0
                                 and a.const == value), None)
                if opposite is None:
                    raise ValueError('Cannot express requested flag value: ' + key)
                argv.append(opposite.option_strings[0])
        else:
            argv.append(option)
            argv.extend(str(v) for v in value) if isinstance(value, list) else argv.append(str(value))
    parser.parse_args(argv)
    return argv


def main():
    parser = argparse.ArgumentParser(description='Configuration-driven local entry')
    parser.add_argument('--validate-only', action='store_true')
    options = parser.parse_args()
    path = os.environ.get('RUN_CONFIG')
    if not path:
        parser.error('Set RUN_CONFIG to a local JSON configuration path.')
    argv = read_configuration(path)
    if options.validate_only:
        print('Configuration syntax accepted; no dataset/model constructed and no training started.')
        return
    from utils.main import main as framework_main
    print('Execution variant:', os.environ.get('CL_EXECUTION_VARIANT', 'original'),
          '| H5 RAM cache:', os.environ.get('CL_H5_RAM_CACHE', '0'),
          '| cache registry MiB:', os.environ.get('CL_H5_RAM_CACHE_MB', '4096'), flush=True)
    framework_main(argv)


if __name__ == '__main__':
    main()
