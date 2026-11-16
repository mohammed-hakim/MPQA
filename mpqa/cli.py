"""Shared command-line helpers: every script takes  --config FILE  and any number of  --set key=value."""
from .config import load_config


def add_config_args(parser):
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set epochs=2 --set data_dir=/my/data (repeatable)")
    return parser


def cfg_from_args(args):
    return load_config(args.config, args.set)
