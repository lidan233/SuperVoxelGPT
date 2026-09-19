"""Let a run be described by a file instead of a twenty-flag command line.

Purpose
    A training run is defined by which weights it produces, and that depends on every
    hyperparameter. Recording them in a file makes a run quotable — "trained with configs/x.json"
    is checkable, "trained with the usual flags" is not.

Input
    A JSON object whose keys are the long flag names, with or without leading dashes and in either
    spelling (`--max-epochs`, `max_epochs`).

Output
    Values installed as argparse defaults, so an explicit command-line flag still wins.
"""
import argparse
import json
import os
from typing import Any, Dict


def apply_config(parser: argparse.ArgumentParser, config_path: str) -> Dict[str, Any]:
    """Install a JSON config as the parser's defaults. Returns what was applied."""
    with open(config_path) as f:
        cfg = json.load(f)

    known = {a.dest for a in parser._actions}
    applied, unknown = {}, []
    for key, value in cfg.items():
        if key.startswith("_"):          # allow "_comment"-style annotations
            continue
        dest = key.lstrip("-").replace("-", "_")
        if dest in known:
            applied[dest] = value
        else:
            unknown.append(key)

    if unknown:
        raise SystemExit(
            f"{config_path}: unknown option(s) {unknown}. A misspelled key would otherwise be "
            f"ignored and the run would silently use defaults instead of what this file says.")

    parser.set_defaults(**applied)
    # A default does not satisfy argparse's `required`: it checks whether the flag appeared on the
    # command line. Without this, every required option a config supplies would still have to be
    # repeated as a flag, which defeats the point of having the file.
    for action in parser._actions:
        if action.required and action.dest in applied:
            action.required = False
    return applied


def add_config_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None,
                        help="JSON file of options; command-line flags still take precedence")


def parse_with_config(parser: argparse.ArgumentParser, argv=None) -> argparse.Namespace:
    """Two-pass parse: read --config, install it as defaults, then parse for real."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    if known.config:
        applied = apply_config(parser, known.config)
        print(f"[config] {known.config}: {len(applied)} options", flush=True)
    return parser.parse_args(argv)


def constructor_args(cfg: Dict[str, Any], target, what: str = "") -> Dict[str, Any]:
    """Keep the keys `target` can actually accept, and say which ones were left out.

    A config section records what a run was, which is not the same list as what a constructor
    takes: `train_captions` names the split the released autoencoder was trained on, and no
    dataset class has a parameter for it. Expanding the section wholesale into a class that
    declares its parameters explicitly turns every such record into a TypeError, so the entry
    point would have to be edited each time a config gains a field.

    A target that declares `**kwargs` is left alone — it has already chosen to accept anything.
    Otherwise the dropped keys are printed rather than silently discarded, because the same
    filtering would quietly swallow a misspelled parameter and run with its default instead.
    """
    import inspect
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return dict(cfg)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(cfg)

    kept = {k: v for k, v in cfg.items() if k in params}
    dropped = sorted(set(cfg) - set(kept))
    if dropped:
        name = what or getattr(target, "__name__", str(target))
        print(f"[config] {name}: ignoring {', '.join(dropped)} "
              f"({'is' if len(dropped) == 1 else 'are'} not a parameter of it)", flush=True)
    return kept


def flatten_args(section: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a config section's nested "args" into its top level.

    The two VAE configs came from different upstream trees and disagree: one nests constructor
    arguments under "args", the other lists them flat. Accepting both here means neither entry
    point has to know which lineage its config came from, and hand-written configs work either
    way. Top-level keys win, so a CLI override written into the top level is not shadowed.
    """
    section = dict(section)
    merged = dict(section.pop("args", {}) or {})
    merged.update(section)
    return merged
