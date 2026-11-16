import pytest

from mpqa.config import load_config


def test_defaults_load():
    cfg = load_config()
    for key in ["epochs", "batch_size", "lr", "lam", "eps", "act", "head_drop", "arch", "seed", "out_dir"]:
        assert key in cfg
    assert cfg.lam == 4.0 and cfg.eps == 0.1 and cfg.act == "gelu" and cfg.head_drop == 0.0
    assert cfg.lr == pytest.approx(5e-4) and cfg.wd == pytest.approx(1e-4)       # YAML floats parsed as floats


def test_overrides_are_typed():
    cfg = load_config(overrides=["epochs=3", "lr=1e-4", "amp=false", "act=relu", "resume_from=null", "head_drop=0.5"])
    assert cfg.epochs == 3 and isinstance(cfg.epochs, int)
    assert cfg.lr == pytest.approx(1e-4) and isinstance(cfg.lr, float)
    assert cfg.amp is False and cfg.act == "relu" and cfg.resume_from is None and cfg.head_drop == 0.5


def test_unknown_key_and_malformed_override_fail():
    with pytest.raises(KeyError):
        load_config(overrides=["epohcs=3"])
    with pytest.raises(ValueError):
        load_config(overrides=["epochs"])


def test_attribute_access():
    cfg = load_config()
    cfg.epochs = 7
    assert cfg["epochs"] == 7
    with pytest.raises(AttributeError):
        cfg.not_a_key
