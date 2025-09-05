"""
SatIR configuration loader.

Loads configuration from satir.toml, with environment variable overrides.
Environment variables always take precedence over config file values.

Usage:
    from smt_core.config import get_config
    cfg = get_config()
    print(cfg.api.endpoint)
    print(cfg.paths.build_dir)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # fallback
    except ImportError:
        tomllib = None


@dataclass
class ApiConfig:
    endpoint: str = ""
    endpoint_gpt5: str = ""
    api_key: str = ""
    model: str = "gpt-4.1"


@dataclass
class ServicesConfig:
    snowstorm_url: str = "http://localhost:8080"
    snowstorm_branch: str = "MAIN"
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "snomed_vectors"


@dataclass
class PathsConfig:
    build_dir: str = "./build"
    data_dir: str = "./dataset/clinical_trial"
    patient_data_dir: str = "./dataset/patient_notes"

    @property
    def build_path(self) -> Path:
        return Path(self.build_dir).resolve()

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).resolve()

    @property
    def patient_data_path(self) -> Path:
        return Path(self.patient_data_dir).resolve()


@dataclass
class RetrievalConfig:
    scope: str = "any"
    important_mode: str = "all"
    alt_mode: str = "act"
    parallel: int = 8
    enable_prevention: bool = True


@dataclass
class CompilerConfig:
    max_retries: int = 3
    enable_profile: bool = True


@dataclass
class KeysConfig:
    umls_api_key: str = ""


@dataclass
class SatIRConfig:
    api: ApiConfig = field(default_factory=ApiConfig)
    services: ServicesConfig = field(default_factory=ServicesConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    compiler: CompilerConfig = field(default_factory=CompilerConfig)
    keys: KeysConfig = field(default_factory=KeysConfig)


# ── Environment variable mapping ─────────────────────────────────────────

_ENV_MAP = {
    # (section, key) → env var name
    ("api", "endpoint"): "OPENAI_ENDPOINT",
    ("api", "endpoint_gpt5"): "OPENAI_ENDPOINT_GPT5",
    ("api", "api_key"): ["OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"],
    ("api", "model"): "OPENAI_MODEL",
    ("services", "snowstorm_url"): "SNOWSTORM_BASE",
    ("services", "snowstorm_branch"): "SNOWSTORM_BRANCH",
    ("services", "elasticsearch_url"): "ELASTICSEARCH_URL",
    ("paths", "build_dir"): "SATIR_BUILD",
    ("paths", "data_dir"): "TRIAL_DATA",
    ("paths", "patient_data_dir"): "PN_DATA",
    ("compiler", "max_retries"): "MAX_RETRIES",
    ("keys", "umls_api_key"): "UMLS_API_KEY",
}


def _resolve_env(env_names) -> Optional[str]:
    """Resolve value from one or more env var names."""
    if isinstance(env_names, str):
        env_names = [env_names]
    for name in env_names:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return None


def _load_toml(path: Path) -> dict:
    """Load TOML file, return empty dict if not found or no parser."""
    if not path.exists():
        return {}
    if tomllib is None:
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def _find_config_file() -> Path:
    """Search for satir.toml in current dir and parent dirs."""
    cwd = Path.cwd()
    for p in [cwd, cwd.parent, cwd.parent.parent]:
        candidate = p / "satir.toml"
        if candidate.exists():
            return candidate
    # Also check package root
    pkg_root = Path(__file__).resolve().parent.parent
    candidate = pkg_root / "satir.toml"
    if candidate.exists():
        return candidate
    return cwd / "satir.toml"  # default location even if doesn't exist


def get_config(config_path: Optional[str] = None) -> SatIRConfig:
    """
    Load SatIR configuration.

    Priority: env vars > satir.toml > defaults

    Args:
        config_path: Explicit path to satir.toml. If None, searches cwd and parents.

    Returns:
        SatIRConfig dataclass with all resolved values.
    """
    path = Path(config_path) if config_path else _find_config_file()
    raw = _load_toml(path)

    cfg = SatIRConfig()

    # Load from TOML
    section_map = {
        "api": cfg.api,
        "services": cfg.services,
        "paths": cfg.paths,
        "retrieval": cfg.retrieval,
        "compiler": cfg.compiler,
        "keys": cfg.keys,
    }

    for section_name, section_obj in section_map.items():
        toml_section = raw.get(section_name, {})
        for key in vars(section_obj):
            if key.startswith("_"):
                continue
            if key in toml_section:
                val = toml_section[key]
                current_type = type(getattr(section_obj, key))
                if current_type == int:
                    val = int(val)
                elif current_type == bool:
                    val = bool(val)
                setattr(section_obj, key, val)

    # Override from environment variables
    for (section_name, key), env_names in _ENV_MAP.items():
        env_val = _resolve_env(env_names)
        if env_val:
            section_obj = section_map[section_name]
            current_type = type(getattr(section_obj, key))
            if current_type == int:
                env_val = int(env_val)
            elif current_type == bool:
                env_val = env_val.lower() in ("1", "true", "yes")
            setattr(section_obj, key, env_val)

    return cfg


# ── Singleton for convenience ────────────────────────────────────────────

_CONFIG: Optional[SatIRConfig] = None


def config() -> SatIRConfig:
    """Get or create the global config singleton."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = get_config()
    return _CONFIG
