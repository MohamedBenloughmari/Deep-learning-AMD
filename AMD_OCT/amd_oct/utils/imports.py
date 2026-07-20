import importlib.util
from typing import Optional


def is_available(package: str) -> bool:
    return importlib.util.find_spec(package) is not None


def require_extra(package: str, extra: str, hint: Optional[str] = None) -> None:
    """Raise an informative ImportError if ``package`` is not installed.

    Used by optional backbones (open_clip, mirage, natten) so that importing
    ``amd_oct`` never fails just because an optional extra is missing.
    """
    if is_available(package):
        return
    msg = (
        f"Optional dependency '{package}' is required for this feature. "
        f"Install it with:  pip install -e .[{extra}]"
    )
    if hint:
        msg += f"\n  {hint}"
    raise ImportError(msg)
