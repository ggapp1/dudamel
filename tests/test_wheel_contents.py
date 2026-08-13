"""`uv build` sanity: the shipped wheel must actually contain the scaffold
template, the deploy (launchd/systemd) templates, web templates/static
assets, and every migrations tree (core, the app template, and the suite
lane) -- a future pyproject.toml packaging regression (e.g. an errant
`packages =` edit) would otherwise ship a broken `dudamel new`/`dudamel run`
silently, since the test suite itself runs against the editable install, not
a built wheel."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def test_wheel_contains_scaffold_template_and_web_assets(tmp_path: Path) -> None:
    subprocess.run(
        ["uv", "build", "--wheel", "-o", str(tmp_path)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, found {wheels}"
    with zipfile.ZipFile(wheels[0]) as zf:
        names = set(zf.namelist())

    def has_prefix(prefix: str) -> bool:
        return any(n.startswith(prefix) for n in names)

    assert has_prefix("dudamel/scaffold_template/"), names
    assert "dudamel/scaffold_template/apps/__init__.py" in names
    assert "dudamel/scaffold_template/pyproject.toml" in names
    assert has_prefix("dudamel/web/templates/"), names
    assert "dudamel/web/static/htmx.min.js" in names
    assert has_prefix("dudamel/migrations/versions/"), names
    assert has_prefix("dudamel/migrations_app_template/"), names
    assert has_prefix("dudamel/migrations_suite_lane/"), names
    assert "dudamel/deploy_templates/dudamel.plist" in names
    assert "dudamel/deploy_templates/dudamel.service" in names
