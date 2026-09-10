"""Load the standalone module without starting CyberHub or touching user data."""
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_module(root=ROOT, name="auto_tagger_test"):
    class Module:
        def __init__(self, hub):
            self.hub = hub

        def setting(self, key, default=None):
            return self.settings_schema.get(key, {}).get("default", default)

    core = types.ModuleType("core")
    core.Module = Module
    server = types.ModuleType("core.server")
    server.build_shell = lambda *args, **kwargs: ""
    path = Path(root) / "modules" / "gallery_auto_tagger" / "__init__.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with patch.dict(sys.modules, {"core": core, "core.server": server}):
        spec.loader.exec_module(module)
    tagger = importlib.import_module(name + ".tagger")
    return module, tagger
