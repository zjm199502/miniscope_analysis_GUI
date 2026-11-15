import json
import zipfile

from miniscope_gui import updater


def test_version_compare():
    assert updater.is_newer("0.2.0", "0.1.9")
    assert updater.is_newer("v1.0", "0.9.9")
    assert not updater.is_newer("0.1.0", "0.1.0")
    assert not updater.is_newer("0.1.0", "0.2.0")
    assert updater.parse_version("v0.3") == (0, 3, 0)


class _Resp:
    def __init__(self, data: bytes):
        self.data, self.pos = data, 0
        self.headers = {"Content-Length": str(len(data))}

    def read(self, n=-1):
        if n < 0:
            n = len(self.data)
        out = self.data[self.pos:self.pos + n]
        self.pos += len(out)
        return out

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_check_latest_and_prepare(monkeypatch, tmp_path):
    name = updater.asset_name() or "MiniscopeAnalysis-Windows.zip"
    monkeypatch.setattr(updater, "asset_name", lambda: name)
    rel_json = {"tag_name": "v99.0.0", "body": "notes", "html_url": "https://x/r",
                "assets": [{"name": name, "browser_download_url": "https://x/a.zip", "size": 10}]}
    zbuf = tmp_path / "src.zip"
    with zipfile.ZipFile(zbuf, "w") as z:
        z.writestr("MiniscopeAnalysis/MiniscopeAnalysis.exe", "new")

    def fake_open(req, timeout=None, context=None):
        url = req.full_url
        if "api.github.com" in url:
            return _Resp(json.dumps(rel_json).encode())
        return _Resp(zbuf.read_bytes())
    monkeypatch.setattr(updater.urllib.request, "urlopen", fake_open)
    rel = updater.check_latest()
    assert rel.version == "99.0.0" and rel.asset_url == "https://x/a.zip"

    app = tmp_path / "apps" / "MiniscopeAnalysis"
    app.mkdir(parents=True)
    monkeypatch.setattr(updater, "install_root", lambda: app)
    monkeypatch.setattr(updater, "_ssl_context", lambda: None)
    monkeypatch.setattr(updater.sys, "platform", "win32")
    seen = []
    new_dir = updater.prepare_update(rel, seen.append)
    assert (new_dir / "MiniscopeAnalysis" / "MiniscopeAnalysis.exe").read_text() == "new"
    assert seen[-1] == 1.0
    assert new_dir.parent.parent == app.parent
    updater.cleanup_stale(app)
    assert not new_dir.exists()


def test_no_update_when_same(monkeypatch):
    def fake_open(req, timeout=None, context=None):
        return _Resp(json.dumps({"tag_name": "v" + updater.__version__, "assets": []}).encode())
    monkeypatch.setattr(updater.urllib.request, "urlopen", fake_open)
    assert updater.check_latest() is None
