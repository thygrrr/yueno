import sys

import pytest

from yueno import audiocpp
from yueno.config import Paths


@pytest.mark.parametrize("options,expected", [({}, "auto"), ({"cuda_arch": "75-real"}, "75-real")])
def test_build_uses_detected_or_explicit_gpu(tmp_path, monkeypatch, options, expected):
    monkeypatch.setenv("YUENO_HOME", str(tmp_path))
    monkeypatch.delenv("YUENO_AUDIOCPP_EXE", raising=False)
    paths = Paths.resolve()
    script = paths.audiocpp_src / "scripts" / "build_windows.ps1"
    script.parent.mkdir(parents=True)
    script.touch()
    paths.exe.parent.mkdir(parents=True)
    paths.exe.touch()
    monkeypatch.setattr(audiocpp, "find_cuda_toolkit", lambda: tmp_path / "v13.5")
    monkeypatch.setattr(audiocpp.shutil, "which", lambda tool: tool if tool in ("git", "powershell") else None)
    monkeypatch.setattr(audiocpp, "checkout", lambda *args, **kwargs: None)
    calls = []
    monkeypatch.setattr(audiocpp, "_run", lambda cmd, **kwargs: calls.append(cmd))

    assert audiocpp.build(paths, **options) == paths.exe
    assert calls[0][calls[0].index("-CudaArchitectures") + 1] == expected


def test_streaming_reports_child_pid(tmp_path):
    pids, lines = [], []
    code, output = audiocpp.run_streaming([sys.executable, "-c", "print('ready')"], cwd=tmp_path,
                                         on_line=lines.append, on_start=pids.append)
    assert code == 0 and output == "ready" and lines == ["ready"]
    assert len(pids) == 1 and pids[0] > 0
