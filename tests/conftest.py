"""pytest 전역 설정 — JAVA_HOME 자동 탐지 (NixOS 지원)."""
import os
import subprocess


def _find_jvm_home() -> str | None:
    """JAVA_HOME을 찾는다. 이미 설정돼 있으면 그대로 반환."""
    if os.environ.get("JAVA_HOME"):
        return os.environ["JAVA_HOME"]

    # nix develop 셸에서 JAVA_HOME 읽기 (NixOS 환경)
    try:
        result = subprocess.run(
            ["nix", "develop", "--command", "bash", "-c", "echo $JAVA_HOME"],
            capture_output=True, text=True, timeout=30,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        java_home = result.stdout.strip()
        if java_home and os.path.isdir(java_home):
            return java_home
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


_jvm_home = _find_jvm_home()
if _jvm_home:
    os.environ["JAVA_HOME"] = _jvm_home
