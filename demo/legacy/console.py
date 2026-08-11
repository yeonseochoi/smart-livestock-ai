"""콘솔 출력 안전장치.

2026-08-07 추가. Claude Code 검증에서 발견된 문제 대응.

    한국어 Windows 의 기본 콘솔 코드페이지는 cp949 다.
    이모지(✅❌⚠️)나 em dash(—) 를 그냥 print 하면
    UnicodeEncodeError 로 프로그램이 죽는다.

    실제 증상:
        python main.py --mock
        → UnicodeEncodeError: 'cp949' codec can't encode character '✅'

의존성이 없는 별도 모듈로 둔 이유
    recommend.py 에 두면 residence.py 가 이걸 쓰려 할 때
    residence → recommend → residence 순환 임포트가 생긴다.
"""

from __future__ import annotations

import sys


def use_utf8_stdout() -> None:
    """stdout/stderr 을 UTF-8 로 강제한다. 실패해도 조용히 넘어간다.

    Python 3.7+ 의 reconfigure() 를 쓴다. 파이프로 리다이렉트된 경우 등
    reconfigure 가 불가능한 상황에서는 아무 것도 하지 않는다.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass
