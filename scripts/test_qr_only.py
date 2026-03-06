"""QR 디코딩만 테스트 (Naver OCR 비용 없음).

Usage:
    uv run python3 scripts/test_qr_only.py [image_dir]
"""
import os
import sys

from forma.qr_decode import decode_qr_from_image

d = sys.argv[1] if len(sys.argv) > 1 else "exams/anp_w1/anp_1A_W1"
files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))
fails = []

for f in files:
    result = decode_qr_from_image(os.path.join(d, f))
    if not result:
        fails.append(f)
        print(f"  FAIL: {f}")

ok = len(files) - len(fails)
print(f"\n{ok}/{len(files)} decoded ({ok * 100 // len(files)}%)")
