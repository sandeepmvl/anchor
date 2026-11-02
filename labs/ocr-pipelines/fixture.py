from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anchor_ai.ocr import GLYPHS

FIXTURES = ROOT / 'anchor_ai/tests/fixtures'


def invoice_pdf(*, blank=False, unreadable=False, pages=1):
    rows = [[255] * 400 for _ in range(160)]
    if not blank:
        for line, text in enumerate(('INVOICE 42', 'TOTAL 18')):
            for column, char in enumerate(text):
                glyph = GLYPHS[char]
                for y in range(7):
                    for x in range(5):
                        if glyph[y * 5 + x] == '1':
                            for dy in range(3):
                                for dx in range(3):
                                    rows[20 + line * 40 + y * 3 + dy][20 + column * 18 + x * 3 + dx] = 0
    if unreadable:
        rows = [[0] * 400 for _ in range(160)]
    sideways = [list(row) for row in zip(*rows[::-1])]
    pixels = bytes(value for row in sideways for value in row)
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>', b'']
    kids = []
    for _ in range(pages):
        page_id = len(objects) + 1
        image_id, content_id = page_id + 1, page_id + 2
        kids.append(f'{page_id} 0 R')
        objects.append((f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 160 400] '
                        f'/Resources << /XObject << /Im0 {image_id} 0 R >> >> '
                        f'/Contents {content_id} 0 R >>').encode())
        objects.append((f'<< /Type /XObject /Subtype /Image /Width 160 /Height 400 '
                        f'/ColorSpace /DeviceGray /BitsPerComponent 8 /Length {len(pixels)} >>\nstream\n').encode()
                       + pixels + b'\nendstream')
        content = b'q 160 0 0 400 0 0 cm /Im0 Do Q\n'
        objects.append(f'<< /Length {len(content)} >>\nstream\n'.encode() + content + b'endstream')
    objects[1] = f'<< /Type /Pages /Count {pages} /Kids [{" ".join(kids)}] >>'.encode()
    output = bytearray(b'%PDF-1.4\n%ANCHOR-PIXEL-FIXTURE\n')
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f'{number} 0 obj\n'.encode() + obj + b'\nendobj\n')
    xref = len(output)
    output.extend(f'xref\n0 {len(offsets)}\n0000000000 65535 f \n'.encode())
    for offset in offsets[1:]:
        output.extend(f'{offset:010d} 00000 n \n'.encode())
    output.extend((f'trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n'
                   f'startxref\n{xref}\n%%EOF\n').encode())
    return bytes(output)


def write_invoice():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    path = FIXTURES / 'scanned_invoice.pdf'
    path.write_bytes(invoice_pdf())
    return path


if __name__ == '__main__':
    print(write_invoice())
