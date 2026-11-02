from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata


# A deliberately narrow offline alphabet, not a general invoice model.
GLYPHS = {
    'I': '11111 00100 00100 00100 00100 00100 11111',
    'N': '10001 11001 11001 10101 10011 10011 10001',
    'V': '10001 10001 10001 10001 10001 01010 00100',
    'O': '01110 10001 10001 10001 10001 10001 01110',
    'C': '01111 10000 10000 10000 10000 10000 01111',
    'E': '11111 10000 10000 11110 10000 10000 11111',
    'T': '11111 00100 00100 00100 00100 00100 00100',
    'A': '01110 10001 10001 11111 10001 10001 10001',
    'L': '10000 10000 10000 10000 10000 10000 11111',
    '4': '10010 10010 10010 11111 00010 00010 00010',
    '2': '01110 10001 00001 00010 00100 01000 11111',
    '1': '00100 01100 00100 00100 00100 00100 01110',
    '8': '01110 10001 10001 01110 10001 10001 01110',
    ' ': '00000 00000 00000 00000 00000 00000 00000',
}
GLYPHS = {key: ''.join(value.split()) for key, value in GLYPHS.items()}


@dataclass(frozen=True)
class Limits:
    max_bytes: int = 2_000_000
    max_pages: int = 8
    max_pixels: int = 10_000_000
    seconds: float = 5.0
    max_char_error: float = 0.05

    def __post_init__(self):
        integers = (self.max_bytes, self.max_pages, self.max_pixels)
        if any(type(value) is not int or value <= 0 for value in integers):
            raise ValueError('invalid_ocr_limits')
        if not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError('invalid_ocr_deadline')
        if not 0 <= self.max_char_error <= 1:
            raise ValueError('invalid_character_error_threshold')


@dataclass(frozen=True)
class Page:
    page: int
    text: str
    quality: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class Extraction:
    source_sha256: str
    engine: str
    fixture: bool
    pages: tuple[Page, ...]
    runtime_ms: float

    def require_evidence(self, *, allow_fixture=False):
        if self.fixture and not allow_fixture:
            raise ValueError('fixture_evidence_not_production_verified')
        accepted = tuple(page for page in self.pages if page.quality == 'accepted')
        if not accepted:
            raise ValueError('no_accepted_ocr_evidence')
        return accepted

    def as_dict(self):
        return asdict(self)


def normalize(text):
    text = unicodedata.normalize('NFKC', text).replace('\r', '\n')
    return '\n'.join(' '.join(line.split()) for line in text.splitlines() if line.strip())


def page_result(number, text, *, unreadable=False):
    if type(number) is not int or number <= 0 or not isinstance(text, str):
        raise ValueError('invalid_ocr_page')
    text = normalize(text)
    if unreadable:
        return Page(number, '', 'unreadable', ('recognition_failed',))
    if not text:
        return Page(number, '', 'empty', ('no_text',))
    # This is a structural screen, not a calibrated invoice-accuracy score.
    if sum(char.isalnum() for char in text) < 4:
        return Page(number, '', 'unreadable', ('insufficient_readable_text',))
    return Page(number, text, 'accepted', ('accuracy_not_calibrated',))


def rotate_left(rows):
    return [list(row) for row in zip(*rows)][::-1]


def recognize_fixture(rows):
    rows = rotate_left(rows)
    if len(rows) != 160 or any(len(row) != 400 for row in rows):
        raise ValueError('unsupported_fixture_layout')
    if not any(value < 128 for row in rows for value in row):
        return '', False
    lines, errors = [], 0
    for line in range(2):
        chars = []
        for column in range(10):
            glyph = ''.join(
                '1' if rows[20 + line * 40 + y * 3 + 1][20 + column * 18 + x * 3 + 1] < 128 else '0'
                for y in range(7) for x in range(5))
            costs = [(sum(a != b for a, b in zip(glyph, pattern)), char)
                     for char, pattern in GLYPHS.items()]
            error, char = min(costs)
            errors += error
            chars.append(char)
        lines.append(''.join(chars).rstrip())
    return '\n'.join(lines), errors > 35


def fixture_pages(raw, limits):
    # Only the generated, uncompressed image-PDF subset is supported offline.
    if not raw.startswith(b'%PDF-1.4\n%ANCHOR-PIXEL-FIXTURE\n'):
        raise ValueError('unsupported_offline_pdf')
    count = re.search(rb'/Type /Pages /Count (\d+)', raw)
    if count is None or not 0 < int(count[1]) <= limits.max_pages:
        raise ValueError('ocr_page_limit')
    pages = []
    pattern = rb'/Subtype /Image /Width (\d+) /Height (\d+) /ColorSpace /DeviceGray /BitsPerComponent 8 /Length (\d+) >>\nstream\n'
    matches = list(re.finditer(pattern, raw))
    if len(matches) != int(count[1]):
        raise ValueError('malformed_fixture_pdf')
    total = 0
    for number, match in enumerate(matches, 1):
        width, height, length = map(int, match.groups())
        total += width * height
        if width <= 0 or height <= 0 or total > limits.max_pixels:
            raise ValueError('ocr_raster_limit')
        pixels = raw[match.end():match.end() + length]
        if length != width * height or len(pixels) != length:
            raise ValueError('malformed_fixture_raster')
        rows = [list(pixels[start:start + width]) for start in range(0, length, width)]
        text, unreadable = recognize_fixture(rows)
        pages.append(page_result(number, text, unreadable=unreadable))
    return tuple(pages)


def run_tool(arguments, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('ocr_deadline')
    if shutil.which(arguments[0]) is None:
        raise ValueError('local_ocr_infrastructure_missing')
    try:
        result = subprocess.run(arguments, check=True, capture_output=True,
                                timeout=remaining)
    except subprocess.TimeoutExpired:
        raise TimeoutError('ocr_deadline') from None
    except subprocess.CalledProcessError:
        raise ValueError('local_ocr_tool_failed') from None
    return result.stdout


def local_pages(raw, limits, deadline):
    with tempfile.TemporaryDirectory(prefix='anchor-ocr-') as temporary:
        root = Path(temporary)
        source = root / 'input.pdf'
        source.write_bytes(raw)
        info = run_tool(['pdfinfo', str(source)], deadline).decode('utf-8', errors='replace')
        count = re.search(r'^Pages:\s+(\d+)', info, re.M)
        if count is None or not 0 < int(count[1]) <= limits.max_pages:
            raise ValueError('ocr_page_limit')
        sizes, total = [], 0
        for number in range(1, int(count[1]) + 1):
            info = run_tool(['pdfinfo', '-f', str(number), '-l', str(number),
                             '-box', str(source)], deadline).decode('utf-8', errors='replace')
            size = re.search(r'Page\s+(?:\d+\s+)?size:\s+([\d.]+)\s+x\s+([\d.]+)', info)
            if size is None:
                raise ValueError('pdf_geometry_unavailable')
            width, height = (math.ceil(float(value) * 300 / 72) for value in size.groups())
            total += width * height
            if min(width, height) <= 0 or total > limits.max_pixels:
                raise ValueError('ocr_raster_limit')
            sizes.append((width, height))
        pages = []
        for number in range(1, len(sizes) + 1):
            prefix = root / 'page'
            run_tool(['pdftoppm', '-f', str(number), '-l', str(number), '-singlefile',
                      '-r', '300', '-gray', str(source), str(prefix)], deadline)
            image = root / 'page.pgm'
            data = image.read_bytes()
            header = re.match(rb'P5\s+(\d+)\s+(\d+)\s+255\s', data)
            if header is None:
                raise ValueError('unsupported_raster')
            width, height = map(int, header.groups())
            pixels = data[header.end():]
            if width * height > limits.max_pixels or len(pixels) != width * height:
                raise ValueError('ocr_raster_limit')
            # The supplied invoice has a known clockwise quarter-turn.
            rows = [pixels[start:start + width] for start in range(0, len(pixels), width)]
            upright = rotate_left(rows)
            image.write_bytes(f'P5\n{height} {width}\n255\n'.encode() +
                              bytes(value for row in upright for value in row))
            text = run_tool(['tesseract', str(image), 'stdout', '-l', 'eng', '--psm', '6'],
                            deadline).decode('utf-8', errors='replace')
            pages.append(page_result(number, text))
        return tuple(pages)


class RecordedVision:
    def __init__(self, expected_pdf, response_path):
        self.expected_digest = sha256(expected_pdf).hexdigest()
        self.records = json.loads(Path(response_path).read_text())

    def extract(self, raw):
        if sha256(raw).hexdigest() != self.expected_digest:
            raise ValueError('unrecorded_vision_input')
        return tuple(page_result(row['page'], row['text'],
                                 unreadable=row.get('unreadable', False))
                     for row in self.records['pages'])


def extract_pdf(raw, *, engine='fixture', limits=None, vision=None):
    limits = limits or Limits()
    started = time.monotonic()
    deadline = started + limits.seconds
    if not isinstance(raw, bytes) or not raw.startswith(b'%PDF-') or len(raw) > limits.max_bytes:
        raise ValueError('ocr_input_rejected')
    if engine == 'tesseract':
        pages = local_pages(raw, limits, deadline)
    elif engine in {'fixture', 'vision_fixture'}:
        # Preflight and decode the same bounded fixture even for recorded vision.
        pages = fixture_pages(raw, limits)
        if engine == 'vision_fixture':
            if vision is None:
                raise ValueError('vision_fixture_missing')
            pages = vision.extract(raw)
    else:
        raise ValueError('ocr_engine_not_verified')
    if [page.page for page in pages] != list(range(1, len(pages) + 1)):
        raise ValueError('ocr_page_provenance_rejected')
    if len(pages) > limits.max_pages:
        raise ValueError('ocr_page_limit')
    if time.monotonic() >= deadline:
        raise TimeoutError('ocr_deadline')
    return Extraction(sha256(raw).hexdigest(), engine, engine != 'tesseract', pages,
                      (time.monotonic() - started) * 1000)


def character_error(text, reference):
    text, reference = normalize(text), normalize(reference)
    if not reference:
        raise ValueError('empty_reference')
    previous = list(range(len(reference) + 1))
    for i, left in enumerate(text, 1):
        current = [i]
        for j, right in enumerate(reference, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1] / len(reference)
