from dataclasses import dataclass
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
import re


@dataclass(frozen=True)
class Unit:
    start: int
    end: int
    metadata: dict


@dataclass(frozen=True)
class Document:
    source: str
    title: str
    text: str
    units: tuple

    @property
    def version(self):
        return sha256(self.text.encode()).hexdigest()


class KnowledgeHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.tag = None
        self.buffer = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.hidden += 1
        if not self.hidden and tag in ('title', 'h2', 'p'):
            self.tag, self.buffer = tag, []

    def handle_data(self, data):
        if not self.hidden and self.tag:
            self.buffer.append(data)

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.hidden = max(0, self.hidden - 1)
        if tag == self.tag:
            self.parts.append((tag, ' '.join(''.join(self.buffer).split())))
            self.tag, self.buffer = None, []


def load_document(path):
    path = Path(path)
    raw = path.read_text(encoding='utf-8').replace('\r\n', '\n').replace('\r', '\n')
    suffix = path.suffix.lower()
    title = path.stem
    pieces, units = [], []
    cursor = 0

    def append(text, metadata):
        nonlocal cursor
        text = text.strip()
        if not text:
            return
        if pieces:
            pieces.append('\n\n')
            cursor += 2
        start = cursor
        pieces.append(text)
        cursor += len(text)
        units.append(Unit(start, cursor, metadata))

    if suffix == '.md':
        headings = list(re.finditer(r'^#{1,6}\s+(.+)$', raw, re.M))
        if headings:
            title = headings[0].group(1).strip()
            append(raw[:headings[0].start()], {'section': 'Preamble'})
            for i, match in enumerate(headings):
                end = headings[i + 1].start() if i + 1 < len(headings) else len(raw)
                append(raw[match.start():end], {'section': match.group(1).strip()})
        else:
            append(raw, {'section': title})
    elif suffix == '.html':
        parser = KnowledgeHTML()
        parser.feed(raw)
        pending = None
        for tag, text in parser.parts:
            if tag == 'title':
                title = text
            elif tag == 'h2':
                if pending is not None:
                    raise ValueError('Each HTML question needs an adjacent paragraph answer')
                pending = text
            elif tag == 'p':
                if pending is None:
                    raise ValueError('HTML paragraph has no question')
                append(pending + '\n' + text, {'question': pending})
                pending = None
        if pending is not None:
            raise ValueError('HTML question has no answer')
    elif suffix == '.txt':
        turns = list(re.finditer(r'^([A-Za-z][\w ]*):[ \t]*', raw, re.M))
        for i, match in enumerate(turns):
            end = turns[i + 1].start() if i + 1 < len(turns) else len(raw)
            append(raw[match.start():end], {'speaker': match.group(1)})
    else:
        raise ValueError('Supported sources: .md, .html, .txt')
    if not units:
        raise ValueError('Document contains no supported units')
    return Document(path.as_posix(), title, ''.join(pieces), tuple(units))
