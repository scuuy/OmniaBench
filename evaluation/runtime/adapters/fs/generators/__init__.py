"""
generators：按 file_type 选择 fixture 生成器。
"""

from __future__ import annotations

from .archive import ZipBundleGenerator
from .base import BaseFixtureGenerator
from .fallback import DefaultBinaryLikeGenerator, DefaultStructuredTextGenerator
from .structured import CsvGenerator, JsonGenerator, JsonlGenerator, TsvGenerator, XlsxGenerator, YamlLikeGenerator
from .textual import CodeTextGenerator, ConfigTextGenerator, DocxGenerator, EmailGenerator, HtmlGenerator, PdfGenerator, TextLikeGenerator, XmlGenerator


_GENERATORS = [
    CsvGenerator(),
    TsvGenerator(),
    JsonGenerator(),
    JsonlGenerator(),
    YamlLikeGenerator(),
    XmlGenerator(),
    HtmlGenerator(),
    ConfigTextGenerator(),
    CodeTextGenerator(),
    XlsxGenerator(),
    DocxGenerator(),
    PdfGenerator(),
    EmailGenerator(),
    ZipBundleGenerator(),
    TextLikeGenerator(),
    DefaultStructuredTextGenerator(),
    DefaultBinaryLikeGenerator(),
]


def get_generator(fs_input: dict) -> BaseFixtureGenerator:
    for generator in _GENERATORS:
        if generator.supports(fs_input):
            return generator
    return DefaultStructuredTextGenerator()
