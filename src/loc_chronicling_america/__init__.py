"""loc-chronicling-america: Research tool for Library of Congress Chronicling America collections."""

from .batch import Batch
from .client import ChroniclingAmerica
from .db import CatalogDB
from .downloader import Downloader, get_default_downloader
from .models import BatchInfo, DownloadRecord, IssueRecord, PageRecord, TitleInfo
from .parsers.alto import AltoBlock, AltoDocument, AltoLine, AltoWord, parse_alto_xml
from .parsers.mets import BatchIssueRef, MetsIssue, MetsPage, parse_batch_manifest, parse_mets_issue
from .hf import HuggingFaceDatasetManager
from .pipeline import BatchPipeline
from .resolver import RecordResolver

__version__ = "0.1.0"


__all__ = [
    "ChroniclingAmerica",
    "RecordResolver",
    "Batch",
    "BatchInfo",
    "IssueRecord",
    "PageRecord",
    "TitleInfo",
    "DownloadRecord",
    "CatalogDB",
    "Downloader",
    "get_default_downloader",
    "parse_alto_xml",
    "AltoDocument",
    "AltoBlock",
    "AltoLine",
    "AltoWord",
    "parse_mets_issue",
    "parse_batch_manifest",
    "MetsIssue",
    "MetsPage",
    "BatchIssueRef",
    "BatchPipeline",
    "HuggingFaceDatasetManager",
]

