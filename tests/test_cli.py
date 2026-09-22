"""Tests for CLI argument parsing and commands."""

import pytest
from loc_chronicling_america.cli import build_parser


def test_cli_parser_resolve():
    parser = build_parser()
    args = parser.parse_args(["resolve", "https://www.loc.gov/resource/00225879/1915-07-03/ed-1/"])
    assert args.command == "resolve"
    assert "00225879" in args.target


def test_cli_parser_download():
    parser = build_parser()
    args = parser.parse_args([
        "download",
        "https://www.loc.gov/resource/00225879/1915-07-03/ed-1/",
        "--format", "pdf",
        "--dest", "/tmp/papers",
        "--page", "2",
    ])
    assert args.command == "download"
    assert args.format == "pdf"
    assert args.dest == "/tmp/papers"
    assert args.page == 2


def test_cli_parser_batch():
    parser = build_parser()
    args = parser.parse_args(["batch", "info", "nbu_indescribablebeast_ver01"])
    assert args.command == "batch"
    assert args.batch_command == "info"
    assert args.name == "nbu_indescribablebeast_ver01"


def test_cli_parser_search():
    parser = build_parser()
    args = parser.parse_args(["search", "titles", "Monitor", "--state", "Nebraska"])
    assert args.command == "search"
    assert args.search_command == "titles"
    assert args.query == "Monitor"
    assert args.state == "Nebraska"


def test_cli_parser_export_pipeline():
    parser = build_parser()
    args = parser.parse_args([
        "export-pipeline",
        "--state", "NE",
        "--output-dir", "/tmp/export",
        "--purge-after-upload",
        "--hf-repo", "my-org/my-dataset",
    ])
    assert args.command == "export-pipeline"
    assert args.state == "NE"
    assert args.output_dir == "/tmp/export"
    assert args.purge_after_upload is True
    assert args.hf_repo == "my-org/my-dataset"
