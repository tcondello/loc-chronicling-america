"""Command-line interface for loc-chronicling-america."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .client import ChroniclingAmerica


console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loc-chronam",
        description="Library of Congress Chronicling America Research Tool - Discover, Resolve & Download Newspaper Records",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Resolve Command
    p_resolve = subparsers.add_parser("resolve", help="Resolve an LoC URL or LCCN to its batch, pages, and download links")
    p_resolve.add_argument("target", help="LoC URL (e.g. https://www.loc.gov/resource/00225879/1915-07-03/ed-1/) or LCCN")
    p_resolve.add_argument("--date", help="Publication date (YYYY-MM-DD), required if passing an LCCN")
    p_resolve.add_argument("--edition", type=int, default=1, help="Edition number (default: 1)")

    # Download Command
    p_download = subparsers.add_parser("download", help="Download assets for an issue or specific page")
    p_download.add_argument("target", help="LoC URL or LCCN")
    p_download.add_argument("--date", help="Publication date (YYYY-MM-DD) if passing LCCN")
    p_download.add_argument("--edition", type=int, default=1, help="Edition number (default: 1)")
    p_download.add_argument("--dest", "-d", default="./downloads", help="Destination directory (default: ./downloads)")
    p_download.add_argument("--format", "-f", choices=["pdf", "txt", "alto", "jp2", "image", "jpg", "all"], default="pdf", help="Asset format to download")
    p_download.add_argument("--page", "-p", type=int, help="Specific page sequence number (optional)")

    # Batch Command
    p_batch = subparsers.add_parser("batch", help="Batch inspection and bulk download operations")
    batch_subs = p_batch.add_subparsers(dest="batch_command", required=True)

    p_b_info = batch_subs.add_parser("info", help="Display metadata for a batch")
    p_b_info.add_argument("name", help="Batch identifier (e.g. nbu_indescribablebeast_ver01)")

    p_b_dl = batch_subs.add_parser("download", help="Download complete .tar.bz2 bulk OCR archive")
    p_b_dl.add_argument("name", help="Batch identifier")
    p_b_dl.add_argument("--dest", "-d", default="./downloads", help="Destination directory")

    # Search Command
    p_search = subparsers.add_parser("search", help="Search newspaper titles or batches in local catalog")
    search_subs = p_search.add_subparsers(dest="search_command", required=True)

    p_s_titles = search_subs.add_parser("titles", help="Search digitized newspaper titles")
    p_s_titles.add_argument("query", nargs="?", help="Title or keyword search query")
    p_s_titles.add_argument("--state", help="Filter by state (e.g. Nebraska)")
    p_s_titles.add_argument("--city", help="Filter by city (e.g. Omaha)")
    p_s_titles.add_argument("--limit", type=int, default=20, help="Results limit (default: 20)")

    p_s_batches = search_subs.add_parser("batches", help="Search batches")
    p_s_batches.add_argument("query", nargs="?", help="Batch name or LCCN query")
    p_s_batches.add_argument("--state", help="Filter by state / awardee code")
    p_s_batches.add_argument("--limit", type=int, default=20, help="Results limit (default: 20)")

    # Catalog Command
    p_cat = subparsers.add_parser("catalog", help="Manage local SQLite catalog")
    p_cat.add_argument("--sync", action="store_true", help="Synchronize batches and titles into SQLite")

    return parser


def handle_resolve(client: ChroniclingAmerica, args: argparse.Namespace) -> None:
    with console.status("[bold green]Resolving record from Library of Congress..."):
        try:
            record = client.resolve(args.target, date=args.date, edition=args.edition)
        except Exception as e:
            console.print(f"[bold red]Resolution error:[/bold red] {e}")
            sys.exit(1)

    # Summary Panel
    meta_text = (
        f"[bold cyan]Title:[/bold cyan] {record.title}\n"
        f"[bold cyan]Publication:[/bold cyan] {record.newspaper_title} ({record.place_of_publication or 'Unknown'})\n"
        f"[bold cyan]LCCN / Date:[/bold cyan] {record.lccn} | {record.date} (Edition {record.edition})\n"
        f"[bold yellow]Source Batch:[/bold yellow] [bold]{record.batch_name}[/bold]\n"
        f"[bold cyan]Pages:[/bold cyan] {record.page_count} pages available\n"
        f"[bold cyan]Bulk Archive:[/bold cyan] {record.bulk_ocr_url}\n"
        f"[bold cyan]Raw Batch Dir:[/bold cyan] {record.raw_batch_url}\n"
        f"[bold cyan]LoC Item Permalink:[/bold cyan] {record.loc_item_url}"
    )
    console.print(Panel(meta_text, title="Newspaper Record Resolved", border_style="green"))

    # Pages Table
    table = Table(title="Available Page Assets", show_header=True, header_style="bold magenta")
    table.add_column("Seq", style="dim", width=6)
    table.add_column("Dimensions", width=12)
    table.add_column("PDF", style="cyan")
    table.add_column("ALTO XML", style="cyan")
    table.add_column("JP2 Master", style="cyan")
    table.add_column("Plain Text OCR", style="green")

    for p in record.pages:
        dims = f"{p.width}x{p.height}" if p.width and p.height else "Scan"
        has_pdf = "[green]Yes[/green]" if p.pdf_url else "[red]No[/red]"
        has_alto = "[green]Yes[/green]" if p.alto_xml_url else "[red]No[/red]"
        has_jp2 = "[green]Yes[/green]" if p.jp2_url else "[red]No[/red]"
        has_txt = "[green]Available[/green]" if p.text_service_url or p.alto_xml_url else "[red]No[/red]"
        table.add_row(f"Page {p.sequence}", dims, has_pdf, has_alto, has_jp2, has_txt)

    console.print(table)


def handle_download(client: ChroniclingAmerica, args: argparse.Namespace) -> None:
    with console.status("[bold green]Resolving record for download..."):
        record = client.resolve(args.target, date=args.date, edition=args.edition)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    if args.page:
        page = record.get_page(args.page)
        console.print(f"[bold]Downloading Page {args.page} in format '{args.format}' to {dest}...[/bold]")
        if args.format == "pdf":
            p = page.download_pdf(dest)
            client.catalog.record_download(f"{record.lccn}/{record.date}/page-{args.page}", "pdf", p, p.stat().st_size, batch_name=record.batch_name)
        elif args.format == "txt":
            p = page.download_text(dest)
            client.catalog.record_download(f"{record.lccn}/{record.date}/page-{args.page}", "txt", p, p.stat().st_size, batch_name=record.batch_name)
        elif args.format == "alto":
            p = page.download_alto(dest)
            client.catalog.record_download(f"{record.lccn}/{record.date}/page-{args.page}", "alto", p, p.stat().st_size, batch_name=record.batch_name)
        elif args.format == "jp2":
            p = page.download_jp2(dest)
            client.catalog.record_download(f"{record.lccn}/{record.date}/page-{args.page}", "jp2", p, p.stat().st_size, batch_name=record.batch_name)
        elif args.format in ("image", "jpg"):
            p = page.download_image(dest)
            client.catalog.record_download(f"{record.lccn}/{record.date}/page-{args.page}", "image", p, p.stat().st_size, batch_name=record.batch_name)
        console.print(f"[bold green]✓ Downloaded:[/bold green] {p}")
    else:
        console.print(f"[bold]Downloading all {record.page_count} pages in format '{args.format}' to {dest}...[/bold]")
        results = record.download_all(dest, asset_types=args.format)
        for fmt, files in results.items():
            for f in files:
                client.catalog.record_download(f"{record.lccn}/{record.date}/{f.name}", fmt, f, f.stat().st_size, batch_name=record.batch_name)
                console.print(f"[bold green]✓ Downloaded:[/bold green] {f.name}")


def handle_batch(client: ChroniclingAmerica, args: argparse.Namespace) -> None:
    batch = client.get_batch(args.name)

    if args.batch_command == "info":
        with console.status("[bold green]Fetching batch metadata..."):
            info = batch.get_info()

        info_text = (
            f"[bold cyan]Batch Identifier:[/bold cyan] {info.name}\n"
            f"[bold cyan]Awardee Code:[/bold cyan] {info.awardee or 'N/A'}\n"
            f"[bold cyan]Archive Size:[/bold cyan] {info.size_mb} MB ({info.size_gb} GB)\n"
            f"[bold cyan]Pages / Issues:[/bold cyan] {info.page_count or 'N/A'} pages | {info.issue_count or 'N/A'} issues\n"
            f"[bold cyan]SHA-256 Checksum:[/bold cyan] {info.sha256 or 'N/A'}\n"
            f"[bold cyan]LCCNs in Batch:[/bold cyan] {', '.join(info.lccns) if info.lccns else 'Parsed from manifest'}\n"
            f"[bold cyan]Bulk Download URL:[/bold cyan] {batch.bulk_archive_url}\n"
            f"[bold cyan]Raw BagIt Directory:[/bold cyan] {batch.raw_batch_url}"
        )
        console.print(Panel(info_text, title=f"Batch Information: {info.name}", border_style="cyan"))

    elif args.batch_command == "download":
        console.print(f"[bold]Downloading bulk archive for batch {batch.name}...[/bold]")
        path = batch.download(args.dest)
        client.catalog.record_download(batch.name, "bulk_archive", path, path.stat().st_size, batch_name=batch.name)
        console.print(f"[bold green]✓ Downloaded bulk archive:[/bold green] {path}")


def handle_search(client: ChroniclingAmerica, args: argparse.Namespace) -> None:
    if args.search_command == "titles":
        titles = client.search_titles(query=args.query, state=args.state, city=args.city, limit=args.limit)
        if not titles:
            console.print("[yellow]No newspaper titles matched your query.[/yellow]")
            return

        table = Table(title=f"Newspaper Titles ({len(titles)} results)", show_header=True, header_style="bold cyan")
        table.add_column("LCCN", style="bold green", width=12)
        table.add_column("Title", style="bold")
        table.add_column("Location")
        table.add_column("Years", width=12)
        table.add_column("Issues", justify="right", width=8)

        for t in titles:
            loc = f"{t.city}, {t.state}" if t.city and t.state else (t.state or "USA")
            table.add_row(t.lccn, t.name, loc, t.formatted_years, str(t.issue_count or "?"))

        console.print(table)

    elif args.search_command == "batches":
        batches = client.search_batches(query=args.query, state=args.state, limit=args.limit)
        if not batches:
            console.print("[yellow]No batches matched your query.[/yellow]")
            return

        table = Table(title=f"Batches ({len(batches)} results)", show_header=True, header_style="bold cyan")
        table.add_column("Batch Name", style="bold green")
        table.add_column("Awardee", width=10)
        table.add_column("Size (MB)", justify="right", width=12)
        table.add_column("Ingested", width=14)

        for b in batches:
            table.add_row(b.name, b.awardee or "?", str(b.size_mb), b.ingested_date or "?")

        console.print(table)


def handle_catalog(client: ChroniclingAmerica, args: argparse.Namespace) -> None:
    if args.sync:
        console.print("[bold]Synchronizing local SQLite catalog from Chronicling America...[/bold]")
        with console.status("Syncing batches..."):
            b_count = client.sync_batches()
        console.print(f"[green]✓ Synced {b_count} batches.[/green]")

        with console.status("Syncing newspaper titles..."):
            t_count = client.sync_titles()
        console.print(f"[green]✓ Synced {t_count} titles.[/green]")

    b_total = client.catalog.count_batches()
    t_total = client.catalog.count_titles()
    dl_total = len(client.catalog.list_downloads())

    console.print(
        Panel(
            f"[bold cyan]Database Path:[/bold cyan] {client.catalog.db_path}\n"
            f"[bold cyan]Indexed Batches:[/bold cyan] {b_total}\n"
            f"[bold cyan]Indexed Titles:[/bold cyan] {t_total}\n"
            f"[bold cyan]Local Downloaded Records:[/bold cyan] {dl_total}",
            title="Local SQLite Catalog Status",
            border_style="blue",
        )
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    client = ChroniclingAmerica()

    if args.command == "resolve":
        handle_resolve(client, args)
    elif args.command == "download":
        handle_download(client, args)
    elif args.command == "batch":
        handle_batch(client, args)
    elif args.command == "search":
        handle_search(client, args)
    elif args.command == "catalog":
        handle_catalog(client, args)


if __name__ == "__main__":
    main()
