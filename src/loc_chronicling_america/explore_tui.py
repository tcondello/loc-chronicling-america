"""Terminal UI for searching Chronicling America OCR on Hugging Face."""

from __future__ import annotations

import threading
from typing import List

from .explore_query import ExploreQueryError, query_help
from .explore_search import Hit, list_papers, paper_layout, search_pages


def make_app(
    *,
    state: str = "new_york",
    paper: str = "the_evening_world",
    year_from: int = 1890,
    year_to: int = 1890,
    limit: int = 30,
    readable: bool = True,
) -> None:
    try:
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Horizontal, Vertical
        from textual.screen import ModalScreen
        from textual.widgets import Button, Checkbox, DataTable, Footer, Header, Input, Label, Static
    except ImportError as exc:
        raise SystemExit(
            "The search UI needs Textual. From the repo root run: uv sync --extra dev"
        ) from exc

    class HelpScreen(ModalScreen[None]):
        BINDINGS = [Binding("escape", "close", "Close")]

        def compose(self) -> ComposeResult:
            with Vertical(id="help-box"):
                yield Static(query_help(), id="help-text")
                yield Button("Close", id="close-help")

        def action_close(self) -> None:
            self.dismiss(None)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "close-help":
                self.dismiss(None)

    class ExploreApp(App[None]):
        TITLE = "Chronicling America search"
        CSS = """
        #scope { height: 3; }
        #query-row { height: 3; }
        #status { height: 1; color: $text-muted; }
        #results { height: 1fr; }
        #detail { height: 12; padding: 0 1; }
        #help-box {
            width: 88;
            height: auto;
            max-height: 90%;
            padding: 1 2;
            border: solid $accent;
            background: $surface;
        }
        #help-text { height: auto; }
        """
        BINDINGS = [
            Binding("f1", "help", "Help"),
            Binding("ctrl+r", "search", "Search"),
            Binding("ctrl+p", "papers", "Papers"),
            Binding("ctrl+y", "years", "Years"),
            Binding("ctrl+q", "quit", "Quit"),
        ]

        def __init__(self) -> None:
            super().__init__()
            self._hits: List[Hit] = []
            self._cancel = threading.Event()
            self._generation = 0
            self._defaults = {
                "state": state,
                "paper": paper,
                "year_from": str(year_from),
                "year_to": str(year_to),
                "limit": str(limit),
                "readable": readable,
            }

        def compose(self) -> ComposeResult:
            yield Header()
            with Horizontal(id="scope"):
                yield Label("State")
                yield Input(value=self._defaults["state"], id="state", placeholder="new_york")
                yield Label("Paper")
                yield Input(value=self._defaults["paper"], id="paper", placeholder="the_evening_world")
                yield Label("From")
                yield Input(value=self._defaults["year_from"], id="year-from", placeholder="1890")
                yield Label("To")
                yield Input(value=self._defaults["year_to"], id="year-to", placeholder="1895")
                yield Label("Limit")
                yield Input(value=self._defaults["limit"], id="limit", placeholder="30")
                yield Checkbox("Readable English", value=self._defaults["readable"], id="readable")
            with Horizontal(id="query-row"):
                yield Input(placeholder=' "buried alive" OR "haunted house" -hazel    year:[1890 TO 1895]', id="query")
            yield Static("Enter runs the search. F1 lists the dialect. Ctrl+P lists papers. Ctrl+Y lists years.", id="status")
            yield DataTable(id="results", cursor_type="row", zebra_stripes=True)
            yield Static("", id="detail")
            yield Footer()

        def on_mount(self) -> None:
            table = self.query_one(DataTable)
            table.add_columns("Date", "Pg", "Read", "Excerpt")
            self.query_one("#query", Input).focus()

        def action_help(self) -> None:
            self.push_screen(HelpScreen())

        def action_papers(self) -> None:
            state_slug = self.query_one("#state", Input).value
            self._set_status(f"Listing papers in {state_slug}...")
            self.run_worker(
                lambda: self._load_papers(state_slug),
                thread=True,
                exclusive=True,
                group="catalog",
            )

        def action_years(self) -> None:
            state_slug = self.query_one("#state", Input).value
            paper_slug = self.query_one("#paper", Input).value
            self._set_status("Looking up years...")
            self.run_worker(
                lambda: self._load_years(state_slug, paper_slug),
                thread=True,
                exclusive=True,
                group="catalog",
            )

        def on_input_submitted(self, event: Input.Submitted) -> None:
            if event.input.id == "query":
                self.action_search()

        def action_search(self) -> None:
            try:
                scope = self._scope()
                query = self.query_one("#query", Input).value.strip()
                if not query:
                    raise ExploreQueryError("Enter a word, phrase, or /regex/.")
            except (ExploreQueryError, ValueError) as exc:
                self._set_status(str(exc))
                return
            self._cancel.set()
            self._cancel = threading.Event()
            self._generation += 1
            generation = self._generation
            token = self._cancel
            span = scope["year_to"] - scope["year_from"] + 1
            self._set_status(f"Starting a {span}-year scan. A daily paper is often about 10 seconds per year.")
            self.run_worker(
                lambda: self._search(query, scope, generation, token),
                thread=True,
                exclusive=True,
                group="search",
            )

        def _scope(self) -> dict:
            def number(widget_id: str, label: str) -> int:
                raw = self.query_one(widget_id, Input).value.strip()
                try:
                    return int(raw)
                except ValueError as exc:
                    raise ExploreQueryError(f"{label} needs a whole number.") from exc

            return {
                "state": self.query_one("#state", Input).value,
                "paper": self.query_one("#paper", Input).value,
                "year_from": number("#year-from", "From-year"),
                "year_to": number("#year-to", "To-year"),
                "limit": number("#limit", "Limit"),
                "readable": self.query_one("#readable", Checkbox).value,
            }

        def _search(self, query: str, scope: dict, generation: int, token: threading.Event) -> None:
            def progress(message: str) -> None:
                if generation == self._generation:
                    self.call_from_thread(self._set_status, message)

            try:
                hits, explanation = search_pages(
                    query,
                    state=scope["state"],
                    paper=scope["paper"],
                    year_from=scope["year_from"],
                    year_to=scope["year_to"],
                    readable=scope["readable"],
                    limit=scope["limit"],
                    progress=progress,
                    cancel=token,
                )
            except ExploreQueryError as exc:
                self.call_from_thread(self._set_status, str(exc))
                return
            except Exception as exc:
                self.call_from_thread(self._set_status, f"Search failed: {exc}")
                return
            if generation != self._generation:
                return
            self.call_from_thread(self._show_hits, hits, explanation)

        def _load_papers(self, state_slug: str) -> None:
            try:
                names = list_papers(state_slug)
            except Exception as exc:
                self.call_from_thread(self._set_detail, f"Could not list papers.\n{exc}")
                return
            lines = "\n".join(names) if names else "No papers in that state folder."
            self.call_from_thread(self._set_detail, lines)
            self.call_from_thread(self._set_status, f"{len(names)} papers. Paste a slug into Paper.")

        def _load_years(self, state_slug: str, paper_slug: str) -> None:
            try:
                layout = paper_layout(state_slug, paper_slug)
            except Exception as exc:
                self.call_from_thread(self._set_detail, f"Could not list years.\n{exc}")
                return
            if layout.flat:
                text = "This paper stores issues in one folder. Year filters still apply."
            elif layout.years:
                text = f"{layout.years[0]}-{layout.years[-1]} ({len(layout.years)} years)\n" + ", ".join(
                    str(year) for year in layout.years
                )
            else:
                text = "No years found."
            self.call_from_thread(self._set_detail, text)
            self.call_from_thread(self._set_status, "Years loaded. Widen From and To to scan more of them.")

        def _show_hits(self, hits: List[Hit], explanation: str) -> None:
            self._hits = hits
            table = self.query_one(DataTable)
            table.clear()
            for index, hit in enumerate(hits):
                table.add_row(
                    hit.date,
                    str(hit.page),
                    f"{hit.readable:.2f}",
                    hit.excerpt[:96],
                    key=str(index),
                )
            if hits:
                table.focus()
                self._show_hit(0)
            else:
                self._set_detail("No readable pages matched. Try a looser phrase, or turn off Readable English.")
            self._set_status(f"{len(hits)} shown. {explanation}")

        def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
            self._show_hit_key(event.row_key)

        def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
            self._show_hit_key(event.row_key)

        def _show_hit_key(self, row_key: object) -> None:
            value = getattr(row_key, "value", row_key)
            try:
                index = int(str(value))
            except (TypeError, ValueError):
                return
            self._show_hit(index)

        def _show_hit(self, index: int) -> None:
            if index < 0 or index >= len(self._hits):
                return
            hit = self._hits[index]
            place = ", ".join(part for part in (hit.city, hit.state) if part)
            self._set_detail(
                f"{hit.newspaper_title}  {hit.date}  page {hit.page}  {place}\n"
                f"Readable {hit.readable:.2f}   {hit.word_count} words\n"
                f"{hit.url}\n\n"
                f"{hit.excerpt}"
            )

        def _set_status(self, message: str) -> None:
            self.query_one("#status", Static).update(message)

        def _set_detail(self, message: str) -> None:
            self.query_one("#detail", Static).update(message)

    return ExploreApp()


def run_tui(
    *,
    state: str = "new_york",
    paper: str = "the_evening_world",
    year_from: int = 1890,
    year_to: int = 1890,
    limit: int = 30,
    readable: bool = True,
) -> None:
    make_app(
        state=state,
        paper=paper,
        year_from=year_from,
        year_to=year_to,
        limit=limit,
        readable=readable,
    ).run()


def format_hits(hits: List[Hit], explanation: str) -> str:
    """Plain-text rendering for a one-shot search."""
    lines = [explanation, ""]
    if not hits:
        lines.append("No pages matched.")
        return "\n".join(lines)
    for hit in hits:
        lines.append(
            f"{hit.date}  p{hit.page}  read {hit.readable:.2f}  {hit.newspaper_title}"
        )
        lines.append(f"  {hit.url}")
        lines.append(f"  {hit.excerpt[:400]}")
        lines.append("")
    return "\n".join(lines)
