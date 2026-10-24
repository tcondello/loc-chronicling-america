"""Smoke test for the search terminal UI. It does not hit the network."""

import asyncio

import pytest


def test_tui_opens_help_and_accepts_a_query():
    pytest.importorskip("textual")
    from loc_chronicling_america.explore_tui import make_app

    async def body() -> None:
        app = make_app()
        async with app.run_test(size=(140, 42)) as pilot:
            await pilot.pause()
            await pilot.press("f1")
            await pilot.pause()
            assert type(app.screen_stack[-1]).__name__ == "HelpScreen"
            await pilot.press("escape")
            await pilot.pause()
            assert len(app.screen_stack) == 1
            await pilot.click("#query")
            await pilot.press(*list('"buried alive"'))
            await pilot.pause()
            assert '"buried alive"' in app.query_one("#query").value

    asyncio.run(body())
