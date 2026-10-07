"""Exercise the real dashboard against the same API contract, without network services."""

from pathlib import Path

import httpx
import streamlit as st
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from retail_pricing.api import app, get_service


def test_dashboard_pages_and_scenario(service, monkeypatch):
    app.dependency_overrides[get_service] = lambda: service
    client = TestClient(app)

    class LocalClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url):
            return client.get(url)

        def post(self, url, json):
            return client.post(url, json=json)

    monkeypatch.setattr(httpx, "Client", LocalClient)
    st.cache_data.clear()
    try:
        at = AppTest.from_file(str(Path(__file__).parents[1] / "dashboard/app.py")).run()
        assert not at.exception
        for page in ["Elasticity & promotions", "Model validation", "Scenario simulator"]:
            at.sidebar.radio[0].set_value(page).run()
            assert not at.exception, str(at.exception)
        at.button[0].click().run()
        assert not at.exception
        assert any(m.label == "Simulated revenue" for m in at.metric)
        assert not at.error
        at.button[1].click().run()
        assert not at.exception
        assert any("Best tested candidate" in s.value for s in at.success)
    finally:
        client.close()
        app.dependency_overrides.clear()
        st.cache_data.clear()
