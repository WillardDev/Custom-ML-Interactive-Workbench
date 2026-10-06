from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "ml_workbench" / "app.py"


def _app() -> AppTest:
    return AppTest.from_file(str(APP_PATH), default_timeout=10)


def _set_scenario(app: AppTest, scenario: str) -> AppTest:
    app.sidebar.radio("sim_scenario").set_value(scenario)
    return app.run()


def test_app_boots_on_data_tab() -> None:
    app = _app().run()
    assert not app.exception
    assert app.header[0].value == "1. Data Insertion"
    assert app.sidebar.radio("workflow_nav").value == "data"


def test_locked_tab_shows_gate_rule() -> None:
    app = _app().run()
    app.sidebar.radio("workflow_nav").set_value("training")
    app.run()
    assert not app.exception
    assert app.header[0].value == "6. Training"
    assert "Locked:" in app.info[0].value
    assert "GATE-0" in app.info[0].value


def test_scaffold_state_unlocks_training() -> None:
    app = _set_scenario(_app().run(), "Model trained")
    assert not app.exception
    app.sidebar.radio("workflow_nav").set_value("training")
    app.run()
    assert not app.exception
    assert app.header[0].value == "6. Training"
    assert not any("Locked:" in item.value for item in app.info)


def test_lower_scenario_relocks_tabs() -> None:
    app = _set_scenario(_app().run(), "Model trained")
    app = _set_scenario(app, "Nothing loaded")
    assert not app.exception
    app.sidebar.radio("workflow_nav").set_value("outcome")
    app.run()
    assert "Locked:" in app.info[0].value


def test_simulate_edit_marks_downstream_stale() -> None:
    app = _set_scenario(_app().run(), "Task selected")
    app.sidebar.radio("workflow_nav").set_value("cleaning")
    app.run()
    assert not app.exception
    assert not any("Locked:" in item.value for item in app.info)

    app.button("sim_edit").click().run()
    assert not app.exception
    stale: dict[str, bool] = app.session_state["state"].stale
    assert stale["data"] is False
    assert stale["cleaning"] is False
    assert sum(1 for value in stale.values() if value) == 8


def test_simulate_rerun_respects_stale_03() -> None:
    app = _set_scenario(_app().run(), "Model trained")
    app.sidebar.radio("workflow_nav").set_value("cleaning")
    app.run()
    app.button("sim_edit").click().run()
    app.sidebar.radio("workflow_nav").set_value("preprocessing")
    app.run()
    app.button("sim_rerun").click().run()
    stale: dict[str, bool] = app.session_state["state"].stale
    assert stale["preprocessing"] is False
    assert stale["eda"] is True
    assert stale["outcome"] is True
