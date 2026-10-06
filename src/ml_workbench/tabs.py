from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class TabInfo:
    id: str
    number: int
    title: str
    design_section: str
    implemented_phase: int


TABS: Final[tuple[TabInfo, ...]] = (
    TabInfo("data", 1, "Data Insertion", "6.1", 2),
    TabInfo("cleaning", 2, "Data Cleaning", "6.2", 2),
    TabInfo("preprocessing", 3, "Data Preprocessing", "6.3", 3),
    TabInfo("eda", 4, "Exploratory Data Analysis", "6.4", 3),
    TabInfo("modelling", 5, "Modelling", "6.5", 4),
    TabInfo("training", 6, "Training", "6.6", 4),
    TabInfo("prediction", 7, "Prediction", "6.7", 4),
    TabInfo("error_analysis", 8, "Error Analysis", "6.8", 5),
    TabInfo("explainability", 9, "Model Explainability", "6.9", 5),
    TabInfo("outcome", 10, "Outcome and Final Prediction", "6.10", 5),
)

TAB_ORDER: Final[tuple[str, ...]] = tuple(tab.id for tab in TABS)

TAB_BY_ID: Final[dict[str, TabInfo]] = {tab.id: tab for tab in TABS}

TITLES: Final[dict[str, str]] = {tab.id: tab.title for tab in TABS}
