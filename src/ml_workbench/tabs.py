from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class TabInfo:
    id: str
    number: int
    title: str
    subtitle: str
    design_section: str
    implemented_phase: int


TABS: Final[tuple[TabInfo, ...]] = (
    TabInfo("data", 1, "Data Insertion", "Bring in your data", "6.1", 2),
    TabInfo("cleaning", 2, "Data Cleaning", "Fix gaps and duplicates", "6.2", 2),
    TabInfo("preprocessing", 3, "Data Preprocessing", "Prepare features for the model", "6.3", 3),
    TabInfo("eda", 4, "Exploratory Data Analysis", "Understand patterns in your data", "6.4", 3),
    TabInfo("modelling", 5, "Modelling", "Pick an algorithm and strategy", "6.5", 4),
    TabInfo("training", 6, "Training", "Fit and evaluate the model", "6.6", 4),
    TabInfo("prediction", 7, "Prediction", "Score new rows", "6.7", 4),
    TabInfo("error_analysis", 8, "Error Analysis", "See where the model fails", "6.8", 5),
    TabInfo(
        "explainability", 9, "Model Explainability", "Understand why the model predicts", "6.9", 5
    ),
    TabInfo("outcome", 10, "Outcome and Final Prediction", "Review results and export", "6.10", 5),
)

TAB_ORDER: Final[tuple[str, ...]] = tuple(tab.id for tab in TABS)

TAB_BY_ID: Final[dict[str, TabInfo]] = {tab.id: tab for tab in TABS}

TITLES: Final[dict[str, str]] = {tab.id: tab.title for tab in TABS}
