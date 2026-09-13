"""Label order and prompts preserved from the completed pilot."""

LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

PLANES = ["Sagittal", "Coronal", "Axial"]

DEFINITIONS = {
    "ACL": "high-grade partial (>50%) or complete anterior cruciate ligament tear; exclude isolated low-grade sprain or degeneration",
    "MCL": "high-grade acute partial or complete medial collateral ligament tear; exclude low-grade sprain or chronic healed injury",
    "Medial Meniscus": "definite medial meniscus tear with surface-reaching signal on at least two images or abnormal morphology; exclude isolated intrameniscal degeneration",
    "Lateral Meniscus": "definite lateral meniscus tear with surface-reaching signal on at least two images or abnormal morphology; exclude isolated intrameniscal degeneration",
    "Medial OA": "medial compartment cartilage loss of high grade (>50% thickness) over a moderate or large area (at least 1 cm)",
    "Lateral OA": "lateral compartment cartilage loss of high grade (>50% thickness) over a moderate or large area (at least 1 cm)",
    "PF OA": "patellofemoral cartilage loss of high grade (>50% thickness) over a moderate or large area (at least 1 cm)",
    "Effusion": "moderate or large joint effusion",
    "Synovitis": "synovitis with synovial thickening or inflammation",
    "Baker's": "moderate or large popliteal (Baker) cyst",
    "Contusion": "traumatic bone marrow edema (bone contusion) without a fracture line; exclude degenerative edema",
    "Fracture": "an acute fracture",
}
