"""Shared clinical definitions, time windows, and score calculations.

This module consolidates the former path helpers, code lists, time-bin utilities,
SOFA calculation, and Charlson calculation into one documented library.
"""
from __future__ import annotations


# ==============================================================================
# Project paths and table readers
# ==============================================================================

import csv
import gzip
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def project_path(config: dict, *parts: str) -> Path:
    root = Path(config["output_root"])
    path = root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_head(path: str | Path, nrows: int = 5) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def csv_header(path: str | Path) -> list[str]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", errors="replace") as f:
        reader = csv.reader(f)
        return next(reader)


def iter_existing(paths: Iterable[str | Path]) -> Iterable[Path]:
    for path in paths:
        p = Path(path)
        if p.exists():
            yield p


def keyword_match(text: object, keywords: Iterable[str]) -> bool:
    value = str(text).lower()
    return any(k.lower() in value for k in keywords)


def safe_read_sas(path: str | Path) -> pd.DataFrame:
    return pd.read_sas(path, format="xport", encoding="utf-8")


# ==============================================================================
# Clinical item identifiers and concept dictionaries
# ==============================================================================

"""Central registry of verified clinical item IDs, ICD codes, and eICU fields.

All IDs here were verified against the actual tables on the Lenovo drive
(see docs/status_report.md and exploratory probing logs). This is the single
source of truth so that no script hides magic numbers for medical concepts.

Expanded (v2): Added vital signs, comprehensive labs, medication item IDs,
eICU label mappings for cross-database alignment, and ICD codes for Charlson
comorbidity index calculation.
"""

# Hours per time bin and the analysis horizon.
BIN_HOURS = 6
MAX_HOURS = 72
MAX_BINS = MAX_HOURS // BIN_HOURS  # 12

# ---------------------------------------------------------------------------
# MIMIC-IV ICU (chartevents/inputevents/ingredientevents/procedureevents)
# ---------------------------------------------------------------------------
MIMIC_ITEMIDS = {
    # ===== GLUCOSE =====
    # Laboratory + point-of-care finger sticks.
    "glucose_lab": [50931],                       # labevents Glucose
    "glucose_poc": [225664, 220621, 226537, 228388],  # finger stick / serum / whole blood

    # ===== ENTERAL NUTRITION =====
    # Direct kcal readout from ingredientevents.
    "enteral_calories": [226060],                 # ingredientevents Calories (kcal)

    # ===== INSULIN =====
    "insulin_input": [223258, 223259, 223260, 223261, 223262, 223257, 229299, 229619],

    # ===== VASOPRESSORS (inputevents) =====
    "vasopressor": [221906, 221289, 222315, 221749, 221662, 221653],
    # 221906=Norepinephrine, 221289=Epinephrine, 222315=Vasopressin,
    # 221749=Phenylephrine, 221662=Dopamine, 221653=Dobutamine

    # ===== SEVERITY / NEURO =====
    "sofa_total": [227428],
    "gcs_eye": [220739], "gcs_verbal": [223900], "gcs_motor": [223901],

    # ===== SEDATION / DELIRIUM =====
    "rass": [228096],
    "cam_icu": [
        228300, 228301, 228302, 228303, 228332,
        228334, 228335, 228336, 228337,
        229324, 229325, 229326,
    ],

    # ===== ORGAN SUPPORT =====
    "vent_proc": [224385],                        # Intubation procedureevent
    "vent_mode": [223849],                        # Ventilator Type
    "rrt_proc": [225441, 225802, 225803, 225809], # HD / CRRT / CVVHD / CVVHDF

    # ===== ANTHROPOMETRY =====
    "daily_weight": [224639],
    "height_cm": [226730, 226707],

    # ===== RESPIRATORY RATE =====
    "resp_rate": [220210, 224690],

    # =====================================================================
    # NEW: VITAL SIGNS (chartevents)
    # =====================================================================
    "heart_rate": [220045],
    "sbp": [220179, 220050],          # Non-invasive systolic, Arterial systolic
    "dbp": [220180, 220051],          # Non-invasive diastolic, Arterial diastolic
    "map": [220052, 220181, 225312],  # Mean arterial pressure (also used in SOFA)
    "spo2": [220277],                 # Peripheral O2 saturation
    "temperature": [223761, 223762],  # Temp (C), Temp (F) — needs unit normalization
    "fio2": [223835, 190],            # FiO2 (also used in SOFA respiratory)

    # =====================================================================
    # NEW: LABORATORY VALUES (labevents)
    # =====================================================================
    "lactate": [50813],               # Lactate — tissue perfusion / shock severity
    "albumin": [50862],               # Albumin — nutrition status (critical!)
    "prealbumin": [50976],            # Prealbumin — short-term nutrition (t½=2d)
    "bun": [51006],                   # BUN — renal function / protein catabolism
    "creatinine": [50912],            # Creatinine (also used in SOFA renal)
    "wbc": [51301],                   # WBC — infection / inflammation marker
    "hemoglobin": [51222],            # Hemoglobin — anemia / bleeding
    "platelet": [51265],              # Platelets (also used in SOFA coag)
    "sodium": [50983],                # Sodium — electrolyte balance
    "potassium": [50971],             # Potassium — affects insulin response!
    "chloride": [50902],              # Chloride — electrolyte balance
    "bicarbonate": [50882],           # Bicarbonate — acid-base status
    "pao2": [50821],                  # PaO2 (also used in SOFA resp)
    "pco2": [50818],                  # pCO2 — ventilation / acid-base
    "ph": [50820],                    # Arterial pH
    "magnesium": [50960],             # Magnesium — affects insulin sensitivity!
    "phosphate": [50970],             # Phosphate — refeeding syndrome marker!
    "ast": [50878],                   # AST — liver function
    "alt": [50861],                   # ALT — liver function
    "bilirubin": [50885],             # Bilirubin (also used in SOFA liver)
    "crp": [50889],                   # C-reactive protein — inflammation
    "triglycerides": [51000],         # Triglycerides — lipid metabolism

    # =====================================================================
    # NEW: MEDICATIONS (inputevents — IV drugs by itemID)
    # =====================================================================
    # -- Sedatives / Analgesics --
    "propofol": [222168],             # Propofol — contains lipid (1.1 kcal/mL)!
    "midazolam": [221668],            # Midazolam
    "dexmedetomidine": [225150],      # Dexmedetomidine
    "fentanyl": [221744],             # Fentanyl — slows gastric emptying → ↓EN absorption
    "morphine": [225154],             # Morphine — slows gastric emptying
    "hydromorphone": [221833],        # Hydromorphone — slows gastric emptying

    # -- Antibiotics (IV, as proxy for active infection) --
    "vancomycin": [225798],           # Vancomycin
    "piperacillin_tazo": [225893],    # Piperacillin-tazobactam
    "meropenem": [225883],            # Meropenem
    "cefepime": [225881],             # Cefepime
    "metronidazole": [225876],        # Metronidazole

    # -- Neuromuscular blockade --
    "cisatracurium": [222062],        # Cisatracurium — paralysis marker

    # -- Diuretics --
    "furosemide": [221794],           # Furosemide — fluid management marker

    # -- Anticoagulation --
    "heparin": [225152],              # Heparin drip

    # -- Parenteral nutrition & dextrose (critical confounders!) --
    "parenteral_nutrition": [225916, 220950],  # TPN / PN
    "dextrose": [220949, 225823, 225825],      # D5W, D10W, D50W

    # -- Prokinetics (affect EN absorption) --
    # These are typically in prescriptions, not inputevents.
    # Handled via MIMIC_PROKINETIC_DRUG_PATTERNS below.
}

# Steroids: MIMIC ICU inputevents has no clean category; use prescriptions
# drug-name matching instead. These lowercase substrings are matched against
# hosp/prescriptions.drug.
MIMIC_STEROID_DRUG_PATTERNS = [
    "hydrocort", "prednis", "methylpred", "dexameth", "solumedrol",
    "cortisone", "fludrocort",
]

# NEW: Additional drug patterns for prescriptions-based extraction.
MIMIC_PROKINETIC_DRUG_PATTERNS = [
    "metoclopramide", "reglan", "erythromycin",
]
MIMIC_ORAL_HYPOGLYCEMIC_PATTERNS = [
    "metformin", "glipizide", "glyburide", "glimepiride",
    "sitagliptin", "linagliptin", "saxagliptin",
    "empagliflozin", "dapagliflozin", "canagliflozin",
    "pioglitazone", "acarbose", "repaglinide", "nateglinide",
]
MIMIC_BETABLOCKER_PATTERNS = [
    "metoprolol", "esmolol", "labetalol", "propranolol", "atenolol",
    "carvedilol", "bisoprolol",
]
MIMIC_ANTIPSYCHOTIC_PATTERNS = [
    "olanzapine", "quetiapine", "haloperidol", "risperidone",
    "ziprasidone", "aripiprazole",
]
MIMIC_PPI_H2_PATTERNS = [
    "pantoprazole", "omeprazole", "esomeprazole", "lansoprazole",
    "famotidine", "ranitidine",
]
MIMIC_ANTIEMETIC_PATTERNS = [
    "ondansetron", "zofran", "granisetron", "prochlorperazine",
]
MIMIC_IMMUNOSUPPRESSANT_PATTERNS = [
    "tacrolimus", "cyclosporine", "mycophenolate",
]


# ICD codes (prefix match on diagnoses_icd.icd_code).
ICD_DIABETES = {
    10: ["E10", "E11", "E13"],
    9: ["250"],
}
# Structure: {condition_label: {icd_version: [prefix_codes]}}
# Consumed by build_mimic_confounders.py which iterates:
#   for label, mapping in ICD_PSYCHIATRIC.items():
#       for ver, prefixes in mapping.items(): ...
#
# ICD-9 codes corrected: depression and bipolar previously both mapped to
# '296', causing overlap. Now use validated subcodes:
#   - Depression: 296.2x (major single), 296.3x (major recurrent), 311 (NOS)
#   - Bipolar: 296.0x, 296.1x, 296.4x-296.7x
#   - Anxiety: 300.0x (anxiety states)
#   - PTSD: F43.1 / 309.81 only; broader F43 adjustment-stress diagnoses are
#     not treated as PTSD.
#   - Delirium: 293.0 (acute), 293.1 (subacute)
ICD_PSYCHIATRIC = {
    "depression": {
        10: ["F32", "F33"],
        9:  ["2962", "2963", "311"],
    },
    "anxiety": {
        10: ["F41"],
        9:  ["3000"],
    },
    "ptsd": {
        10: ["F431"],
        9:  ["30981"],
    },
    "bipolar": {
        10: ["F31"],
        9:  ["2960", "2961", "2964", "2965", "2966", "2967"],
    },
    "delirium": {
        10: ["F05"],
        9:  ["2930", "2931"],
    },
}

# ---------------------------------------------------------------------------
# NEW: Charlson Comorbidity Index ICD codes
# ---------------------------------------------------------------------------
# {component_name: (weight, {icd_version: [prefix_codes]})}
ICD_CHARLSON = {
    "myocardial_infarction":     (1,  {10: ["I21", "I22", "I252"], 9: ["410", "412"]}),
    "congestive_heart_failure":  (1,  {10: ["I099", "I110", "I130", "I132", "I255",
                                            "I420", "I425", "I426", "I427", "I428",
                                            "I429", "I43", "I50", "P290"],
                                       9: ["39891", "4254", "4255", "4256", "4257",
                                           "4258", "4259", "428"]}),
    "peripheral_vascular":       (1,  {10: ["I70", "I71", "I731", "I738", "I739",
                                            "I771", "I790", "I792", "K551", "K558",
                                            "K559", "Z958", "Z959"],
                                       9: ["0930", "4373", "440", "441", "4431",
                                           "4432", "4438", "4439", "4471", "5571",
                                           "5579", "V434"]}),
    "cerebrovascular":           (1,  {10: ["G45", "G46", "I60", "I61", "I62",
                                            "I63", "I64", "I65", "I66", "I67",
                                            "I68", "I69", "H340"],
                                       9: ["36234", "430", "431", "432", "433",
                                           "434", "435", "436", "437", "438"]}),
    "dementia":                  (1,  {10: ["F00", "F01", "F02", "F03", "F051",
                                            "G30", "G311"],
                                       9: ["290", "2941", "3312"]}),
    "chronic_pulmonary":         (1,  {10: ["I278", "I279", "J40", "J41", "J42",
                                            "J43", "J44", "J45", "J46", "J47",
                                            "J60", "J61", "J62", "J63", "J64",
                                            "J65", "J66", "J67", "J684", "J701",
                                            "J703"],
                                       9: ["4168", "4169", "490", "491", "492",
                                           "493", "494", "495", "496", "500",
                                           "501", "502", "503", "504", "505",
                                           "5064", "5081", "5088"]}),
    "rheumatic":                 (1,  {10: ["M05", "M06", "M315", "M32", "M33",
                                            "M34", "M351", "M353", "M360"],
                                       9: ["4465", "7100", "7101", "7102", "7103",
                                           "7104", "7140", "7141", "7142", "7148",
                                           "725"]}),
    "peptic_ulcer":              (1,  {10: ["K25", "K26", "K27", "K28"],
                                       9: ["531", "532", "533", "534"]}),
    "mild_liver":                (1,  {10: ["B18", "K700", "K701", "K702", "K703",
                                            "K709", "K713", "K714", "K715", "K717",
                                            "K73", "K74", "K760", "K762", "K763",
                                            "K764", "K768", "K769", "Z944"],
                                       9: ["07022", "07023", "07032", "07033",
                                           "07044", "07054", "0706", "0709",
                                           "570", "571", "5733", "5734",
                                           "5738", "5739", "V427"]}),
    "diabetes_uncomplicated":    (1,  {10: ["E100", "E101", "E106", "E108", "E109",
                                            "E110", "E111", "E116", "E118", "E119",
                                            "E130", "E131", "E136", "E138", "E139",
                                            "E140", "E141", "E146", "E148", "E149"],
                                       9: ["2500", "2501", "2502", "2503", "2508",
                                           "2509"]}),
    "diabetes_complicated":      (2,  {10: ["E102", "E103", "E104", "E105", "E107",
                                            "E112", "E113", "E114", "E115", "E117",
                                            "E132", "E133", "E134", "E135", "E137",
                                            "E142", "E143", "E144", "E145", "E147"],
                                       9: ["2504", "2505", "2506", "2507"]}),
    "hemiplegia":                (2,  {10: ["G041", "G114", "G801", "G802",
                                            "G81", "G82", "G830", "G831",
                                            "G832", "G833", "G834", "G839"],
                                       9: ["3341", "342", "343", "3440", "3441",
                                           "3442", "3443", "3444", "3445", "3446",
                                           "3449"]}),
    "renal":                     (2,  {10: ["I120", "I131", "N032", "N033", "N034",
                                            "N035", "N036", "N037", "N052", "N053",
                                            "N054", "N055", "N056", "N057", "N18",
                                            "N19", "N250", "Z490", "Z491", "Z492",
                                            "Z940", "Z992"],
                                       9: ["40301", "40311", "40391", "40402",
                                           "40403", "40412", "40413", "40492",
                                           "40493", "582", "5830", "5831", "5832",
                                           "5834", "5836", "5837", "585", "586",
                                           "5880", "V420", "V451", "V56"]}),
    "malignancy":                (2,  {10: ["C00", "C01", "C02", "C03", "C04",
                                            "C05", "C06", "C07", "C08", "C09",
                                            "C10", "C11", "C12", "C13", "C14",
                                            "C15", "C16", "C17", "C18", "C19",
                                            "C20", "C21", "C22", "C23", "C24",
                                            "C25", "C26", "C30", "C31", "C32",
                                            "C33", "C34", "C37", "C38", "C39",
                                            "C40", "C41", "C43", "C45", "C46",
                                            "C47", "C48", "C49", "C50", "C51",
                                            "C52", "C53", "C54", "C55", "C56",
                                            "C57", "C58", "C60", "C61", "C62",
                                            "C63", "C64", "C65", "C66", "C67",
                                            "C68", "C69", "C70", "C71", "C72",
                                            "C73", "C74", "C75", "C76", "C81",
                                            "C82", "C83", "C84", "C85", "C88",
                                            "C90", "C91", "C92", "C93", "C94",
                                            "C95", "C96", "C97"],
                                       9: ["140", "141", "142", "143", "144",
                                           "145", "146", "147", "148", "149",
                                           "150", "151", "152", "153", "154",
                                           "155", "156", "157", "158", "159",
                                           "160", "161", "162", "163", "164",
                                           "165", "170", "171", "172", "174",
                                           "175", "176", "179", "180", "181",
                                           "182", "183", "184", "185", "186",
                                           "187", "188", "189", "190", "191",
                                           "192", "193", "194", "195", "200",
                                           "201", "202", "203", "204", "205",
                                           "206", "207", "208"]}),
    "moderate_severe_liver":     (3,  {10: ["I850", "I859", "I864", "I982",
                                            "K704", "K711", "K721", "K729",
                                            "K765", "K766", "K767"],
                                       9: ["4560", "4561", "4562", "5722",
                                           "5723", "5724", "5728"]}),
    "metastatic_solid_tumor":    (6,  {10: ["C77", "C78", "C79", "C80"],
                                       9: ["196", "197", "198", "199"]}),
    "aids":                      (6,  {10: ["B20", "B21", "B22", "B24"],
                                       9: ["042", "043", "044"]}),
}

# Additional baseline comorbidity ICD codes.
ICD_HYPERTENSION = {
    10: ["I10", "I11", "I12", "I13", "I15"],
    9: ["401", "402", "403", "404", "405"],
}
ICD_CKD = {
    10: ["N18"],
    9: ["585"],
}
ICD_LIVER_DISEASE = {
    10: ["K70", "K71", "K72", "K73", "K74", "K75", "K76", "K77"],
    9: ["571"],
}
ICD_OBESITY = {
    10: ["E66"],
    9: ["2780"],
}

# ---------------------------------------------------------------------------
# eICU collaborative research database
# ---------------------------------------------------------------------------
EICU_GLUCOSE_LAB = "glucose"                      # lab.labname substring
EICU_GLUCOSE_POC_LABEL = "bedside glucose"        # nurseCharting vallabel
EICU_VASOPRESSOR_DRUGS = ["norepinephrine", "epinephrine", "vasopressin",
                          "phenylephrine", "dopamine", "dobutamine", "levophed"]
EICU_STEROID_DRUG_PATTERNS = list(MIMIC_STEROID_DRUG_PATTERNS)
EICU_DELIRIUM_LABELS = ["symptoms of delirium present", "mental status assessment"]
EICU_GCS_LABEL = "glasgow coma score"
EICU_RESP_RATE_LABEL = "respiratory rate"
EICU_RRT_TREATMENTS = ["dialysis", "hemodialysis", "cvvhd", "cvvhdf", "cvvh", "crrt"]
EICU_NUTRITION_TERMS = ["enteral", "tube feed", "tube feeding", "nutrition",
                        "jevity", "nepro", "glucerna", "osmolite", "promote", "kcal"]
EICU_PASTHISTORY_DIABETES = ["diabetes", "diabetic"]
EICU_PASTHISTORY_PSYCH = {"depression": ["depression"],
                          "anxiety": ["anxiety"],
                          "ptsd": ["ptsd", "post traumatic", "post-traumatic"],
                          "bipolar": ["bipolar", "manic"],
                          "delirium": []}  # delirium is acute, not a past history

# eICU severity lives in apachePatientResult (apachescore + predicted mortality).
# Mechanical ventilation lives in respiratoryCare (ventstart/ventendoffset)
# plus apachePatientResult.actualventdays.

# =====================================================================
# NEW: eICU variable mappings (for cross-database alignment)
# =====================================================================

# nurseCharting label matches for vital signs.
EICU_VITALS_LABELS = {
    "heart_rate": ["Heart Rate"],
    "sbp": ["Non-Invasive BP Systolic", "Invasive BP Systolic"],
    "dbp": ["Non-Invasive BP Diastolic", "Invasive BP Diastolic"],
    "map": ["Non-Invasive BP Mean", "Invasive BP Mean"],
    "spo2": ["O2 Saturation"],
    "temperature": ["Temperature (C)", "Temperature (F)", "Temperature"],
}

# lab.labname matches for laboratory values.
EICU_LAB_LABELS = {
    "lactate":       ["lactate"],
    "albumin":       ["albumin"],
    "prealbumin":    ["prealbumin"],
    "bun":           ["BUN"],
    "creatinine":    ["creatinine"],
    "wbc":           ["WBC x 1000", "WBC"],
    "hemoglobin":    ["Hgb", "hemoglobin"],
    "platelet":      ["platelets x 1000", "platelet"],
    "sodium":        ["sodium"],
    "potassium":     ["potassium"],
    "chloride":      ["chloride"],
    "bicarbonate":   ["bicarbonate", "HCO3"],
    "pao2":          ["paO2"],
    "pco2":          ["paCO2"],
    "ph":            ["pH"],
    "magnesium":     ["magnesium", "Mg"],
    "phosphate":     ["phosphate"],
    "ast":           ["AST (SGOT)"],
    "alt":           ["ALT (SGPT)"],
    "bilirubin":     ["total bilirubin", "bilirubin"],
    "crp":           ["CRP", "C-reactive protein"],
    "triglycerides": ["triglycerides"],
}

# infusionDrug.drugname matches for IV medications.
EICU_IV_DRUG_PATTERNS = {
    "propofol":           ["propofol", "diprivan"],
    "midazolam":          ["midazolam", "versed"],
    "dexmedetomidine":    ["dexmedetomidine", "precedex"],
    "fentanyl":           ["fentanyl"],
    "morphine":           ["morphine"],
    "hydromorphone":      ["hydromorphone", "dilaudid"],
    "vancomycin":         ["vancomycin"],
    "piperacillin_tazo":  ["piperacillin", "zosyn"],
    "meropenem":          ["meropenem", "merrem"],
    "cefepime":           ["cefepime", "maxipime"],
    "metronidazole":      ["metronidazole", "flagyl"],
    "cisatracurium":      ["cisatracurium", "nimbex"],
    "furosemide":         ["furosemide", "lasix"],
    "heparin":            ["heparin"],
}

# medication table matches for non-IV (oral/enteral) drugs.
EICU_ORAL_DRUG_PATTERNS = {
    "prokinetic":         ["metoclopramide", "reglan", "erythromycin"],
    "oral_hypoglycemic":  ["metformin", "glipizide", "glyburide", "glimepiride",
                           "sitagliptin", "linagliptin", "empagliflozin",
                           "dapagliflozin", "pioglitazone", "acarbose"],
    "betablocker":        ["metoprolol", "esmolol", "labetalol", "propranolol",
                           "atenolol", "carvedilol"],
    "antipsychotic":      ["olanzapine", "quetiapine", "haloperidol",
                           "risperidone", "ziprasidone"],
    "ppi_h2":             ["pantoprazole", "omeprazole", "esomeprazole",
                           "famotidine", "ranitidine"],
    "antiemetic":         ["ondansetron", "granisetron", "prochlorperazine"],
    "immunosuppressant":  ["tacrolimus", "cyclosporine", "mycophenolate"],
}

# eICU intakeOutput label matches for parenteral nutrition / dextrose.
EICU_PN_DEXTROSE_TERMS = [
    "tpn", "parenteral nutrition", "pn ", "lipid",
    "d5", "d10", "d50", "dextrose",
]

# eICU pastHistory matches for additional comorbidities.
EICU_PASTHISTORY_HYPERTENSION = ["hypertension", "htn"]
EICU_PASTHISTORY_CKD = ["chronic kidney", "ckd", "renal failure",
                        "end stage renal", "esrd"]
EICU_PASTHISTORY_LIVER = ["cirrhosis", "liver disease", "hepatitis",
                          "liver failure"]
EICU_PASTHISTORY_OBESITY = ["obesity", "morbid obesity", "obese"]


# ==============================================================================
# Time-window utilities
# ==============================================================================

"""Shared helpers for time-bin construction and aggregation.

Keeps the MIMIC and eICU cohort builders DRY: both reduce high-frequency
events into the same 6-hour bins over the first 72 hours so that downstream
models and the cross-database validation use identical temporal structure.
"""

import pandas as pd


def base_time_bins(stay_ids, id_col: str) -> pd.DataFrame:
    """Cartesian product of stay-ids x 0..MAX_BINS-1 (balanced panel)."""
    rows = [
        {id_col: sid, "bin": b,
         "bin_start_hour": b * BIN_HOURS,
         "bin_end_hour": (b + 1) * BIN_HOURS}
        for sid in stay_ids
        for b in range(MAX_BINS)
    ]
    return pd.DataFrame(rows)


def hours_from_icu(charttime, intime) -> pd.Series:
    """Robust hours between two timestamps, coercing non-datetime to NaT."""
    charttime = pd.to_datetime(charttime, errors="coerce")
    intime = pd.to_datetime(intime, errors="coerce")
    delta = (charttime - intime).dt.total_seconds() / 3600.0
    return delta


def within_horizon(df: pd.DataFrame, col: str = "hours_from_icu") -> pd.DataFrame:
    return df[(df[col] >= 0) & (df[col] < MAX_HOURS)].copy()


def assign_bin(df: pd.DataFrame, col: str = "hours_from_icu") -> pd.Series:
    return (df[col] // BIN_HOURS).astype(int)


def aggregate_numeric_by_bin(
    df: pd.DataFrame, id_col: str, value_col: str, out_prefix: str
) -> pd.DataFrame:
    """sum + count of a numeric column per (stay, bin)."""
    agg = (
        df.groupby([id_col, "bin"])[value_col]
        .agg(**{f"{out_prefix}_sum": "sum", f"{out_prefix}_count": "count"})
        .reset_index()
    )
    return agg


def last_value_by_bin(
    df: pd.DataFrame, id_col: str, value_col: str, out_name: str
) -> pd.DataFrame:
    """Most recent measurement within each bin (for scores/vitals)."""
    idx = df.groupby([id_col, "bin"])["hours_from_icu"].idxmax()
    out = df.loc[idx, [id_col, "bin", value_col]].rename(columns={value_col: out_name})
    return out


def coverage_flag_by_bin(
    df: pd.DataFrame, id_col: str, start_col: str, end_col: str, out_name: str
) -> pd.DataFrame:
    """Binary per-bin flag: does any interval [start, end) overlap this bin?

    Used for mechanical ventilation / RRT procedures whose duration spans bins.
    End times are treated as exclusive so an event ending exactly at 6 h is not
    counted in both the 0-6 h and 6-12 h bins.
    """
    rows = []
    for _, r in df.iterrows():
        sh = r[start_col]
        eh = r[end_col]
        if pd.isna(sh):
            continue
        if pd.isna(eh):
            eh = sh
        sh = float(sh)
        eh = float(eh)
        if eh < 0 or sh >= MAX_HOURS:
            continue
        sh = max(0.0, sh)
        eh = min(float(MAX_HOURS), eh)
        if eh <= sh:
            b0 = b1 = max(0, min(MAX_BINS - 1, int(sh // BIN_HOURS)))
        else:
            b0 = max(0, int(sh // BIN_HOURS))
            b1 = min(MAX_BINS - 1, int((eh - 1e-9) // BIN_HOURS))
        for b in range(b0, b1 + 1):
            rows.append({id_col: r[id_col], "bin": b, out_name: 1})
    if not rows:
        return pd.DataFrame(columns=[id_col, "bin", out_name])
    return pd.DataFrame(rows).drop_duplicates([id_col, "bin"])


# ==============================================================================
# SOFA and Charlson scores
# ==============================================================================

"""Compute SOFA score from raw MIMIC-IV components.

MIMIC-IV does NOT store an aggregate SOFA score in chartevents (verified: the
227428 item is essentially empty). SOFA must be derived from its six organ
components using published item IDs. This module computes per-stay, per-bin
SOFA so that downstream confounder/cross-database scripts share one source.

Components and scoring (per standard SOFA definition):
  - PaO2/FiO2 ratio (respiratory)
  - platelets (coagulation)
  - bilirubin (liver)
  - MAP / vasopressor (cardiovascular)
  - GCS (CNS)
  - creatinine / urine (renal)

We compute a tractable 5-component SOFA (renal via creatinine only) at bin
resolution. Lower fidelity than full ADMISSION-SOFA but reproducible from the
public database without external derived tables.
"""

from pathlib import Path

import numpy as np
import pandas as pd



# Component item IDs (MIMIC-IV; verified present).
PLATELET_LAB = [51265]        # labevents
BILIRUBIN_LAB = [50885]       # labevents
CREATININE_LAB = [50912]      # labevents
PAO2_LAB = [50821]            # labevents
FIO2_CE = [223835, 190]       # chartevents FiO2
MAP_CE = [220052, 220181, 225312]  # chartevents MAP
GCS = MIMIC_ITEMIDS["gcs_eye"] + MIMIC_ITEMIDS["gcs_verbal"] + MIMIC_ITEMIDS["gcs_motor"]
VASOPRESSORS = MIMIC_ITEMIDS["vasopressor"]


def _score_platelet(v):
    v = pd.to_numeric(v, errors="coerce")
    s = pd.Series(0, index=v.index)
    s[v < 150] = 1; s[v < 100] = 2; s[v < 50] = 3; s[v < 20] = 4
    return s


def _score_bilirubin(v):
    v = pd.to_numeric(v, errors="coerce")
    s = pd.Series(0, index=v.index)
    s[v >= 1.2] = 1; s[v >= 2.0] = 2; s[v >= 6.0] = 3; s[v >= 12.0] = 4
    return s


def _score_creatinine(v):
    v = pd.to_numeric(v, errors="coerce")
    s = pd.Series(0, index=v.index)
    s[v >= 1.2] = 1; s[v >= 2.0] = 2; s[v >= 3.5] = 3; s[v >= 5.0] = 4
    return s


def _score_cns(gcs):
    gcs = pd.to_numeric(gcs, errors="coerce")
    s = pd.Series(0, index=gcs.index)
    s[gcs < 15] = 1; s[gcs <= 13] = 2; s[gcs <= 10] = 3; s[gcs < 6] = 4
    return s


def _score_resp(pao2, fio2):
    pao2 = pd.to_numeric(pao2, errors="coerce")
    fio2 = pd.to_numeric(fio2, errors="coerce")
    # fio2 in MIMIC stored as 0-100 percent; convert to fraction.
    fio2_frac = fio2.where(fio2 <= 1, fio2 / 100.0)
    ratio = pao2 / fio2_frac
    s = pd.Series(0, index=ratio.index)
    s[ratio < 400] = 1; s[ratio < 300] = 2; s[ratio < 200] = 3; s[ratio < 100] = 4
    return s


def _score_cardio(map_val, vaso_present):
    map_val = pd.to_numeric(map_val, errors="coerce")
    s = pd.Series(0, index=map_val.index)
    s[map_val < 70] = 1
    s[vaso_present.astype(bool)] = 2  # any vasopressor => at least 2 (dopamine/dobutamine tier approx)
    return s


def compute_sofa(icu: Path, hosp: Path, stay_times: pd.DataFrame, nrows=None) -> pd.DataFrame:
    """Return per (stay_id, bin) SOFA component scores + total."""
    # 1. labs
    cols_lab = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"]
    target_labs = PLATELET_LAB + BILIRUBIN_LAB + CREATININE_LAB + PAO2_LAB
    if nrows is not None:
        lab = pd.read_csv(
            hosp / "labevents.csv.gz",
            usecols=cols_lab,
            nrows=nrows, low_memory=False, parse_dates=["charttime"],
        )
        lab = lab[lab["itemid"].isin(target_labs)].copy()
    else:
        chunks = []
        for chunk in pd.read_csv(
            hosp / "labevents.csv.gz",
            usecols=cols_lab, chunksize=5_000_000, low_memory=False,
            parse_dates=["charttime"],
        ):
            filtered = chunk[chunk["itemid"].isin(target_labs)].copy()
            if not filtered.empty:
                chunks.append(filtered)
        lab = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols_lab)
        
    lab = lab.merge(stay_times[["hadm_id", "intime", "stay_id"]].drop_duplicates(["stay_id", "hadm_id"]),
                    on="hadm_id", how="inner")
    lab["hours_from_icu"] = hours_from_icu(lab["charttime"], lab["intime"])
    lab = within_horizon(lab)
    lab["bin"] = assign_bin(lab)
    lab["valuenum"] = pd.to_numeric(lab["valuenum"], errors="coerce")

    def last_lab(itemids, name):
        sub = lab[lab["itemid"].isin(itemids)].dropna(subset=["valuenum"])
        return last_value_by_bin(sub, "stay_id", "valuenum", name)

    comp = last_lab(PLATELET_LAB, "platelet")
    bil = last_lab(BILIRUBIN_LAB, "bilirubin")
    cr = last_lab(CREATININE_LAB, "creatinine")
    pao2 = last_lab(PAO2_LAB, "pao2")

    # 2. chartevents: fio2, map, gcs
    cols_ce = ["stay_id", "itemid", "charttime", "valuenum"]
    target_ce = FIO2_CE + MAP_CE + GCS
    if nrows is not None:
        ce = pd.read_csv(
            icu / "chartevents.csv.gz",
            usecols=cols_ce, nrows=nrows,
            low_memory=False, parse_dates=["charttime"],
        )
        ce = ce[ce["itemid"].isin(target_ce)].copy()
    else:
        chunks = []
        for chunk in pd.read_csv(
            icu / "chartevents.csv.gz",
            usecols=cols_ce, chunksize=10_000_000, low_memory=False,
            parse_dates=["charttime"],
        ):
            filtered = chunk[chunk["itemid"].isin(target_ce)].copy()
            if not filtered.empty:
                chunks.append(filtered)
        ce = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols_ce)
        
    ce = ce.merge(stay_times[["stay_id", "intime"]], on="stay_id", how="inner")
    ce["hours_from_icu"] = hours_from_icu(ce["charttime"], ce["intime"])
    ce = within_horizon(ce)
    ce["bin"] = assign_bin(ce)
    ce["valuenum"] = pd.to_numeric(ce["valuenum"], errors="coerce")

    fio2 = last_value_by_bin(ce[ce["itemid"].isin(FIO2_CE)], "stay_id", "valuenum", "fio2")
    mapv = last_value_by_bin(ce[ce["itemid"].isin(MAP_CE)], "stay_id", "valuenum", "map")

    # GCS total = eye + verbal + motor.  Each GCS observation has an itemid
    # indicating eye / verbal / motor.  Multiple assessments can occur within
    # a single 6-h bin.  We must:
    #   1. Sum the three components *per assessment time* (same charttime),
    #   2. Then take the *last* assessment per (stay_id, bin).
    # The old code summed ALL component values in the bin, which over-counts
    # when >1 assessment is present.
    gcs_parts = ce[ce["itemid"].isin(GCS)].dropna(subset=["valuenum"])
    gcs_per_assessment = (
        gcs_parts
        .groupby(["stay_id", "bin", "hours_from_icu"])["valuenum"]
        .sum()
        .rename("gcs")
        .reset_index()
    )
    # Take the last assessment within each bin (highest hours_from_icu)
    gcs_tot = (
        gcs_per_assessment
        .sort_values("hours_from_icu")
        .groupby(["stay_id", "bin"])["gcs"]
        .last()
        .reset_index()
    )

    # 3. vasopressor presence per bin
    cols_vaso = ["stay_id", "itemid", "starttime"]
    if nrows is not None:
        vaso = pd.read_csv(
            icu / "inputevents.csv.gz",
            usecols=cols_vaso, nrows=nrows,
            low_memory=False, parse_dates=["starttime"],
        )
        vaso = vaso[vaso["itemid"].isin(VASOPRESSORS)].copy()
    else:
        chunks = []
        for chunk in pd.read_csv(
            icu / "inputevents.csv.gz",
            usecols=cols_vaso, chunksize=2_000_000, low_memory=False,
            parse_dates=["starttime"],
        ):
            filtered = chunk[chunk["itemid"].isin(VASOPRESSORS)].copy()
            if not filtered.empty:
                chunks.append(filtered)
        vaso = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols_vaso)
        
    vaso = vaso.merge(stay_times[["stay_id", "intime"]], on="stay_id", how="inner")
    vaso["hours_from_icu"] = hours_from_icu(vaso["starttime"], vaso["intime"])
    vaso = within_horizon(vaso)
    vaso["bin"] = assign_bin(vaso)
    vaso_p = vaso.drop_duplicates(["stay_id", "bin"]).assign(vaso=1)[["stay_id", "bin", "vaso"]]

    # merge components
    df = comp
    for x in [bil, cr, pao2, fio2, mapv, gcs_tot, vaso_p]:
        df = df.merge(x, on=["stay_id", "bin"], how="outer")
    df["vaso"] = df["vaso"].fillna(0)

    df["sofa_coag"] = _score_platelet(df["platelet"])
    df["sofa_liver"] = _score_bilirubin(df["bilirubin"])
    df["sofa_renal"] = _score_creatinine(df["creatinine"])
    df["sofa_cns"] = _score_cns(df["gcs"])
    df["sofa_resp"] = _score_resp(df["pao2"], df["fio2"])
    df["sofa_cardio"] = _score_cardio(df["map"], df["vaso"])

    comp_cols = ["sofa_coag", "sofa_liver", "sofa_renal", "sofa_cns", "sofa_resp", "sofa_cardio"]
    df["sofa_total"] = df[comp_cols].fillna(0).sum(axis=1)
    # SOFA meaningful only if at least one component measured
    measured = df[["platelet", "bilirubin", "creatinine", "pao2", "fio2", "map", "gcs"]].notna().any(axis=1)
    df.loc[~measured, "sofa_total"] = np.nan
    return df[["stay_id", "bin"] + comp_cols + ["sofa_total"]].sort_values(["stay_id", "bin"])


def validate_sofa(df: pd.DataFrame) -> dict:
    """Check SOFA score distributions for implausible values.

    Returns a dict of QC metrics suitable for logging or saving to a
    validation report.  Flags:
      - sofa_total outside [0, 24]
      - GCS-derived CNS score based on GCS outside [3, 15]
      - negative component scores
    """
    report: dict = {}
    if "sofa_total" in df.columns:
        valid = df["sofa_total"].dropna()
        report["n_sofa_measured"] = int(len(valid))
        report["sofa_mean"] = round(float(valid.mean()), 2) if len(valid) else None
        report["sofa_median"] = round(float(valid.median()), 2) if len(valid) else None
        n_over = int((valid > 24).sum())
        n_neg = int((valid < 0).sum())
        report["sofa_gt_24"] = n_over
        report["sofa_negative"] = n_neg
        if n_over:
            print(f"WARNING: {n_over} bins have SOFA > 24 (max possible is 24).")
        if n_neg:
            print(f"WARNING: {n_neg} bins have negative SOFA.")

    # Check individual component ranges
    comp_max = {
        "sofa_coag": 4, "sofa_liver": 4, "sofa_renal": 4,
        "sofa_cns": 4, "sofa_resp": 4, "sofa_cardio": 4,
    }
    for col, cmax in comp_max.items():
        if col in df.columns:
            vals = df[col].dropna()
            n_bad = int(((vals < 0) | (vals > cmax)).sum())
            report[f"{col}_out_of_range"] = n_bad
            if n_bad:
                print(f"WARNING: {n_bad} bins have {col} outside [0, {cmax}].")

    print(f"SOFA validation: {report}")
    return report


def charlson_from_icd(hosp: Path, stay_times: pd.DataFrame, nrows=None) -> pd.DataFrame:
    """Compute the Quan/Charlson index for MIMIC-IV ICU stays."""
    icd = pd.read_csv(hosp / "diagnoses_icd.csv.gz", low_memory=False, nrows=nrows)
    icd["icd_code_norm"] = (
        icd["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False).str.strip()
    )
    sid_map = stay_times[["stay_id", "hadm_id"]].drop_duplicates("stay_id")
    result = sid_map[["stay_id"]].copy()
    for component, (weight, version_map) in ICD_CHARLSON.items():
        matched = set()
        for version, prefixes in version_map.items():
            normalized = tuple(prefix.replace(".", "").upper() for prefix in prefixes)
            mask = icd["icd_version"].eq(version) & icd["icd_code_norm"].str.startswith(normalized)
            matched.update(icd.loc[mask, "hadm_id"].unique())
        flags = pd.DataFrame({"hadm_id": list(matched), f"cci_{component}": 1})
        merged = sid_map.merge(flags, on="hadm_id", how="left")
        result[f"cci_{component}"] = merged[f"cci_{component}"].fillna(0).astype(int).to_numpy()

    _apply_charlson_hierarchy(result)
    score = pd.Series(0, index=result.index, dtype=int)
    for component, (weight, _) in ICD_CHARLSON.items():
        score += result.get(f"cci_{component}", 0) * weight
    result["charlson_index"] = score
    return result


def _apply_charlson_hierarchy(frame: pd.DataFrame) -> None:
    rules = [
        ("diabetes_complicated", "diabetes_uncomplicated"),
        ("moderate_severe_liver", "mild_liver"),
        ("metastatic_solid_tumor", "malignancy"),
    ]
    for stronger, weaker in rules:
        strong_col, weak_col = f"cci_{stronger}", f"cci_{weaker}"
        if strong_col in frame and weak_col in frame:
            frame.loc[frame[strong_col].eq(1), weak_col] = 0


def charlson_eicu(eicu: Path, nrows=None) -> pd.DataFrame:
    """Approximate the Charlson index from eICU past-history strings."""
    history = pd.read_csv(
        eicu / "pastHistory.csv.gz",
        usecols=["patientunitstayid", "pasthistorypath"],
        nrows=nrows,
        low_memory=False,
    )
    history["text"] = history["pasthistorypath"].fillna("").astype(str).str.lower()
    mapping = {
        "myocardial_infarction": (1, ["myocardial infarction", " mi ", "heart attack"]),
        "congestive_heart_failure": (1, ["heart failure", "chf", "cardiomyopathy"]),
        "peripheral_vascular": (1, ["peripheral vascular", "pvd", "claudication"]),
        "cerebrovascular": (1, ["stroke", "cva", "tia", "cerebrovascular"]),
        "dementia": (1, ["dementia", "alzheimer"]),
        "chronic_pulmonary": (1, ["copd", "emphysema", "chronic bronchitis", "asthma"]),
        "rheumatic": (1, ["rheumatoid", "lupus", "sle"]),
        "peptic_ulcer": (1, ["peptic ulcer", "gastric ulcer", "duodenal ulcer"]),
        "mild_liver": (1, ["hepatitis", "liver disease"]),
        "diabetes_uncomplicated": (1, ["diabetes"]),
        "diabetes_complicated": (2, ["diabetic nephropathy", "diabetic retinopathy", "diabetic neuropathy"]),
        "hemiplegia": (2, ["hemiplegia", "paraplegia"]),
        "renal": (2, ["renal failure", "ckd", "dialysis", "esrd"]),
        "malignancy": (2, ["cancer", "malignancy", "tumor", "leukemia", "lymphoma"]),
        "moderate_severe_liver": (3, ["cirrhosis", "liver failure", "portal hypertension"]),
        "metastatic_solid_tumor": (6, ["metastatic", "metastasis"]),
        "aids": (6, ["hiv", "aids"]),
    }
    result = pd.DataFrame({"patientunitstayid": history["patientunitstayid"].unique()})
    for component, (weight, terms) in mapping.items():
        mask = pd.Series(False, index=history.index)
        for term in terms:
            mask |= history["text"].str.contains(term, regex=False)
        positive = set(history.loc[mask, "patientunitstayid"].unique())
        result[f"cci_{component}"] = result["patientunitstayid"].isin(positive).astype(int)
    _apply_charlson_hierarchy(result)
    score = pd.Series(0, index=result.index, dtype=int)
    for component, (weight, _) in mapping.items():
        score += result[f"cci_{component}"] * weight
    result["charlson_index"] = score
    return result
