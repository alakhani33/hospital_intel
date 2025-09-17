# caliber_utils/data_loader.py
import pandas as pd
import numpy as np
import re

# --------- helpers ---------
# def _normalize_text(s: pd.Series) -> pd.Series:
#     return (
#         s.astype("string")
#          .str.strip()
#          .str.replace(r"\s+", " ", regex=True)
#          .str.title()
#     )
CATEGORICAL_LOCK = {
    "FAC_NAME","COUNTY","OWNER","CITY","HSA","HFPA",
    "TYPE_CNTRL","TYPE_CARE","TYPE_HOSP","TEACH_RURL"
}

def _normalize_text(s: pd.Series) -> pd.Series:
    # Normalize strings, convert blanks/whitespace to NA, then fill with "Unknown"
    s = s.astype("string")
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.replace(r"^\s*$", pd.NA, regex=True)
    s = s.str.title()
    return s.fillna("Unknown")

def _canonize(name: str) -> str:
    """Upper-case and remove non-alnum to match header aliases robustly."""
    return re.sub(r"[^A-Z0-9]+", "", str(name).upper().strip())

def _rename_with_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map friendly/variant headers to canonical names our app expects.
    Matching uses canonical form (uppercase, no punctuation/space).
    """
    if df.empty:
        return df

    alias_map = {
        # Dates
        "BEG_DATE": [
            "BEGDATE","BEG_DATE","BEGINDATE","STARTDATE","START_DATE",
            "REPORTSTARTDATE","REPORTBEGDATE","PERIODSTART","PERIOD_BEGIN"
        ],
        "END_DATE": [
            "ENDDATE","END_DATE","THRUDATE","CLOSEDATE","REPORTENDDATE",
            "PERIODEND","PERIOD_END"
        ],

        # Identity / dimensions
        "FAC_NAME": ["FACNAME","FAC_NAME","FACILITY","FACILITYNAME","FACILITY_NAME","HOSPITALNAME","HOSPITAL_NAME","ORG_NAME","ORGNAME"],
        "COUNTY": ["COUNTY","COUNTYNAME","COUNTY_NM","COUNTYDESC","COUNTYDESCRIPTION","COUNTY(HCAI)","COUNTY_","COUNTY NAME"],
        "OWNER": ["OWNER","OWNERNAME","PARENT","SYSTEM","SYSTEMNAME","OWNER_NAME","ORG_OWNER","SYSTEM_OWNER"],
        "HSA": ["HSA"],
        "HFPA": ["HFPA"],
        "CITY": ["CITY","CITYNAME","CITY_NAME"],
        "TYPE_CNTRL": ["TYPECNTRL","TYPE_CNTRL","CONTROLTYPE","OWNERSHIP","TYPECONTROL"],
        "TYPE_CARE": ["TYPECARE","TYPE_CARE","CARETYPE"],
        "TYPE_HOSP": ["TYPEHOSP","TYPE_HOSP","HOSPTYPE","COMPARABILITY","COMPARABLEFLAG","COMPFLAG"],
        "TEACH_RURL": ["TEACHRURL","TEACH_RURL","TEACHINGRURAL","TEACHING_RURAL"],

        # Period length
        "DAY_PER": ["DAYPER","DAY_PER","DAYSINPERIOD","PERIODDAYS"],

        # Common awkward headers
        "NAT_BIRTHS": ["NATBIRTHS","NAT_BIRTHS","NAT_ BIRTHS"],
        "ACCTS_REC": [
            "ACCTSREC","ACCTS_REC","ACCTS_ REC",
            "ACCOUNTSRECEIVABLE","ACCOUNTS_RECEIVABLE","ACCTS RECEIVABLE"
        ],
    }

    canon_to_actual = {_canonize(c): c for c in df.columns}
    renames = {}

    for target, aliases in alias_map.items():
        found_actual = None
        for alias in aliases:
            alias_key = _canonize(alias)
            if alias_key in canon_to_actual:
                found_actual = canon_to_actual[alias_key]
                break
        if found_actual and target not in df.columns:
            renames[found_actual] = target

    if renames:
        df = df.rename(columns=renames)

    return df

def _ensure_county_column(df: pd.DataFrame) -> pd.DataFrame:
    """If COUNTY is missing, try to discover a county-like text column and rename it to COUNTY."""
    if "COUNTY" in df.columns:
        return df
    for c in df.columns:
        canon = _canonize(c)
        if "COUNTY" in canon and "COUNTRY" not in canon:
            if df[c].dtype == "object" or pd.api.types.is_string_dtype(df[c]):
                return df.rename(columns={c: "COUNTY"})
    for candidate in ["CNTY_NAME", "CO_NAME", "COUNTY_NM", "COUNTY NAME"]:
        if candidate in df.columns:
            return df.rename(columns={candidate: "COUNTY"})
    return df

# --------- public loaders ---------
def load_workbook(path: str):
    """
    Loads your Excel workbook.
    - Data sheet: main dataset
    - Glossary sheet: field definitions
    - Chart/Profile sheets: optional
    """
    xls = pd.ExcelFile(path)
    sheet_names = {s.strip().lower(): s for s in xls.sheet_names}

    def _read(name_guess, **kw):
        for key in sheet_names:
            if key.startswith(name_guess):
                return pd.read_excel(xls, sheet_name=sheet_names[key], **kw)
        return None

    data = _read("data")
    glossary = _read("glossary")
    charts = _read("chart")
    profile = _read("profile")

    if data is None:
        raise ValueError("Could not find a 'Data' worksheet in the workbook.")

    data.columns = [str(c).strip() for c in data.columns]
    data = _rename_with_aliases(data)
    data = _ensure_county_column(data)

    # Parse reporting period
    if "BEG_DATE" in data.columns:
        data["BEG_DATE"] = pd.to_datetime(data["BEG_DATE"], errors="coerce")
    if "END_DATE" in data.columns:
        data["END_DATE"] = pd.to_datetime(data["END_DATE"], errors="coerce")

    # YEAR convenience
    if "END_DATE" in data.columns:
        data["YEAR"] = pd.to_datetime(data["END_DATE"], errors="coerce").dt.year
    elif "BEG_DATE" in data.columns:
        data["YEAR"] = pd.to_datetime(data["BEG_DATE"], errors="coerce").dt.year
    else:
        data["YEAR"] = pd.NA

    # Normalize key categoricals for filters/peers
    for c in ["FAC_NAME","COUNTY","OWNER","CITY","HSA","HFPA","TYPE_CNTRL","TYPE_CARE","TYPE_HOSP","TEACH_RURL"]:
        if c in data.columns:
            data[c] = _normalize_text(data[c]).fillna("Unknown")

    # Numeric coercion where columns are mostly numeric
    # def try_numeric(col: pd.Series) -> pd.Series:
    #     if col.dtype == "object" or pd.api.types.is_string_dtype(col):
    #         cleaned = col.str.replace(",", "", regex=False)
    #         num = pd.to_numeric(cleaned, errors="coerce")
    #         if num.notna().mean() >= 0.5:
    #             return num
    #     return col
    def try_numeric(col: pd.Series, colname: str) -> pd.Series:
    # Never coerce known categoricals like COUNTY, OWNER, etc.
        if colname in CATEGORICAL_LOCK:
            return col.astype("string") if col.dtype != "string" else col
        # Only coerce if the column is mostly numeric
        if col.dtype == "object" or pd.api.types.is_string_dtype(col):
            cleaned = col.str.replace(",", "", regex=False)
            num = pd.to_numeric(cleaned, errors="coerce")
            if num.notna().mean() >= 0.5:
                return num
        return col
    # data = data.apply(try_numeric)
    for c in list(data.columns):
        data[c] = try_numeric(data[c], c)

    return data.copy(), glossary, charts, profile

def period_overlap_mask(df: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.Series:
    beg = pd.to_datetime(df.get("BEG_DATE"), errors="coerce")
    end = pd.to_datetime(df.get("END_DATE"), errors="coerce")
    return (end.notna() & beg.notna() & (end >= start_ts) & (beg <= end_ts))
