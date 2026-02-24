import re
import json
import sys
import unicodedata
from pathlib import Path
import pandas as pd

EXCEL_PATH = Path(r"d:\___MeusScripts\NacCarona\PEDAGOGO.xlsx")
OUT_JSON = Path("pedagogo_mapping.json")
OUT_CSV = Path("pedagogo_sample.csv")
OUT_XLSX = Path("pedagogo_mapping.xlsx")
OUT_CLEAN_XLSX = Path("pedagogo_cleaned.xlsx")
OUT_CLEAN_CSV = Path("pedagogo_cleaned.csv")

EMAIL_RE = re.compile(r"^[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}$")
CPF_RE = re.compile(r"^\D*(\d)\D*(\d)\D*(\d)\D*(\d)\D*(\d)\D*(\d)\D*(\d)\D*(\d)\D*(\d)\D*(\d)\D*$")
PHONE_RE = re.compile(r"\d{8,14}")
YESNO = {"sim": True, "nao": False, "não": False, "s": True, "n": False, "yes": True, "no": False, "y": True, "n": False}

CITY_STATE_MAP = {
    "TAGUATINGA": ("Taguatinga", "DF"),
    "MOSSORO": ("Mossoró", "RN"),
    "MANAUS": ("Manaus", "AM"),
    "SUDOESTE": ("Brasília", "DF"),
    "NOVA PARNAMIRIM": ("Parnamirim", "RN"),
    "ASA NORTE": ("Brasília", "DF"),
    "RIO VERDE": ("Rio Verde", "GO"),
    "FORTALEZA FATIMA": ("Fortaleza", "CE"),
    "GOIANIA JARDIM AMERICA": ("Goiânia", "GO"),
    "RECIFE BOA VIAGEM": ("Recife", "PE"),
    "GOIANIA CIDADE JARDIM": ("Goiânia", "GO"),
    "RECIFE DERBY": ("Recife", "PE"),
    "NATAL PONTA NEGRA": ("Natal", "RN"),
    "NATAL MORRO BRANCO": ("Natal", "RN"),
    "PALMAS": ("Palmas", "TO"),
    "LAGO SUL": ("Brasília", "DF"),
    "ANAPOLIS": ("Anápolis", "GO"),
    "GOIANIA UNIVERSITARIO": ("Goiânia", "GO"),
    "AGUAS CLARAS": ("Brasília", "DF"),
    "BELEM": ("Belém", "PA"),
    "NATAL TIROL": ("Natal", "RN"),
    "BRASILIA": ("Brasília", "DF"),
    "CATALAO": ("Catalão", "GO"),
    "GUARA": ("Guará", "DF"),
    "CARUARU": ("Caruaru", "PE"),
    "SAO LUIS": ("São Luís", "MA"),
    "BOA VISTA": ("Boa Vista", "RR"),
    "FORTALEZA DIONISIO TORRES": ("Fortaleza", "CE"),
    "SOBRADINHO": ("Sobradinho", "DF"),
    "GAMA": ("Gama", "DF"),
    "CEILANDIA": ("Ceilândia", "DF"),
    "ASA SUL": ("Brasília", "DF"),
    "GOIANIA ELDORADO": ("Goiânia", "GO"),
    "TERESINA": ("Teresina", "PI"),
    "SORRISO": ("Sorriso", "MT"),
    "CUIABA": ("Cuiabá", "MT"),
    "ALTO DA GLORIA": ("Fortaleza", "CE"),
    "ALTIPLANO": ("Natal", "RN"),
    "NATAL ZONA NORTE": ("Natal", "RN"),
    "PLANALTINA": ("Planaltina", "DF"),
    "JOAO PESSOA": ("João Pessoa", "PB"),
    "PORTO VELHO": ("Porto Velho", "RO"),
    "BARRA DO GARAS": ("Barra do Garças", "MT"),
    "SINOP": ("Sinop", "MT"),
    "FORTALEZA SUL": ("Fortaleza", "CE"),
    "GOIANIA II": ("Goiânia", "GO"),
    "RONDONOPOLIS": ("Rondonópolis", "MT"),
    "ANANINDEUA": ("Ananindeua", "PA"),
    "SAO LUIS VINHAIS": ("São Luís", "MA"),
    "ARAGUAINA": ("Araguaína", "TO"),
    "PALMAS AURENY": ("Palmas", "TO"),
    "JUAZEIRO DO NORTE": ("Juazeiro do Norte", "CE"),
    "SENADOR CANEDO": ("Senador Canedo", "GO"),
    "SOBRAL": ("Sobral", "CE"),
    "VALPARAISO": ("Valparaíso de Goiás", "GO"),
    "GOIANIA ITUMBIARA": ("Itumbiara", "GO"),
    "FORTALEZA MEIRELES": ("Fortaleza", "CE"),
    "CAMPINA GRANDE": ("Campina Grande", "PB"),
    "NATAL CANDELARIA": ("Natal", "RN"),
    "FORMOSA": ("Formosa", "GO"),
    "MACAPA": ("Macapá", "AP"),
    "BRASILIA SAMAMBAIA": ("Brasília", "DF"),
    "GOIANIA SETOR OESTE": ("Goiânia", "GO"),
    "CAMPO NOVO DO PARECIS": ("Campo Novo do Parecis", "MT"),
    "PETROLINA": ("Petrolina", "PE"),
    "AVULSA": (pd.NA, pd.NA),
    "GOIANIA GARAVELO": ("Goiânia", "GO"),
}


def is_name_column(col_name: str) -> bool:
    return any(k in col_name.lower() for k in ["nome", "name"])


def clean_string_value(value, col_name: str):
    if pd.isna(value):
        return value
    text = str(value).strip()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    if text == "":
        return pd.NA
    if is_name_column(col_name):
        text = text.title()
    return text


def clean_email_value(value):
    if pd.isna(value):
        return value
    text = str(value).strip().lower()
    return text if text != "" else pd.NA


def clean_numeric_value(value):
    if pd.isna(value):
        return value
    normalized = str(value).strip().replace(",", ".")
    try:
        return float(normalized)
    except Exception:
        return pd.NA


def clean_cpf_value(value):
    if pd.isna(value):
        return value
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        return pd.NA
    return format_cpf(digits.zfill(11))


def clean_phone_value(value):
    if pd.isna(value):
        return value
    digits = re.sub(r"\D", "", str(value))
    return format_phone(digits)


def format_cpf(digits: str):
    if len(digits) != 11:
        return digits
    return f"{digits[:3]}.{digits[3:6]}.{digits[6:9]}-{digits[9:]}"


def format_phone(digits: str):
    if not digits:
        return pd.NA
    if len(digits) == 11:
        return f"({digits[:2]}) {digits[2:7]}-{digits[7:]}"
    if len(digits) == 10:
        return f"({digits[:2]}) {digits[2:6]}-{digits[6:]}"
    if len(digits) > 11:
        country = digits[:-11]
        ddd = digits[-11:-9]
        first = digits[-9:-4]
        last = digits[-4:]
        prefix = f"+{country} " if country else ""
        return f"{prefix}({ddd}) {first}-{last}"
    return digits


def clean_yesno_value(value):
    if pd.isna(value):
        return value
    normalized = str(value).strip().lower()
    return YESNO.get(normalized, pd.NA)


def normalize_lookup_key(value):
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ASCII", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9 ]+", "", text)
    return text.upper().strip()


def map_sede_location(value):
    key = normalize_lookup_key(value)
    return CITY_STATE_MAP.get(key, (pd.NA, pd.NA))


def clean_date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def clean_series(series: pd.Series, inferred: str, col_name: str) -> pd.Series:
    if inferred == "empty":
        return series
    if inferred == "date":
        return clean_date_series(series)
    if inferred == "string":
        return series.apply(lambda v: clean_string_value(v, col_name))
    if inferred == "email":
        return series.apply(clean_email_value)
    if inferred == "numeric":
        return series.apply(clean_numeric_value)
    if inferred == "cpf":
        return series.apply(clean_cpf_value)
    if inferred == "phone":
        return series.apply(clean_phone_value)
    if inferred == "yesno":
        return series.apply(clean_yesno_value)
    return series


def clean_dataframe(df: pd.DataFrame, mapping: list) -> pd.DataFrame:
    cleaned = df.copy()
    type_by_column = {item["original_name"]: item["inferred_type"] for item in mapping}
    for col in df.columns:
        inferred = type_by_column.get(col, "string")
        cleaned[col] = clean_series(cleaned[col], inferred, str(col))
    sede_mapped = cleaned["Sede"].apply(map_sede_location)
    cleaned["sede_estado"] = sede_mapped.apply(lambda v: v[1])
    cleaned["sede_cidade"] = sede_mapped.apply(lambda v: v[0])
    # bring the derived sede_* fields right after the original Sede column
    desired_order = ["sede_estado", "sede_cidade"]
    reordered = [col for col in cleaned.columns if col not in desired_order]
    try:
        idx = reordered.index("Sede") + 1
    except ValueError:
        idx = 0
    for col in desired_order:
        if col in reordered:
            reordered.remove(col)
        reordered.insert(idx, col)
        idx += 1
    cleaned = cleaned.loc[:, reordered]
    return cleaned


def normalize_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^0-9A-Za-z_]+", "", name)
    name = name.lower()
    if name == "":
        name = "col"
    return name


def infer_type(series: pd.Series):
    s = series.dropna().astype(str)
    if s.empty:
        return "empty"
    # try numeric
    try:
        pd.to_numeric(s.sample(min(len(s), 50)))
        numeric_frac = s.apply(lambda v: bool(re.match(r"^-?\d+(\.\d+)?$", v))).mean()
    except Exception:
        numeric_frac = 0
    # try datetime
    date_frac = 0
    try:
        parsed = pd.to_datetime(s.sample(min(len(s), 50)), dayfirst=True, errors="coerce")
        date_frac = parsed.notna().mean()
    except Exception:
        date_frac = 0
    # email
    email_frac = s.apply(lambda v: bool(EMAIL_RE.match(v))).mean()
    # cpf
    cpf_frac = s.apply(lambda v: bool(CPF_RE.match(v))).mean()
    # phone
    phone_frac = s.apply(lambda v: bool(PHONE_RE.search(re.sub(r"\D", "", v)))).mean()
    # yes/no
    yn_frac = s.str.strip().str.lower().isin(YESNO.keys()).mean()

    scores = {
        "numeric": numeric_frac,
        "date": date_frac,
        "email": email_frac,
        "cpf": cpf_frac,
        "phone": phone_frac,
        "yesno": yn_frac,
    }
    best = max(scores, key=scores.get)
    if scores[best] < 0.6:
        # fallback to string
        return "string"
    return best


def suggest_transforms(col_name: str, inferred: str):
    t = []
    if inferred == "string":
        t.append("trim_whitespace")
        t.append("normalize_unicode")
    if inferred == "numeric":
        t.append("remove_thousands_and_cast_float_or_int")
    if inferred == "date":
        t.append("parse_date_iso_or_dayfirst")
    if inferred == "email":
        t.append("lowercase")
        t.append("validate_email_format")
    if inferred == "cpf":
        t.append("remove_non_digits")
        t.append("zero_pad_or_validate_length(11)")
    if inferred == "phone":
        t.append("remove_non_digits")
        t.append("normalize_international_prefix")
    if inferred == "yesno":
        t.append("map_yes_no_to_boolean")
    # heuristics for name fields
    if any(k in col_name.lower() for k in ["nome", "name"]):
        t.append("title_case_names")
    return t


def analyze(path: Path):
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)
    try:
        df = pd.read_excel(path, sheet_name=0, nrows=500, engine="openpyxl")
    except Exception as e:
        print("Erro ao ler Excel:", e)
        sys.exit(1)
    # collect mapping
    mapping = []
    for col in df.columns:
        series = df[col]
        normalized = normalize_name(col)
        missing_rate = series.isna().mean()
        unique_count = series.nunique(dropna=True)
        samples = list(series.dropna().astype(str).head(10).unique())
        inferred = infer_type(series)
        transforms = suggest_transforms(col, inferred)
        mapping.append({
            "original_name": str(col),
            "normalized_name": normalized,
            "inferred_type": inferred,
            "missing_rate": float(missing_rate),
            "unique_count": int(unique_count),
            "sample_values": samples,
            "suggested_transforms": transforms,
        })
    # save outputs
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump({"source": str(path), "fields": mapping}, f, ensure_ascii=False, indent=2)
    # sample CSV
    df.head(200).to_csv(OUT_CSV, index=False)
    # export mapping to Excel for convenient inspection
    df_mapping = pd.DataFrame(mapping)
    for column in ("sample_values", "suggested_transforms"):
        df_mapping[column] = df_mapping[column].apply(lambda values: "; ".join(values))
    df_mapping.to_excel(OUT_XLSX, index=False)
    cleaned = clean_dataframe(df, mapping)
    temp_excel = OUT_CLEAN_XLSX.with_suffix(".tmp.xlsx")
    temp_csv = OUT_CLEAN_CSV.with_suffix(".tmp.csv")
    cleaned.to_excel(temp_excel, index=False)
    cleaned.to_csv(temp_csv, index=False)
    for temp, final in ((temp_excel, OUT_CLEAN_XLSX), (temp_csv, OUT_CLEAN_CSV)):
        try:
            if final.exists():
                final.unlink()
            temp.replace(final)
        except PermissionError:
            print(f"Não foi possível substituir {final}; verifique se está aberto e mova {temp.name} manualmente.")
    print(f"Mapping salvo em: {OUT_JSON}")
    print(f"Amostra salva em: {OUT_CSV}")
    print(f"Planilha de mapeamento salva em: {OUT_XLSX}")
    print(f"Planilha limpa salva em: {OUT_CLEAN_XLSX}")
    print(f"CSV limpo salvo em: {OUT_CLEAN_CSV}")


if __name__ == '__main__':
    analyze(EXCEL_PATH)
