#!/usr/bin/env python3
"""
Generate an LGA median-income choropleth SVG based on:
- SVG map URL: https://upload.wikimedia.org/wikipedia/commons/7/79/Australian_local_government_areas_2023.svg
- ABS tab-delimited text file: abs_lga_earnings.txt (the attachment you provided)
Outputs:
 - australia_lga_median_income_heatmap.svg
 - lga_median_income_mapping.json
"""

import re
import io
import json
import math
import statistics
import argparse
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from matplotlib import colors, colormaps
from PIL import Image

try:
    import cairosvg
except ImportError:  # handled later so PNG export can be optional
    cairosvg = None
from rapidfuzz import fuzz, process

# ---------- Config ----------
SVG_URL = "Australian_local_government_areas_2023.svg"  # remote fallback used if this file is absent
INPUT_TAB_FILE = "abs_lga_earnings.txt"   # the attached data saved locally with this name
OUTPUT_SVG = "australia_lga_median_income_heatmap.svg"
OUTPUT_JSON = "lga_median_income_mapping.json"
OUTPUT_PNG = "australia_lga_median_income_heatmap.png"
NUM_BINS = 7
COLOR_MAP = "viridis"
PNG_TRIM_PADDING = 12

# Provenance for source material.
REPO_URL = "https://github.com/Fuzzwah/lga-income-heatmap-australia"
SOURCE_SVG_URL = "https://upload.wikimedia.org/wikipedia/commons/7/79/Australian_local_government_areas_2023.svg"
SOURCE_DATA_URL = "https://www.abs.gov.au/statistics/labour/earnings-and-working-conditions/personal-income-australia/latest-release"

# Handle recent ABS LGA code/name changes that are not yet reflected in the
# published earnings extract. Map new 2023 codes to the legacy identifiers so
# we can still colour those shapes.
CODE_ALIASES = {
    "24700": "25250",  # Merri-bek (new code) -> Moreland (legacy code)
}

# Provide updated display names when we rely on a legacy record.
NAME_OVERRIDES = {
    "24700": "Merri-bek",
}

# Manually assign codes to SVG element ids when the geometry lacks metadata.
ELEMENT_CODE_OVERRIDES = {
    "path265": "35800",  # Paroo (Qld)
}
# ----------------------------

def read_abs_table(path):
    # Attempt to read a tab/whitespace delimited file with header rows.
    # We will try to find columns for LGA code and Median income.
    txt = open(path, "r", encoding="utf-8").read()

    # First try: treat the file as tab-separated, using the second header row for labels.
    # The first header row contains units ("Median", "Mean" etc.) while the second row holds
    # placeholder symbols ("$", "%", "ratio"). We combine them so that repeated
    # placeholders inherit the descriptive label, which lets us reliably locate the
    # "Median" income column even when pandas would otherwise create duplicate names.
    lines = txt.splitlines()
    df = None
    if len(lines) >= 2:
        header_units = lines[0].split("\t")
        header_names = lines[1].split("\t")

        def _combine_header(unit, name):
            unit = unit.strip()
            name = name.strip()
            if not name:
                return unit
            if name in {"$", "%", "ratio", "coef."}:
                return unit or name
            return name

        combined_headers = [_combine_header(u, n) for u, n in zip_longest(header_units, header_names, fillvalue="")]

        buffer = io.StringIO("\n".join(lines[1:]))  # keep the descriptive header row for pandas
        try:
            df = pd.read_csv(buffer, sep="\t", dtype=str)
            if len(df.columns) == len(combined_headers):
                df.columns = combined_headers
        except pd.errors.ParserError:
            df = None

    if df is None:
        # Fallback: normalize header separators and force into a CSV-like table using regex delimiter
        # This handles legacy extracts that may not be cleanly tab-separated.
        df = pd.read_csv(path, sep=r'\t+|\s{2,}', engine='python', dtype=str)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Common column names we expect: 'LGA', 'LGA NAME', 'Median' or 'Median income' or 'Median' in $.
    # Make them consistent
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc == 'lga' or lc.startswith('lga\t') or lc == 'lga code' or lc.startswith('lga code'):
            col_map[c] = 'LGA'
        elif 'name' in lc and 'lga' in lc:
            col_map[c] = 'LGA_NAME'
        elif lc == 'median' or lc.startswith('median'):
            col_map[c] = 'MEDIAN'
        elif lc == 'median income' or 'median' in lc and '$' in lc:
            col_map[c] = 'MEDIAN'
    df = df.rename(columns=col_map)
    if 'LGA' not in df.columns or 'MEDIAN' not in df.columns:
        # Try positional parsing: first two columns LGA code and LGA Name, then Median at position where header contains '$'
        # Fall back: try reading with fixed column positions from the attached file (LGA, LGA NAME, Earners, Median age of earners, Sum, Median, Mean...)
        # We'll attempt to parse using whitespace splitting per line.
        rows = []
        for i, line in enumerate(txt.splitlines()):
            if not line.strip(): continue
            # Skip possible header lines that are the strings "Australia", state names etc.
            parts = re.split(r'\t+|\s{2,}', line.strip())
            rows.append(parts)
        # find header row index (the one that contains 'LGA' and 'LGA NAME')
        header_idx = None
        for idx, parts in enumerate(rows[:10]):
            up = " ".join(parts).lower()
            if 'lga' in up and 'lga name' in up:
                header_idx = idx
                break
        if header_idx is None:
            # try first row as header
            header = rows[0]
            body = rows[1:]
        else:
            header = rows[header_idx]
            body = rows[header_idx+1:]
        # build DataFrame with header as columns, then try to locate median
        # pad rows
        max_len = max(len(r) for r in body)
        body2 = [r + ['']*(max_len-len(r)) for r in body]
        df2 = pd.DataFrame(body2, columns=[f"c{i}" for i in range(len(header))])
        # map header labels heuristically
        header_labels = [" ".join(h.split()) for h in header]
        # try to detect median column index
        median_idx = None
        lga_idx = None
        lga_name_idx = None
        for i, lab in enumerate(header_labels):
            ll = lab.lower()
            if ll == 'lga' or 'lga' == ll.split()[0]:
                lga_idx = i
            if 'lga name' in ll or ('name' in ll and 'lga' in ll):
                lga_name_idx = i
            if 'median' in ll and ('$' in ll or 'income' in ll or 'median' == ll.split()[0]):
                median_idx = i
        # fallback for median: often median is column 5 (0-based ~5) in provided file; here detect numeric-like column where values look like numbers with no commas.
        if median_idx is None:
            for i in range(df2.shape[1]):
                sample = df2.iloc[:20, i].astype(str).str.replace(',', '').str.strip()
                # detect numeric-ish majority
                numeric_count = sample.str.match(r'^\d+$').sum()
                if numeric_count > 5:
                    median_idx = i
                    break
        if lga_idx is None:
            lga_idx = 0
        if lga_name_idx is None:
            lga_name_idx = 1
        # build normalized DataFrame
        df = pd.DataFrame({
            'LGA': df2.iloc[:, lga_idx].astype(str).str.strip(),
            'LGA_NAME': df2.iloc[:, lga_name_idx].astype(str).str.strip(),
            'MEDIAN': df2.iloc[:, median_idx].astype(str).str.strip() if median_idx is not None else ''
        })
    # Clean MEDIAN values: remove commas, dollar signs, handle 'np' and missing
    df['MEDIAN_RAW'] = df['MEDIAN']
    def parse_median(x):
        if pd.isna(x): return None
        s = str(x).strip()
        if s == '' or s.lower() in ('np','na','-','n.a.'): return None
        s = s.replace('$','').replace(',','').strip()
        if s == '': return None
        # sometimes median column may contain non-integers or decimals; cast to int
        try:
            return int(round(float(s)))
        except:
            # if contains extra text, extract first number
            m = re.search(r'(\d{1,3}(?:,\d{3})*|\d+)', s)
            if m:
                return int(m.group(1).replace(',',''))
            return None
    df['MEDIAN_INT'] = df['MEDIAN_RAW'].apply(parse_median)
    # Normalize LGA codes to strings with no decimals
    def norm_code(x):
        if pd.isna(x): return None
        s = str(x).strip()
        # if contains non-digit and digits separated by spaces, extract leading digits
        m = re.match(r'^\s*(\d+)', s)
        if m:
            return m.group(1)
        # sometimes LGA field may be like '10050\tAlbury' combined - try to extract leading digits
        m2 = re.search(r'(\d{3,6})', s)
        if m2:
            return m2.group(1)
        return s
    df['LGA_CODE'] = df['LGA'].apply(norm_code)
    # If LGA_NAME empty, try to parse from original 'LGA' column where it contained code+name
    df['LGA_NAME'] = df['LGA_NAME'].where(df['LGA_NAME'].notna() & (df['LGA_NAME'] != ''), df['LGA'].apply(lambda x: " ".join(re.split(r'\s{2,}|\t+', str(x))[1:]).strip()))
    # Final selection: LGA_CODE, LGA_NAME, MEDIAN_INT
    tbl = df[['LGA_CODE','LGA_NAME','MEDIAN_INT','MEDIAN_RAW']].copy()
    tbl = tbl.rename(columns={'MEDIAN_INT':'MEDIAN'})
    tbl['MEDIAN'] = pd.to_numeric(tbl['MEDIAN'], errors='coerce').astype('Int64')
    return tbl

def fetch_svg(source):
    # Allow passing a local path (default) or remote URL.
    path_candidate = Path(source)
    if path_candidate.exists():
        return path_candidate.read_text(encoding="utf-8")

    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        headers = {
            "User-Agent": "LGAIncomeHeatmap/1.0 (+https://github.com/Fuzzwah/lga-income-heatmap-australia)"
        }
        r = requests.get(source, timeout=30, headers=headers)
        r.raise_for_status()
        return r.text

    raise FileNotFoundError(f"SVG source '{source}' not found locally and is not a valid URL")

def parse_svg_paths(svg_text):
    # Use BeautifulSoup to parse the SVG and gather candidate path/polygon elements
    soup = BeautifulSoup(svg_text, "xml")
    # Collect elements that represent LGAs: path, polygon, g (groups)
    candidates = []
    for tagname in ('path','polygon','rect','g','circle','polyline','ellipse'):
        for el in soup.find_all(tagname):
            # extract id and attributes
            eid = el.get('id') or el.get('inkscape:label') or el.get('data-name') or ''
            # gather attributes helpful for matching: id, gid, LGA_CODE21, GID, name, etc.
            attrs = dict(el.attrs)
            candidates.append({'element': el, 'tag': tagname, 'id': eid, 'attrs': attrs})
    return soup, candidates

def try_match_by_code(candidates, tbl):
    # Build dict of code->table median
    code_to_row = {}
    for _, row in tbl.iterrows():
        code = row['LGA_CODE']
        if code and code not in code_to_row:
            code_to_row[code] = {'name': row['LGA_NAME'], 'median': row['MEDIAN']}
    # Extend with aliases so new codes fall back to legacy identifiers.
    for new_code, legacy_code in CODE_ALIASES.items():
        if new_code not in code_to_row and legacy_code in code_to_row:
            alias_rec = code_to_row[legacy_code].copy()
            alias_rec['name'] = NAME_OVERRIDES.get(new_code, alias_rec['name'])
            code_to_row[new_code] = alias_rec
    matched = {}
    unmatched_candidates = set()
    for c in candidates:
        attrs = c['attrs']
        found_code = None
        element_id = c['element'].get('id')
        if element_id and element_id in ELEMENT_CODE_OVERRIDES:
            found_code = ELEMENT_CODE_OVERRIDES[element_id]
        # Check common attribute names (prefer explicit code attributes before generic ids)
        if not found_code:
            for key in ('LGA_CODE','LGA_CODE21','GID','LGA_PID','data-lga-code','lga','inkscape:label','id','ID'):
                if key in attrs:
                    val = str(attrs[key]).strip()
                    # extract leading digits
                    m = re.search(r'(\d{3,6})', val)
                    if m:
                        candidate = m.group(1)
                        # Ignore short ids like "path123"
                        if len(candidate) >= 4:
                            found_code = candidate
                            break
        if found_code and found_code in code_to_row:
            matched[c['element']] = (found_code, code_to_row[found_code])
        else:
            unmatched_candidates.add(c['element'])
    return matched, list(unmatched_candidates)

def fuzzy_match_names(candidates, tbl, already_matched_elements):
    # Create searchable name list from tbl for fuzzy matching
    name_records = {}
    for _, row in tbl.iterrows():
        code = row['LGA_CODE']
        name = str(row['LGA_NAME']).strip()
        if not name: continue
        name_normal = normalize_name(name)
        name_records[name_normal] = {'code': code, 'name': name, 'median': row['MEDIAN']}
    # Build list of normalized names for process.extractOne
    names_keys = list(name_records.keys())
    matches = {}
    unmatched = []
    for c in candidates:
        el = c['element']
        if el in already_matched_elements:
            continue
        # candidate label: try id attribute, title child, aria-label, data-name
        candidate_label = ''
        for k in ('inkscape:label','data-name','name','label','id'):
            v = c['attrs'].get(k)
            if v:
                candidate_label = str(v)
                break
        # if element has a child <title>, use that
        title_tag = el.find('title')
        if title_tag and title_tag.string:
            candidate_label = title_tag.string.strip()
        # fallback to class or 'label' attr
        if not candidate_label:
            # use some concatenation of attributes
            candidate_label = " ".join([str(v) for v in c['attrs'].values() if v])[:120]
        cand_norm = normalize_name(candidate_label)
        if not cand_norm:
            unmatched.append({'svg_id_or_label': c['id'] or candidate_label[:60], 'reason': 'no label', 'candidate_matches': []})
            continue
        best = process.extractOne(cand_norm, names_keys, scorer=fuzz.WRatio)
        if best:
            matched_name_norm, score, idx = best
            if score >= 80:
                rec = name_records[matched_name_norm]
                matches[el] = (rec['code'], rec)
            else:
                unmatched.append({'svg_id_or_label': c['id'] or candidate_label[:60], 'reason': 'low fuzzy match', 'candidate_matches': [best[0]]})
        else:
            unmatched.append({'svg_id_or_label': c['id'] or candidate_label[:60], 'reason': 'no candidate names', 'candidate_matches': []})
    return matches, unmatched

def normalize_name(s):
    if s is None:
        return ''
    s2 = s.lower()
    # remove diacritics roughly
    import unicodedata
    s2 = ''.join(c for c in unicodedata.normalize('NFD', s2) if unicodedata.category(c) != 'Mn')
    # replace tokens
    tokens_replace = {
        'city of ': '',
        'shire of ': '',
        'shire ': '',
        'city ': '',
        'council ': '',
        'regional ': '',
        '&': 'and',
        ' - ': ' ',
        '_': ' ',
        '.': '',
        ',': '',
        "'": ''
    }
    for a,b in tokens_replace.items():
        s2 = s2.replace(a, b)
    s2 = re.sub(r'\d+', ' ', s2)
    s2 = re.sub(r'[^a-z0-9\s]', '', s2)
    s2 = re.sub(r'\s+', ' ', s2).strip()
    return s2

def compute_bins(medians, k=7):
    arr = np.array([v for v in medians if v is not None])
    if len(arr) == 0:
        return []
    min_val = float(arr.min())
    max_val = float(arr.max())
    if min_val == max_val:
        single = int(math.floor(min_val))
        return [single] * (k + 1)
    linear_edges = np.linspace(min_val, max_val, k + 1)
    breaks = [int(round(edge)) for edge in linear_edges]
    for i in range(1, len(breaks)):
        if breaks[i] <= breaks[i-1]:
            breaks[i] = breaks[i-1] + 1
    breaks[0] = int(math.floor(min_val))
    breaks[-1] = int(math.ceil(max_val))
    if breaks[-1] <= breaks[-2]:
        breaks[-1] = breaks[-2] + 1
    return breaks

def get_viridis_colors(k):
    cmap = colormaps[COLOR_MAP].resampled(k)
    if hasattr(cmap, 'colors'):
        samples = cmap.colors
    else:
        steps = np.linspace(0, 1, k, endpoint=False) if k > 1 else [0.5]
        samples = [cmap(step) for step in steps]
    return [colors.to_hex(rgba) for rgba in samples]

def assign_bin(value, breaks):
    # breaks has length k+1
    if value is None:
        return None
    for i in range(len(breaks)-1):
        if value >= breaks[i] and value <= breaks[i+1]:
            return i+1
    # if larger than max
    if value > breaks[-1]:
        return len(breaks)-1
    return 1

def to_int_or_none(value):
    if value is None:
        return None
    if pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def trim_png_whitespace(image_path, padding=0):
    try:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            arr = np.array(rgb)
            mask = np.any(arr < 236, axis=2)
            if not np.any(mask):
                return
            ys, xs = np.where(mask)
            left = max(int(xs.min()) - padding, 0)
            right = min(int(xs.max()) + padding + 1, rgb.width)
            upper = max(int(ys.min()) - padding, 0)
            lower = min(int(ys.max()) + padding + 1, rgb.height)
            cropped = rgb.crop((left, upper, right, lower))
            cropped.save(image_path)
    except Exception as exc:
        print(f"Warning: failed to trim PNG whitespace ({exc}).")

def annotate_svg_and_write(soup, matched_map, name_matched_map, unmatched_list, tbl, breaks, colors_hex, output_svg_path, output_png_path=None):
    # Build mapping structure
    mapping = {}
    medians = []
    excluded_rows = 0
    for _, row in tbl.iterrows():
        if pd.isna(row['MEDIAN']):
            excluded_rows += 1
        else:
            medians.append(int(row['MEDIAN']))
    all_els = soup.find_all()

    # annotate matched_map elements (code matches)
    for el, (code, rec) in matched_map.items():
        median_val = to_int_or_none(rec['median'])
        if median_val is None:
            bin_index = None
            color_hex = "#f0f0f0"
        else:
            bin_index = assign_bin(median_val, breaks)
            color_hex = colors_hex[bin_index-1] if bin_index is not None else "#f0f0f0"
        # set attributes
        el['data-lga-code'] = str(code)
        el['data-lga-name'] = str(rec['name'])
        el['data-median-income'] = str(median_val) if median_val is not None else ""
        el['style'] = f"fill:{color_hex};stroke:#333;stroke-opacity:0.25;"
        # ensure title and desc
        existing_title = el.find('title')
        title_text = f"{rec['name']} — Median annual income: ${median_val}" if median_val is not None else f"{rec['name']} — Median annual income: n/a"
        if existing_title:
            existing_title.string = title_text
        else:
            t = soup.new_tag("title")
            t.string = title_text
            el.insert(0, t)
        desc = el.find('desc')
        desc_text = f"{code}|{median_val}|bin:{bin_index}"
        if desc:
            desc.string = desc_text
        else:
            d = soup.new_tag("desc")
            d.string = desc_text
            # insert after title
            if el.title:
                el.insert(1, d)
            else:
                el.insert(0, d)
        mapping[str(code)] = {
            'name': rec['name'],
            'median_income': median_val,
            'color_hex': color_hex,
            'bin_index': int(bin_index) if bin_index is not None else None,
            'svg_id': el.get('id') or ''
        }
    # annotate name_matched_map (fuzzy matches)
    for el, (code, rec) in name_matched_map.items():
        median_val = to_int_or_none(rec['median'])
        if median_val is None:
            bin_index = None
            color_hex = "#f0f0f0"
        else:
            bin_index = assign_bin(median_val, breaks)
            color_hex = colors_hex[bin_index-1] if bin_index is not None else "#f0f0f0"
        el['data-lga-code'] = str(code)
        el['data-lga-name'] = str(rec['name'])
        el['data-median-income'] = str(median_val) if median_val is not None else ""
        el['style'] = f"fill:{color_hex};stroke:#333;stroke-opacity:0.25;"
        # title + desc
        existing_title = el.find('title')
        title_text = f"{rec['name']} — Median annual income: ${median_val}" if median_val is not None else f"{rec['name']} — Median annual income: n/a"
        if existing_title:
            existing_title.string = title_text
        else:
            t = soup.new_tag("title")
            t.string = title_text
            el.insert(0, t)
        desc = el.find('desc')
        desc_text = f"{code}|{median_val}|bin:{bin_index}"
        if desc:
            desc.string = desc_text
        else:
            d = soup.new_tag("desc")
            d.string = desc_text
            if el.title:
                el.insert(1, d)
            else:
                el.insert(0, d)
        mapping[str(code)] = {
            'name': rec['name'],
            'median_income': median_val,
            'color_hex': color_hex,
            'bin_index': int(bin_index) if bin_index is not None else None,
            'svg_id': el.get('id') or ''
        }
    # Mark unmatched elements: light grey with dashed stroke
    matched_els = set(list(matched_map.keys()) + list(name_matched_map.keys()))
    unmatched_svg_details = []
    for el in all_els:
        if el.name in ('path','polygon','rect','g','circle','polyline','ellipse'):
            if el not in matched_els:
                el['data-unmatched'] = "true"
                prev_style = el.get('style', '')
                prev_style = re.sub(r'fill\s*:[^;]+;?', '', prev_style)
                add = "fill:none;stroke:#333;stroke-dasharray:3,3;stroke-opacity:0.25;pointer-events:none;"
                el['style'] = (prev_style + add).strip()
                unmatched_svg_details.append({'svg_id_or_label': el.get('id') or (el.get('inkscape:label') or '') or el.name, 'reason': 'no match', 'candidate_matches': []})

    # Build stats
    med_list = [v for v in medians if v is not None]
    stats = {}
    if med_list:
        stats['min'] = int(min(med_list))
        stats['max'] = int(max(med_list))
        stats['mean'] = int(round(statistics.mean(med_list)))
        stats['median'] = int(round(statistics.median(med_list)))
        stats['stddev'] = int(round(statistics.pstdev(med_list)))
    else:
        stats['min']=stats['max']=stats['mean']=stats['median']=stats['stddev']=None
    # bins info
    def format_bucket_label(lo_val, hi_val):
        def fmt(value):
            value_k = max(1, int(round(value / 1000.0)))
            return f"${value_k}k"
        return f"{fmt(lo_val)} - {fmt(hi_val)}"

    bins_info = []
    for i in range(len(breaks)-1):
        lo = int(breaks[i])
        hi = int(breaks[i+1])
        count = sum(1 for v in med_list if v is not None and v>=lo and v<=hi)
        bins_info.append({'min':lo,'max':hi,'count':int(count),'color':colors_hex[i],'label':format_bucket_label(lo, hi)})
    stats['bins'] = bins_info
    stats['excluded_rows'] = int(excluded_rows)

    # Prepare JSON artifact
    artifact = {
        'mapping': mapping,
        'stats': stats,
        'unmatched': unmatched_svg_details,
        'sources': {
            'svg': SOURCE_SVG_URL,
            'data': SOURCE_DATA_URL,
        },
        'outputs': {
            'svg': output_svg_path,
        },
    }
    if output_png_path:
        artifact['outputs']['png'] = output_png_path

    # Insert legend into SVG (top-right) using viewBox coordinates if present
    svg_tag = soup.find('svg')
    def _scale_dimension(value, factor):
        if not value:
            return value
        match = re.match(r'^([0-9.]+)([a-zA-Z%]*)$', str(value))
        if not match:
            try:
                return str(float(value) * factor)
            except Exception:
                return value
        magnitude = float(match.group(1)) * factor
        unit = match.group(2)
        return f"{magnitude}{unit}" if unit else str(int(round(magnitude)) if magnitude.is_integer() else magnitude)

    svg_width = svg_tag.get('width')
    svg_height = svg_tag.get('height')
    if svg_width:
        svg_tag['width'] = _scale_dimension(svg_width, 2)
    if svg_height:
        svg_tag['height'] = _scale_dimension(svg_height, 2)
    viewBox = svg_tag.get('viewBox')
    # create a legend group
    # Ensure a solid white background so browsers without default styling do not render grey.
    existing_background = svg_tag.find('rect', id='background') if svg_tag else None
    if svg_tag and existing_background is None:
        background = soup.new_tag('rect', id='background', x='0', y='0', width='100%', height='100%', fill='#ffffff')
        svg_tag.insert(0, background)

    legend = soup.new_tag('g', id='legend', **{'transform':'translate(10,20)', 'font-family':'sans-serif','font-size':'12','fill':'#222'})
    title = soup.new_tag('text', **{'x':'0','y':'0','font-weight':'bold'})
    title.string = "Median annual income AUD"
    legend.append(title)
    # swatches
    y = 18
    for b in reversed(bins_info):
        gsw = soup.new_tag('g')
        rect = soup.new_tag('rect', x=str(0), y=str(y-12), width="16", height="12", fill=b['color'], stroke="#333", **{'stroke-opacity':'0.25'})
        gsw.append(rect)
        label = soup.new_tag('text', x=str(22), y=str(y-2))
        label.string = f"{b['label']} ({b['count']})"
        gsw.append(label)
        legend.append(gsw)
        y += 18
    # caption
    caption = soup.new_tag('text', x='10', y='80%', **{'font-size':'10','fill':'#333'})
    caption_lines = [
        "Hacked up by: Fuzzwah",
        REPO_URL,
        "Source: Australian Bureau of Statistics Personal Income in Australia",
        SOURCE_DATA_URL,
        "Map base: Australian local government areas 2023 (Wikimedia Commons)",
        SOURCE_SVG_URL,
    ]
    for idx, line in enumerate(caption_lines):
        attrs = {'x': '10'}
        attrs['dy'] = '0' if idx == 0 else '1.2em'
        tspan = soup.new_tag('tspan', **attrs)
        tspan.string = line
        caption.append(tspan)
    # append legend and caption
    svg_tag.append(legend)
    svg_tag.append(caption)

    # embed small JS for hover and tooltip (compact)
    script_content = """
(function(){
  function $(s,root){return (root||document).querySelector(s);}
  function createTooltip(){
    var t=document.createElement('div');
    t.id='lga-tooltip';
    t.style.position='absolute';
    t.style.pointerEvents='none';
    t.style.background='rgba(255,255,255,0.95)';
    t.style.border='1px solid #333';
    t.style.padding='6px';
    t.style.font='12px/1.2 sans-serif';
    t.style.borderRadius='3px';
    t.style.boxShadow='0 1px 4px rgba(0,0,0,0.2)';
    t.style.display='none';
    document.body.appendChild(t);
    return t;
  }
  var tooltip = createTooltip();
  var svg = document.querySelector('svg');
  if(!svg) return;
  svg.addEventListener('mousemove', function(ev){
    if(tooltip.style.display==='none') return;
    tooltip.style.left = (ev.pageX+12)+'px';
    tooltip.style.top  = (ev.pageY+12)+'px';
  });
  var shapes = svg.querySelectorAll('[data-lga-code]');
  shapes.forEach(function(s){
    s.style.transition='stroke-width 0.08s, opacity 0.08s';
    s.addEventListener('mouseover', function(ev){
      s.style.strokeWidth='1.5';
      s.style.strokeOpacity='1';
      var name = s.getAttribute('data-lga-name') || s.id || 'LGA';
      var code = s.getAttribute('data-lga-code') || '';
      var median = s.getAttribute('data-median-income') || 'n/a';
      var bin = (s.getAttribute('data-median-income')) ? (s.getAttribute('style').match(/fill:([^;]+)/)||['',''])[1] : '';
    tooltip.innerHTML = '<strong>'+name+'</strong><br/>Code: '+code+'<br/>Median annual: $'+median;
      tooltip.style.display='block';
    });
    s.addEventListener('mouseout', function(ev){
      s.style.strokeWidth='';
      s.style.strokeOpacity='0.25';
      tooltip.style.display='none';
    });
    s.addEventListener('click', function(ev){
      var obj = {lga_code: s.getAttribute('data-lga-code'), lga_name: s.getAttribute('data-lga-name'), median_income: s.getAttribute('data-median-income'), color_hex: s.getAttribute('style').match(/fill:([^;]+)/)[1]};
      console.log(obj);
    });
  });
})();
"""
    script_tag = soup.new_tag('script')
    script_tag.string = script_content
    svg_tag.append(script_tag)

    # write SVG and optional PNG
    svg_content = str(soup)
    with open(output_svg_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)

    if output_png_path:
        if cairosvg is None:
            print("Warning: cairosvg not installed; skipping PNG export.")
        else:
            try:
                cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), write_to=output_png_path)
                trim_png_whitespace(output_png_path, PNG_TRIM_PADDING)
            except Exception as exc:
                print(f"Warning: failed to export PNG ({exc}).")

    # write JSON artifact
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as jf:
        json.dump(artifact, jf, indent=2)

    return artifact

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--svg-url', default=SVG_URL)
    parser.add_argument('--input', default=INPUT_TAB_FILE)
    parser.add_argument('--out-svg', default=OUTPUT_SVG)
    parser.add_argument('--out-json', default=OUTPUT_JSON)
    parser.add_argument('--out-png', default=OUTPUT_PNG, help="Path for PNG export (leave blank to disable).")
    args = parser.parse_args()

    tbl = read_abs_table(args.input)
    print("Rows parsed:", len(tbl))

    svg_text = fetch_svg(args.svg_url)

    soup, candidates = parse_svg_paths(svg_text)
    print("Candidate geometry-like elements found:", len(candidates))

    matched_by_code, unmatched_candidates = try_match_by_code(candidates, tbl)
    print("Exact matches found:", len(matched_by_code))
    candidate_structs = [c for c in candidates]
    name_matches, fuzzy_unmatched = fuzzy_match_names(candidate_structs, tbl, list(matched_by_code.keys()))
    print("Fuzzy matches found:", len(name_matches))

    # Build combined medians list for binning from tbl for rows with numeric medians
    medians = []
    for _, r in tbl.drop_duplicates(subset=['LGA_CODE']).iterrows():
        val = to_int_or_none(r['MEDIAN'])
        if val is not None:
            medians.append(val)

    breaks = compute_bins(medians, NUM_BINS)
    if not breaks:
        print("No medians available to compute bins. Exiting.")
        return
    colors_hex = get_viridis_colors(NUM_BINS)

    png_path = args.out_png or None
    print("Annotating SVG and exporting artifacts...")
    print("This will take a few minutes")
    artifact = annotate_svg_and_write(soup, matched_by_code, name_matches, fuzzy_unmatched, tbl, breaks, colors_hex, args.out_svg, png_path)

    # print brief summary
    outputs_line = f"\nSVG: {args.out_svg}\nJSON: {args.out_json}"
    if png_path:
        outputs_line += f"\nPNG: {png_path}"
    print("Outputs ->", outputs_line)
    print(
        "Counts -> exact matches:{exact} | fuzzy matches:{fuzzy} | unmatched shapes:{unmatched}".format(
            exact=len(matched_by_code),
            fuzzy=len(name_matches),
            unmatched=len(artifact['unmatched'])
        )
    )

if __name__ == "__main__":
    main()
