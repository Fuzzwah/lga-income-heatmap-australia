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

import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from matplotlib import cm, colors
from rapidfuzz import fuzz, process

# ---------- Config ----------
SVG_URL = "https://upload.wikimedia.org/wikipedia/commons/7/79/Australian_local_government_areas_2023.svg"
INPUT_TAB_FILE = "abs_lga_earnings.txt"   # the attached data saved locally with this name
OUTPUT_SVG = "australia_lga_median_income_heatmap.svg"
OUTPUT_JSON = "lga_median_income_mapping.json"
NUM_BINS = 7
COLOR_MAP = "viridis"
# ----------------------------

def read_abs_table(path):
    # Attempt to read a tab/whitespace delimited file with header rows.
    # We will try to find columns for LGA code and Median income.
    txt = open(path, "r", encoding="utf-8").read()

    # Normalize header separators and force into a CSV-like table
    # The provided file is tab/whitespace separated with header including "LGA" "LGA NAME" "Median"
    # We'll use pandas with delim_whitespace and try to find the median column.
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
    return tbl

def fetch_svg(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text

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
    matched = {}
    unmatched_candidates = set()
    for c in candidates:
        attrs = c['attrs']
        found_code = None
        # Check common attribute names
        for key in ('LGA_CODE','LGA_CODE21','GID','LGA_PID','id','ID','data-lga-code','lga'):
            if key in attrs:
                val = str(attrs[key]).strip()
                # extract leading digits
                m = re.search(r'(\d{3,6})', val)
                if m:
                    found_code = m.group(1)
                    break
                else:
                    # sometimes ID is like 'australia-10050' - extract digits
                    m2 = re.search(r'(\d{3,6})', val)
                    if m2:
                        found_code = m2.group(1)
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
        for k in ('id','inkscape:label','data-name','name','label'):
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
    s2 = re.sub(r'[^a-z0-9\s]', '', s2)
    s2 = re.sub(r'\s+', ' ', s2).strip()
    return s2

def compute_bins(medians, k=7):
    arr = np.array([v for v in medians if v is not None])
    if len(arr) == 0:
        return []
    # quantile classification into k bins
    quantiles = np.linspace(0, 1, k+1)
    breaks = np.quantile(arr, quantiles)
    # round to ints and ensure increasing
    breaks = [int(round(float(b))) for b in breaks]
    for i in range(1, len(breaks)):
        if breaks[i] <= breaks[i-1]:
            breaks[i] = breaks[i-1] + 1
    return breaks

def get_viridis_colors(k):
    cmap = cm.get_cmap(COLOR_MAP, k)
    hexes = []
    for i in range(k):
        rgba = cmap(i)
        hexes.append(colors.to_hex(rgba))
    return hexes

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

def annotate_svg_and_write(soup, matched_map, name_matched_map, unmatched_list, tbl, breaks, colors_hex, output_svg_path):
    # Build mapping structure
    mapping = {}
    medians = []
    excluded_rows = 0
    for _, row in tbl.iterrows():
        if row['MEDIAN'] is None:
            excluded_rows += 1
        else:
            medians.append(int(row['MEDIAN']))
    # annotate matched_map elements (code matches)
    for el, (code, rec) in matched_map.items():
        median = rec['median']
        if median is None:
            bin_index = None
            color_hex = "#f0f0f0"
        else:
            bin_index = assign_bin(median, breaks)
            color_hex = colors_hex[bin_index-1] if bin_index is not None else "#f0f0f0"
        # set attributes
        el['data-lga-code'] = str(code)
        el['data-lga-name'] = str(rec['name'])
        el['data-median-income'] = str(median) if median is not None else ""
        el['style'] = f"fill:{color_hex};stroke:#333;stroke-opacity:0.25;"
        # ensure title and desc
        existing_title = el.find('title')
        title_text = f"{rec['name']} — Median weekly income: ${median}" if median is not None else f"{rec['name']} — Median weekly income: n/a"
        if existing_title:
            existing_title.string = title_text
        else:
            t = soup.new_tag("title")
            t.string = title_text
            el.insert(0, t)
        desc = el.find('desc')
        desc_text = f"{code}|{median}|bin:{bin_index}"
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
            'median_income': int(rec['median']) if rec['median'] is not None else None,
            'color_hex': color_hex,
            'bin_index': int(bin_index) if bin_index is not None else None,
            'svg_id': el.get('id') or ''
        }
    # annotate name_matched_map (fuzzy matches)
    for el, (code, rec) in name_matched_map.items():
        median = rec['median']
        if median is None:
            bin_index = None
            color_hex = "#f0f0f0"
        else:
            bin_index = assign_bin(median, breaks)
            color_hex = colors_hex[bin_index-1] if bin_index is not None else "#f0f0f0"
        el['data-lga-code'] = str(code)
        el['data-lga-name'] = str(rec['name'])
        el['data-median-income'] = str(median) if median is not None else ""
        el['style'] = f"fill:{color_hex};stroke:#333;stroke-opacity:0.25;"
        # title + desc
        existing_title = el.find('title')
        title_text = f"{rec['name']} — Median weekly income: ${median}" if median is not None else f"{rec['name']} — Median weekly income: n/a"
        if existing_title:
            existing_title.string = title_text
        else:
            t = soup.new_tag("title")
            t.string = title_text
            el.insert(0, t)
        desc = el.find('desc')
        desc_text = f"{code}|{median}|bin:{bin_index}"
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
            'median_income': int(rec['median']) if rec['median'] is not None else None,
            'color_hex': color_hex,
            'bin_index': int(bin_index) if bin_index is not None else None,
            'svg_id': el.get('id') or ''
        }
    # Mark unmatched elements: light grey with dashed stroke
    all_els = soup.find_all()
    matched_els = set(list(matched_map.keys()) + list(name_matched_map.keys()))
    unmatched_svg_details = []
    for el in all_els:
        if el.name in ('path','polygon','rect','g','circle','polyline','ellipse'):
            if el not in matched_els:
                # skip decorative elements (no geometry)
                el['data-unmatched'] = "true"
                # ensure fill gray and dashed border
                # preserve existing style but append
                prev_style = el.get('style','')
                add = f"fill:#f0f0f0;stroke:#333;stroke-dasharray:3,3;stroke-opacity:0.25;"
                el['style'] = prev_style + add
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
    bins_info = []
    for i in range(len(breaks)-1):
        lo = int(breaks[i])
        hi = int(breaks[i+1])
        count = sum(1 for v in med_list if v is not None and v>=lo and v<=hi)
        bins_info.append({'min':lo,'max':hi,'count':int(count),'color':colors_hex[i]})
    stats['bins'] = bins_info
    stats['excluded_rows'] = int(excluded_rows)

    # Prepare JSON artifact
    artifact = {
        'mapping': mapping,
        'stats': stats,
        'unmatched': unmatched_svg_details
    }

    # Insert legend into SVG (top-right) using viewBox coordinates if present
    svg_tag = soup.find('svg')
    viewBox = svg_tag.get('viewBox')
    # create a legend group
    legend = soup.new_tag('g', id='legend', **{'transform':'translate(80,20)', 'font-family':'sans-serif','font-size':'12','fill':'#222'})
    title = soup.new_tag('text', **{'x':'0','y':'0','font-weight':'bold'})
    title.string = "Median weekly income AUD"
    legend.append(title)
    # swatches
    y = 18
    for i, b in enumerate(bins_info):
        gsw = soup.new_tag('g')
        rect = soup.new_tag('rect', x=str(0), y=str(y-12), width="16", height="12", fill=b['color'], stroke="#333", **{'stroke-opacity':'0.25'})
        gsw.append(rect)
        label = soup.new_tag('text', x=str(22), y=str(y-2))
        label.string = f"${b['min']}–${b['max']} ({b['count']})"
        gsw.append(label)
        legend.append(gsw)
        y += 18
    # caption
    caption = soup.new_tag('text', x='10', y=str(int(svg_tag.get('height') or 20)+20 if svg_tag.get('height') else '95%'), **{'font-size':'10','fill':'#333'})
    caption.string = "Source: ABS median income per LGA; map base: Wikimedia Commons"
    # append legend and caption
    svg_tag.insert(0, legend)
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
      tooltip.innerHTML = '<strong>'+name+'</strong><br/>Code: '+code+'<br/>Median weekly: $'+median;
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

    # write SVG
    with open(output_svg_path, 'w', encoding='utf-8') as f:
        f.write(str(soup))

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
    args = parser.parse_args()

    print("Reading ABS table...", args.input)
    tbl = read_abs_table(args.input)
    print("Rows parsed:", len(tbl))

    print("Fetching SVG...")
    svg_text = fetch_svg(args.svg_url)

    print("Parsing SVG elements...")
    soup, candidates = parse_svg_paths(svg_text)
    print("Candidate geometry-like elements found:", len(candidates))

    print("Trying exact code matches...")
    matched_by_code, unmatched_candidates = try_match_by_code(candidates, tbl)
    print("Exact matches found:", len(matched_by_code))
    # Prepare candidate list structure for fuzzy matching
    candidate_structs = [c for c in candidates]
    print("Fuzzy-matching remaining shapes by name...")
    name_matches, fuzzy_unmatched = fuzzy_match_names(candidate_structs, tbl, list(matched_by_code.keys()))
    print("Fuzzy matches found:", len(name_matches))

    # Build combined medians list for binning from tbl for rows with numeric medians
    medians = [int(r['MEDIAN']) for _, r in tbl.drop_duplicates(subset=['LGA_CODE']).dropna(subset=['MEDIAN']).iterrows()]

    breaks = compute_bins(medians, NUM_BINS)
    if not breaks:
        print("No medians available to compute bins. Exiting.")
        return
    colors_hex = get_viridis_colors(NUM_BINS)

    # Annotate SVG and write outputs
    artifact = annotate_svg_and_write(soup, matched_by_code, name_matches, fuzzy_unmatched, tbl, breaks, colors_hex, args.out_svg)

    # print brief summary
    matched_count = len(artifact['mapping'])
    unmatched_count = len(artifact['unmatched'])
    print("Wrote SVG:", args.out_svg)
    print("Wrote JSON:", args.out_json)
    print("Matched LGAs:", matched_count)
    print("Unmatched SVG shapes:", unmatched_count)
    print("Color scale:", COLOR_MAP, "; bins:", breaks)

if __name__ == "__main__":
    main()
