#!/usr/bin/env python3
"""Forecast GTA VI launch-day Twitch viewers hour by hour (acct3 entry).

Method: kernel-weighted Gaussian log-ratio forecaster with pre-release
signals. Each past launch donates its release-day curve as log-ratios
ln(viewers_h / viewers_1). For the target case the donors are weighted by a
kernel over launch similarity, the top 16 are kept, and per hour the
forecast is a Gaussian in log space: centre = kernel-weighted mean log-ratio,
spread = kernel-weighted standard deviation.

What is new here versus a plain kernel mean (the earlier entries used only
size/platform/buzz kernels): the kernel adds two pre-release signals, and
only for console targets, where the rolling-midnight unlock makes them
meaningful -
  * the last-hour pre-release buzz: log(viewers in the last hour before the
    release day / hour-1 print). The audience already waiting in the
    category at midnight is a strong level signal.
  * the pre-release diurnal peak hour: the UTC hour at which the category's
    pre-release audience peaks (mean log viewers by UTC hour over the 7 days
    before release). It proxies the fanbase's geography, which drives the
    release-day shape through the timezone unlock sequence. Donors whose
    pre-release audience peaks at a similar UTC hour get more weight
    (Gaussian on the circular hour distance, bandwidth 5 h).

The other kernel factors are tier (comparable 1.0 / early-access 0.7 /
weak 0.5), platform-class mismatch (0.2), log hour-1 print (bandwidth 1.2)
and log 24 h pre-release buzz ratio (bandwidth 1.3). Spreads are remapped
for console targets (sd' = (0.5*sd + 0.15) * 0.65) and scaled by 0.9 for
multi-platform targets. Stdlib only, deterministic, no network.
"""
import json
import math
import os
import re
import sys
from datetime import datetime

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

BW_H1 = 1.2
BW_BUZZ24 = 1.3
BW_BUZZ_LAST1 = 1.0
BW_PEAK_HOUR = 5.0
PLAT_MISMATCH = 0.2
TOP_K = 16
TIER_W = {'comparable': 1.0, 'early-access-contaminated': 0.7, 'weak': 0.5}


def gauss(x, bw):
    return math.exp(-0.5 * (x / bw) ** 2)


def circ_dist(h1, h2):
    d = abs(h1 - h2) % 24
    return min(d, 24 - d)


def cos_sim(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def parse_time(s):
    return datetime.fromisoformat(s.replace('Z', '+00:00'))


def tier_of(comp):
    m = re.search(r'Tier:\s*([A-Za-z\-]+)', comp.get('unlock', ''))
    return m.group(1) if m else '?'


def donor_features(comp):
    """Extract one comparable launch as a donor. Returns None when the
    launch has no complete 24 h release day or is the target's own
    pre-release history (not a launch day)."""
    if tier_of(comp) == 'target-history':
        return None
    ds = parse_time(comp['day_start'])
    day = {}
    pre = []
    for p in comp.get('hourly') or []:
        t = parse_time(p['from'])
        h = int((t - ds).total_seconds() // 3600) + 1
        if 1 <= h <= 24:
            day[h] = p['viewers']
        elif t < ds:
            pre.append((t, p['viewers']))
    if len(day) != 24 or any(v is None or v <= 0 for v in day.values()):
        return None
    day_list = [day[h] for h in range(1, 25)]
    pre24 = [v for (t, v) in pre if (ds - t).total_seconds() <= 24 * 3600]
    buzz24 = sum(pre24) / len(pre24) if pre24 else 1.0
    last1 = [v for (t, v) in pre if (ds - t).total_seconds() <= 3600]
    buzz_last1 = last1[-1] if last1 else buzz24
    # diurnal profile: mean log viewers by UTC hour over the 7 days before
    prof = {}
    for (t, v) in pre:
        if (ds - t).total_seconds() <= 7 * 24 * 3600:
            prof.setdefault(t.hour, []).append(math.log(max(v, 1)))
    peak_hr = None
    dprof = None
    if prof:
        avg = {h: sum(v) / len(v) for h, v in prof.items()}
        mean_prof = sum(avg.values()) / len(avg)
        profile = [avg.get(h, mean_prof) for h in range(24)]
        mp = sum(profile) / 24.0
        dprof = [x - mp for x in profile]
        peak_hr = max(range(24), key=lambda h: profile[h])
    platforms = comp.get('platforms', '')
    return {
        'name': comp.get('name', ''),
        'day': day_list,
        'h1': day_list[0],
        'buzz24': buzz24,
        'buzz_last1': buzz_last1,
        'peak_hr': peak_hr,
        'dprof': dprof,
        'console': 'pc' not in platforms.lower(),
        'tier_w': TIER_W.get(tier_of(comp), 0.5),
    }


def target_is_console(case):
    """Decide whether the target launch is console-only from the case notes.
    Parses the 'released <date> on <platforms>' clause first (a later 'PC
    2019' remark must not flip a console launch), then falls back to a
    'console only' mention, then to any standalone PC mention, else console
    (every launch in this competition is console-only)."""
    notes = case.get('notes', '') or ''
    m = re.search(r'released\s+\S+\s+on\s+([^;(\n]+)', notes, re.IGNORECASE)
    if m:
        return 'pc' not in m.group(1).lower()
    if 'console only' in notes.lower():
        return True
    if re.search(r'\bPC\b', notes):
        return False
    return True


def target_features(case):
    """Features of the target case itself, from its known hour-1 print and
    its pre-release history (history_hourly)."""
    known = (case.get('known') or [{}])[0]
    h1 = known.get('viewers') or 1
    if h1 <= 0:
        h1 = 1
    ds = parse_time(case['day_start'])
    hist = case.get('history_hourly') or []
    pre = []
    for p in hist:
        t = parse_time(p['from'])
        if t < ds:
            pre.append((t, p['viewers']))
    pre24 = [v for (t, v) in pre if (ds - t).total_seconds() <= 24 * 3600]
    buzz24 = sum(pre24) / len(pre24) if pre24 else 1.0
    last1 = [v for (t, v) in pre if (ds - t).total_seconds() <= 3600]
    buzz_last1 = last1[-1] if last1 else buzz24
    prof = {}
    for (t, v) in pre:
        if (ds - t).total_seconds() <= 7 * 24 * 3600:
            prof.setdefault(t.hour, []).append(math.log(max(v, 1)))
    peak_hr = None
    dprof = None
    if prof:
        avg = {h: sum(v) / len(v) for h, v in prof.items()}
        mean_prof = sum(avg.values()) / len(avg)
        profile = [avg.get(h, mean_prof) for h in range(24)]
        mp = sum(profile) / 24.0
        dprof = [x - mp for x in profile]
        peak_hr = max(range(24), key=lambda h: profile[h])
    return {
        'h1': h1,
        'buzz24': buzz24,
        'buzz_last1': buzz_last1,
        'peak_hr': peak_hr,
        'dprof': dprof,
        'console': target_is_console(case),
    }


def kernel_weight(donor, target):
    """Similarity weight between one donor and the target case."""
    w = donor['tier_w']
    if donor['console'] != target['console']:
        w *= PLAT_MISMATCH
    w *= gauss(math.log(donor['h1']) - math.log(target['h1']), BW_H1)
    rb_d = math.log(max(donor['buzz24'], 1.0) / donor['h1'])
    rb_t = math.log(max(target['buzz24'], 1.0) / target['h1'])
    w *= gauss(rb_d - rb_t, BW_BUZZ24)
    if target['console']:
        # pre-release signals: last-hour buzz and diurnal peak hour.
        # They only carry meaning for console launches, whose rolling
        # midnight unlock makes the pre-release audience's geography and
        # last-hour momentum predictive of the day curve.
        lb_d = math.log(max(donor['buzz_last1'], 1.0) / donor['h1'])
        lb_t = math.log(max(target['buzz_last1'], 1.0) / target['h1'])
        w *= gauss(lb_d - lb_t, BW_BUZZ_LAST1)
        if donor['peak_hr'] is not None and target['peak_hr'] is not None:
            w *= gauss(circ_dist(donor['peak_hr'], target['peak_hr']),
                       BW_PEAK_HOUR)
    return w


def wmean(vals, ws):
    s = sum(ws)
    if s <= 0:
        return sum(vals) / len(vals)
    return sum(v * w for v, w in zip(vals, ws)) / s


def wsd(vals, ws, mean):
    s = sum(ws)
    if s <= 0:
        m = sum(vals) / len(vals)
        return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
    return math.sqrt(sum(w * (v - mean) ** 2 for v, w in zip(vals, ws)) / s)


def forecast_case(case):
    """Forecast hours 2..24 for one case. Returns {hour: [q05..q95]}."""
    target = target_features(case)
    donors = []
    for comp in case.get('comparables') or []:
        d = donor_features(comp)
        if d is not None:
            donors.append(d)
    h1 = target['h1']
    if not donors:
        # No usable donor: fall back to the house baseline shape so the
        # case stays valid (persistence with a widening spread).
        out = {}
        for h in range(2, 25):
            out[h] = [h1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]
        return out
    scored = sorted(
        ((kernel_weight(d, target), d) for d in donors),
        key=lambda t: (-t[0], t[1]['name']))
    scored = scored[:TOP_K]
    ws = [w for w, _ in scored]
    pool = [d for _, d in scored]
    out = {}
    for h in range(2, 25):
        lrs = [math.log(d['day'][h - 1] / d['h1']) for d in pool]
        mu = wmean(lrs, ws)
        sd = wsd(lrs, ws, mu)
        if target['console']:
            sd_eff = (0.5 * sd + 0.15) * 0.65
        else:
            sd_eff = sd * 0.9
        qs = [h1 * math.exp(mu + z * sd_eff) for z in Z]
        # safety net: finite, positive, capped, non-decreasing
        fixed = []
        prev = 0.0
        for q in qs:
            q = min(max(q, 1.0), 1e9)
            q = max(q, prev)
            fixed.append(q)
            prev = q
        out[h] = fixed
    return out


def main():
    in_dir = os.environ.get('MAKEFUN_INPUT_DIR') or '/input'
    out_dir = os.environ.get('MAKEFUN_OUTPUT_DIR') or '/output'
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith('.in'))
    for name in names:
        with open(os.path.join(in_dir, name), 'r', encoding='utf-8') as f:
            case = json.load(f)
        fc = forecast_case(case)
        hours = [{'hour': h, 'q': fc[h]} for h in range(2, 25)]
        out_name = name[:-3] + '.out'
        with open(os.path.join(out_dir, out_name), 'w', encoding='utf-8') as f:
            json.dump({'hours': hours}, f)
    return 0


if __name__ == '__main__':
    sys.exit(main())
