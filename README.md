[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/fansevanoo/gta6-twitch-ccu-forecast-01-entry-acct3/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/fansevanoo/gta6-twitch-ccu-forecast-01-entry-acct3/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch CCU forecaster — pre-release-signal kernel

Forecasts Grand Theft Auto VI's launch-day Twitch concurrent viewers, hours
2–24, as five quantiles per hour. The program reads one case file and writes
one forecast file; Python 3.13 standard library only, deterministic, no
network.

## Method

Each past launch donates its release-day curve as log-ratios
`ln(viewers_h / viewers_1)`. Donors are weighted by a kernel over launch
similarity (tier, platform class, log hour-1 print, log 24 h pre-release
buzz ratio); the top 16 are kept. Per hour the forecast is Gaussian in log
space: centre = kernel-weighted mean log-ratio, spread = kernel-weighted
standard deviation, with a remapped spread for console targets.

New in this entry: for console targets the kernel also uses two pre-release
signals from the case's own history — the last-hour pre-release buzz
(log viewers in the last hour before the release day, relative to hour 1)
and the pre-release diurnal peak hour (the UTC hour at which the category's
audience peaks over the 7 days before release, a proxy for fanbase
geography, which drives the release-day shape through the timezone unlock
sequence). Both were selected by leave-one-out cross-validation over the
kit's comparables, not by fitting the four practice cases.

## Validation (leave-one-out over the 37 usable comparables)

| split | this entry | v2-style reference |
| --- | --- | --- |
| ALL (37) | 0.3802 | 0.3847 |
| CONSOLE (8) | 0.1529 | 0.1855 |
| practice-like (7, console + comparable tier) | 0.1371 | 0.1831 |

Expected practice loss is lower than the earlier entries' 0.19–0.21, but
the cross-validation-to-practice gap is real and unmeasurable locally:
the practice answers are not in the kit.
