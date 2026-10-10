# -*- coding: utf-8 -*-
"""
toby_rating_convert.py - converts a rating (FIDE, Chess.com, Lichess) into a FIDE-EQUIVALENT strength.

DATA SOURCE: ChessGoals, "Rating Comparison" - https://chessgoals.com/rating-comparison/ (public page).
ChessGoals built these tables from about 20,000 active players who have a rating on several platforms (Chess.com blitz as the common base; 1,923 active FIDE players),
themselves taken from the public Chess.com and Lichess data. Margins given by the source: +/- 75 (FIDE), +/- 70 (Lichess blitz), +/- 90 (Lichess rapid and classical),
+/- 115 (Chess.com rapid). There is no FIDE data below about 1550-1660 FIDE.
The values below are copied from the tables of that public page; the source is credited here and in NOTICE.md.

Usage: to_fide(2610, "lichess", "blitz") -> ~2400
"""
import bisect

# Table 1: Chess.com blitz (column 0) -> bullet, rapid, USCF, FIDE (None = no data)
_CC = [
    (500, 550, 815, 925, None), (600, 615, 905, 1000, None), (700, 685, 995, 1075, None), (800, 760, 1085, 1150, None),
    (900, 840, 1170, 1220, None), (1000, 920, 1255, 1285, None), (1100, 1010, 1340, 1350, 1660), (1150, 1055, 1380, 1385, 1665),
    (1200, 1100, 1420, 1415, 1670), (1250, 1145, 1460, 1445, 1675), (1300, 1195, 1500, 1475, 1685), (1350, 1240, 1540, 1510, 1700),
    (1400, 1290, 1580, 1540, 1710), (1450, 1340, 1620, 1570, 1725), (1500, 1385, 1655, 1600, 1740), (1550, 1435, 1690, 1630, 1760),
    (1600, 1485, 1730, 1655, 1780), (1650, 1540, 1765, 1685, 1800), (1700, 1590, 1800, 1715, 1820), (1750, 1640, 1835, 1745, 1840),
    (1800, 1690, 1870, 1775, 1865), (1850, 1745, 1900, 1805, 1890), (1900, 1795, 1935, 1835, 1915), (1950, 1850, 1965, 1860, 1940),
    (2000, 1900, 1995, 1890, 1965), (2100, 2005, 2055, 1950, 2020), (2200, 2110, 2110, 2010, 2075), (2300, 2215, 2165, 2070, 2135),
    (2400, 2320, 2215, 2130, 2190), (2500, 2425, 2260, 2195, 2245), (2600, 2530, 2300, 2255, 2300), (2700, 2630, 2340, 2325, 2350),
    (2800, 2725, 2375, 2390, 2400), (2900, 2825, 2405, 2460, 2445), (3000, 2915, 2430, 2535, 2485),
]
# Table 3: Lichess blitz, bullet, rapid, classical, USCF, FIDE
_LI = [
    (1090, 1060, 1290, 1505, 925, None), (1160, 1120, 1360, 1540, 1000, None), (1225, 1180, 1425, 1575, 1075, None),
    (1290, 1240, 1490, 1610, 1150, None), (1360, 1305, 1555, 1650, 1220, None), (1425, 1375, 1615, 1690, 1285, None),
    (1490, 1445, 1680, 1730, 1350, 1660), (1525, 1480, 1710, 1750, 1385, 1665), (1555, 1515, 1740, 1770, 1415, 1670),
    (1590, 1550, 1765, 1795, 1445, 1675), (1620, 1585, 1795, 1815, 1475, 1685), (1655, 1625, 1825, 1835, 1510, 1700),
    (1685, 1660, 1850, 1860, 1540, 1710), (1720, 1700, 1880, 1880, 1570, 1725), (1755, 1735, 1905, 1900, 1600, 1740),
    (1785, 1775, 1935, 1925, 1630, 1760), (1820, 1815, 1960, 1945, 1655, 1780), (1850, 1855, 1990, 1970, 1685, 1800),
    (1885, 1890, 2015, 1990, 1715, 1820), (1915, 1930, 2040, 2015, 1745, 1840), (1950, 1970, 2065, 2035, 1775, 1865),
    (1980, 2010, 2090, 2055, 1805, 1890), (2015, 2050, 2115, 2080, 1835, 1915), (2045, 2095, 2140, 2100, 1860, 1940),
    (2080, 2135, 2165, 2120, 1890, 1965), (2145, 2215, 2215, 2165, 1950, 2020), (2210, 2300, 2260, 2205, 2010, 2075),
    (2275, 2380, 2310, 2245, 2070, 2135), (2340, 2465, 2355, 2280, 2130, 2190), (2410, 2550, 2400, 2320, 2195, 2245),
    (2475, 2635, 2445, 2350, 2255, 2300), (2540, 2715, 2490, 2385, 2325, 2350), (2610, 2800, 2530, 2415, 2390, 2400),
    (2675, 2885, 2575, 2440, 2460, 2445), (2745, 2970, 2615, 2465, 2535, 2485),
]
_COLS_LI = {"blitz": 0, "bullet": 1, "rapid": 2, "classical": 3}
_COLS_CC = {"blitz": 0, "bullet": 1, "rapid": 2}


def _pairs(table, col):
    p = sorted((r[col], r[-1]) for r in table if r[-1] is not None)
    return [a for a, _ in p], [b for _, b in p]


def _interp(x, xs, ys):
    """Linear interpolation; outside the range: extrapolation with the slope of the last (or first) segment."""
    if x <= xs[0]:
        s = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return ys[0] + s * (x - xs[0])
    if x >= xs[-1]:
        s = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return ys[-1] + s * (x - xs[-1])
    k = bisect.bisect_right(xs, x)
    s = (ys[k] - ys[k - 1]) / (xs[k] - xs[k - 1])
    return ys[k - 1] + s * (x - xs[k - 1])


def to_fide(rating, platform, cadence="blitz"):
    """FIDE equivalent of a rating. platform: 'fide', 'chesscom', 'lichess'; cadence: 'blitz', 'rapid', 'bullet', 'classical' (Lichess only)."""
    platform = platform.lower().replace(".", "").replace("-", "")
    if platform == "fide":
        return float(rating)
    if platform in ("chesscom", "chess"):
        if cadence not in _COLS_CC:
            raise ValueError("chess.com : cadence blitz, rapid ou bullet")
        xs, ys = _pairs(_CC, _COLS_CC[cadence])
        return _interp(rating, xs, ys)
    if platform == "lichess":
        xs, ys = _pairs(_LI, _COLS_LI[cadence])
        return _interp(rating, xs, ys)
    raise ValueError("plateforme inconnue : fide, chesscom ou lichess")


def from_fide(fide, platform, cadence="blitz"):
    """Inverse function: rating (on the given platform and time control) of a player whose strength is the given FIDE equivalent."""
    platform = platform.lower().replace(".", "").replace("-", "")
    if platform == "fide":
        return float(fide)
    if platform in ("chesscom", "chess"):
        xs, ys = _pairs(_CC, _COLS_CC[cadence])
    elif platform == "lichess":
        xs, ys = _pairs(_LI, _COLS_LI[cadence])
    else:
        raise ValueError("plateforme inconnue : fide, chesscom ou lichess")
    order = sorted(range(len(ys)), key=lambda k: ys[k])
    xs, ys = [xs[k] for k in order], [ys[k] for k in order]
    if fide < ys[0]:
        # Below the lowest FIDE value of the tables (about 1660) the source has no data: extrapolating its last, nearly flat segment gives absurd platform ratings
        # (FIDE 1400 -> Lichess blitz -330). Below that point one FIDE point is taken to be one platform point.
        return xs[0] - (ys[0] - fide)
    return _interp(fide, ys, xs)   # x = FIDE values, y = platform ratings


if __name__ == "__main__":
    for args in [(2610, "lichess", "blitz"), (2745, "lichess", "blitz"), (2800, "lichess", "blitz"), (2400, "lichess", "rapid"), (2600, "lichess", "rapid"),
                 (2000, "lichess", "classical"), (2600, "chesscom", "blitz"), (1800, "chesscom", "rapid"), (2200, "fide", "blitz")]:
        print(args, "->", round(to_fide(*args)))
