"""
Normalization layer for Business Entity Resolution (ML Challenge 2026).

Self-contained: no external gazetteers, no network calls, no geocoding.
Indic romanization is *derived from the Unicode character database* (stdlib
`unicodedata`), i.e. an algorithm over codepoint names -- not a data lookup.

Per record it produces:
    name.toks      content tokens of the name (legal suffixes / fillers removed)
    name.skels     phonetic consonant skeleton per token (cross-script folding)
    name.dom       stem of an embedded web domain, if any
    addr.toks      street-identity tokens (street-type words & geo words removed)
    addr.nums      building/number tokens, leading zeros stripped
    addr.geo       administrative-geography tokens (city / state / region)
"""
from __future__ import annotations

import os
import re
import unicodedata

# --------------------------------------------------------------------------
# 1. Indic romanization, read off the Unicode character database.
# --------------------------------------------------------------------------
# Every Indic codepoint is *named* after its phonetic value
# ("DEVANAGARI LETTER SHA", "TAMIL VOWEL SIGN AA", "KANNADA SIGN VIRAMA"), so
# the phonetics can be read straight from `unicodedata` instead of assuming a
# uniform block layout -- which is wrong for Tamil (its consonant rows have
# gaps) and off-by-one for Devanagari (U+0929 NNNA).  Fully deterministic.

_RANGES = [(0x0900, 0x0D7F)]  # Devanagari .. Malayalam

# phonetic name -> latin.  Indic CA is /tS/, so it romanizes to "ch".
_PHON = {
    "KA": "ka", "KHA": "kha", "GA": "ga", "GHA": "gha", "NGA": "nga",
    "CA": "cha", "CHA": "chha", "JA": "ja", "JHA": "jha", "NYA": "nya",
    "TTA": "ta", "TTHA": "tha", "DDA": "da", "DDHA": "dha", "NNA": "na",
    "TA": "ta", "THA": "tha", "DA": "da", "DHA": "dha", "NA": "na",
    "NNNA": "na", "NNNNA": "na",
    "PA": "pa", "PHA": "pha", "BA": "ba", "BHA": "bha", "MA": "ma",
    "YA": "ya", "YYA": "ya", "RA": "ra", "RRA": "ra", "RHA": "ra",
    "LA": "la", "LLA": "la", "LLLA": "la", "LLLLA": "la",
    "VA": "va", "SHA": "sha", "SSA": "sha", "SA": "sa", "HA": "ha",
    "QA": "ka", "KHHA": "kha", "GHHA": "ga", "ZA": "za", "FA": "fa",
    "DDDHA": "da", "TTTA": "ta", "BBA": "ba", "LLLLLA": "la",
}
# vowels resolve to vowel letters only -- never to a consonant, otherwise the
# consonant skeleton picks up phantom letters (VOCALIC R must not emit "r").
_VOWNAME = {
    "A": "a", "AA": "aa", "I": "i", "II": "ii", "U": "u", "UU": "uu",
    "E": "e", "EE": "ee", "AI": "ai", "O": "o", "OO": "oo", "AU": "au",
    "VOCALIC R": "i", "VOCALIC RR": "i", "VOCALIC L": "i", "VOCALIC LL": "i",
    "CANDRA E": "e", "SHORT E": "e", "CANDRA O": "o", "SHORT O": "o",
    "CANDRA A": "a", "AY": "ai", "AW": "au", "PRISHTHAMATRA E": "e",
    "OE": "o", "OOE": "o", "UE": "u", "AA WITH AU": "au", "AP": "a",
}
_DIGITNAMES = "ZERO ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE".split()
_KILL = "\x00"  # virama sentinel: kills the preceding inherent vowel


def _build_indic_map():
    cons, vowel, other = {}, {}, {}
    for lo, hi in _RANGES:
        for cp in range(lo, hi + 1):
            ch = chr(cp)
            try:
                nm = unicodedata.name(ch)
            except ValueError:
                continue
            if " LETTER " in nm:
                ph = nm.split(" LETTER ", 1)[1]
                if ph in _VOWNAME:
                    vowel[ch] = _VOWNAME[ph]
                elif ph in _PHON:
                    cons[ch] = _PHON[ph]
                else:
                    other[ch] = ""
            elif " VOWEL SIGN " in nm:
                vowel[ch] = _VOWNAME.get(nm.split(" VOWEL SIGN ", 1)[1], "a")
            elif "SIGN VIRAMA" in nm or "SIGN PAMAADI" in nm or "AL-LAKUNA" in nm:
                other[ch] = _KILL
            elif "SIGN ANUSVARA" in nm or "SIGN CANDRABINDU" in nm:
                other[ch] = "n"
            elif "SIGN VISARGA" in nm:
                other[ch] = "h"
            elif " DIGIT " in nm and nm.split(" DIGIT ", 1)[1] in _DIGITNAMES:
                other[ch] = str(_DIGITNAMES.index(nm.split(" DIGIT ", 1)[1]))
            else:
                other[ch] = ""  # nukta, avagraha, punctuation, marks
    return cons, vowel, other


_CONS_MAP, _VOWEL_MAP, _OTHER_MAP = _build_indic_map()
_HAS_INDIC = re.compile(r"[ऀ-ൿ]")


def romanize(s: str) -> str:
    """Indic abugida -> latin.

    A consonant carries its inherent 'a' unless the next character is a
    dependent vowel sign or a virama.
    """
    out = []
    n = len(s)
    for i, ch in enumerate(s):
        c = _CONS_MAP.get(ch)
        if c is not None:
            nxt = s[i + 1] if i + 1 < n else ""
            if nxt in _VOWEL_MAP or _OTHER_MAP.get(nxt) == _KILL:
                c = c[:-1] if c.endswith("a") else c  # inherent vowel suppressed
            out.append(c)
            continue
        v = _VOWEL_MAP.get(ch)
        if v is not None:
            out.append(v)
            continue
        o = _OTHER_MAP.get(ch)
        if o is not None:
            out.append("" if o == _KILL else o)
            continue
        out.append(ch)
    return "".join(out)


# --------------------------------------------------------------------------
# 2. Unicode folding
# --------------------------------------------------------------------------
# digit-for-letter substitutions the generator injects ("Nati0nal", "Lifec0")
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"})


def fold(s: str | None, leet: bool = False) -> str:
    """Romanize Indic, strip accents (NFKD), lowercase, punctuation -> space."""
    if not s:
        return ""
    if _HAS_INDIC.search(s):
        s = romanize(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    if leet:
        s = s.translate(_LEET)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------
# 3. Phonetic consonant skeleton (cross-script / cross-spelling folding)
# --------------------------------------------------------------------------
_DIGRAPH = [("sch", "S"), ("sh", "S"), ("ch", "S"), ("ph", "P"), ("th", "T"),
            ("gh", "K"), ("kh", "K"), ("dh", "T"), ("bh", "P"), ("ck", "K"),
            ("qu", "K"), ("ng", "N"), ("ts", "S"), ("zh", "S"), ("gn", "N")]
_SINGLE = str.maketrans({
    "b": "P", "p": "P", "v": "P", "w": "P", "f": "P",
    "c": "K", "k": "K", "g": "K", "q": "K", "x": "K",
    "d": "T", "t": "T",
    "s": "S", "z": "S", "j": "S",
    "l": "L", "m": "M", "n": "N", "r": "R",
    "a": "", "e": "", "i": "", "o": "", "u": "", "y": "", "h": "",
})


# English orthography -> how the word is *pronounced*, which is what an Indic
# transliteration spells.  Measured on 150k true India pairs: share of S1 name
# skeletons found in the target rises 0.784 -> 0.850 (S2) and 0.787 -> 0.822
# (S3), while distinct US skeletons go *up* (6068 -> 6232 on 47k tokens), i.e.
# the rules separate more tokens than they merge.  Applied to both sides of a
# pair, so they only need to be deterministic, not linguistically exact.
_PHONO = [(re.compile(p), r) for p, r in (
    (r"tech", "tek"),          # technology / teknolojii
    (r"chr", "kr"),            # christ / krist
    (r"x", "ks"),              # impex / impeks, laxmi / lakshmi
    (r"gh(?=t)", ""),          # bright / brait
    (r"g(?=[eiy])", "j"),      # management / manejment, logistics / lojistiks
    (r"c(?=[eiy])", "s"),      # producer / prodyuusar
    (r"t(?=i[aou])", "sh"),    # solutions / solyuushans
    (r"t(?=ur)", "ch"),        # ventures / vencharsa
    (r"s(?=ion)", "sh"),       # vision / vijan
)]
SKEL2 = os.environ.get("BER_SKEL2", "1") == "1"


def skel(s: str) -> str:
    """Consonant skeleton; collapses spelling *and* transliteration variants.

    'consulting' and romanized Devanagari 'kansaltinga' both -> 'KNSLTN'.
    """
    if not s:
        return ""
    s = s.replace(" ", "")
    if SKEL2:
        for pat, rep in _PHONO:
            s = pat.sub(rep, s)
    for a, b in _DIGRAPH:
        s = s.replace(a, b)
    s = s.translate(_SINGLE)
    s = re.sub(r"[^A-Z]", "", s)
    return re.sub(r"(.)\1+", r"\1", s)  # collapse doubled consonants


# --------------------------------------------------------------------------
# 4. Vocabularies (hand-written; no external gazetteer)
# --------------------------------------------------------------------------
LEGAL = {
    # US / generic
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "lp", "llp", "plc", "pllc", "pc", "pa", "na", "ltda",
    # India
    "pvt", "private", "opc", "huf", "prvt",
    # France
    "sarl", "sasu", "sas", "eurl", "sci", "sa", "sca", "scs", "snc", "scop",
    "scm", "sel", "selarl", "gie", "eirl", "cie", "fils", "freres", "frere",
    "ei", "sem", "scic", "scp",
    # romanized Indic legal suffixes
    "praivet", "praivate", "limitad", "limited", "limitida", "limiteda",
    "praa", "li", "prai", "lita", "lamitada", "praivhet", "limitta",
}
# generic words the generator substitutes for a real name token: they carry no
# identity of their own and must never drive a match on their own
FILLER = {
    "service", "services", "center", "centre", "partners", "group", "holdings",
    "holding", "international", "intl", "enterprise", "enterprises",
    "solutions", "solution", "associates", "associate", "and", "the", "of",
    "de", "du", "des", "la", "le", "les", "au", "aux", "en", "et", "pour",
    "sur", "sous", "www", "com", "net", "org", "http", "https",
}
SUFFIX_NOISE = {"formerly", "known", "as", "dba", "aka", "nee", "trading",
                "smt", "shri", "sri", "messrs", "mr", "ms", "the"}

# street-type / unit words: structural, not identifying
STREET = {
    # US
    "st", "street", "str", "rd", "road", "ave", "av", "avenue", "dr", "drive",
    "ln", "lane", "ct", "court", "cir", "circle", "blvd", "boulevard", "pl",
    "place", "pkwy", "parkway", "hwy", "highway", "way", "trl", "trail",
    "ter", "terrace", "sq", "square", "pike", "loop", "path", "run", "row",
    "bend", "walk", "crossing", "xing", "unit", "apt", "suite", "ste",
    "floor", "fl", "building", "bldg", "block", "blk", "po", "box", "rm",
    "n", "s", "e", "w", "north", "south", "east", "west", "ne", "nw", "se", "sw",
    # France
    "rue", "r", "bd", "imp", "impasse", "chemin", "che", "ch", "allee", "all",
    "quai", "cite", "cours", "route", "rte", "voie", "passage", "villa",
    "sentier", "faubourg", "fg", "esplanade", "espl", "bis", "lieu", "dit",
    "residence", "batiment", "bat", "etage", "porte", "zone", "za", "zi", "zac",
    # India
    "marg", "chs", "plot", "survey", "door", "house", "hno", "hn", "no",
    "number", "num", "opp", "opposite", "near", "behind", "bh", "beside",
    "above", "care", "vill", "village", "taluk", "tq", "teh", "tehsil", "ps",
    "indl", "industrial", "gr", "ground", "flr", "gali", "galli", "kh",
}
# administrative-geography words: coarse location, weak identity
GEO_WORDS = {
    "county", "city", "town", "township", "twp", "cdp", "region", "hq",
    "dist", "district", "state", "province", "suburban", "urban", "rural",
    "metro", "greater", "central", "new", "old", "borough", "parish",
    "arrondissement", "commune", "cedex",
}

_DOMAIN = re.compile(r"\b([a-z0-9][a-z0-9\-]{2,})\s*\.\s*"
                     r"(com|net|org|in|co|io|fr|biz|info|us|edu|gov)\b")
_HASDIGIT = re.compile(r"\d")
_SPLITALNUM = re.compile(r"[^0-9a-z]+")


class Parsed(dict):
    """dict with attribute access, so downstream code reads cleanly."""
    __getattr__ = dict.__getitem__


def parse_name(raw: str | None) -> Parsed:
    f = fold(raw, leet=True)
    dom = ""
    m = _DOMAIN.search(f)
    if m:
        dom = m.group(1).replace("-", "")
        f = (f[:m.start()] + " " + f[m.end():]).strip()
    content, legal = [], []
    for t in f.split():
        if t in SUFFIX_NOISE or t in FILLER or len(t) == 1:
            continue
        if t in LEGAL:
            legal.append(t)
        else:
            content.append(t)
    seen, toks = set(), []
    for t in content:  # de-duplicate "Peridance Peridance", keep order
        if t not in seen:
            seen.add(t)
            toks.append(t)
    if not toks and dom:
        toks = [dom]
    return Parsed(norm=f, toks=toks, legal=legal, dom=dom,
                  skels=[skel(t) for t in toks])


def parse_addr(raw: str | None) -> Parsed:
    f = fold(raw, leet=False)
    if not f:
        return Parsed(norm="", toks=[], nums=[], geo=[], skels=[], tail="")
    nums, street, geo = [], [], []
    for t in f.split():
        if _HASDIGIT.search(t):
            for p in _SPLITALNUM.split(t):
                if not p:
                    continue
                if p.isdigit():
                    nums.append(p.lstrip("0") or "0")
                elif p not in STREET and len(p) > 1:
                    street.append(p)
        elif t in STREET or len(t) == 1:
            continue
        elif t in GEO_WORDS:
            geo.append(t)
        elif t in FILLER:
            continue
        else:
            street.append(t)
    seen, toks = set(), []
    for t in street:
        if t not in seen:
            seen.add(t)
            toks.append(t)
    tail = fold((raw or "").rsplit(",", 1)[-1])
    return Parsed(norm=f, toks=toks, nums=nums, geo=geo, tail=tail,
                  skels=[skel(t) for t in toks])


if __name__ == "__main__":
    pairs = [
        ("श्याम कंसल्टिंग प्राइवेट लिमिटेड", "Shyam Consulting Private Limited"),
        ("ಯುನೀಕ್ ಟೆಕ್ನಾಲಜಿ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "Unique Technology Pvt Ltd"),
        ("சிவா மீடியா பிரைவேட் லிமிடெட்", "Shiva Media Private Limited"),
        ("హరి కన్సల్టెన్సీ లిమిటెడ్", "Hari Consultancy Limited"),
        ("બોમ્બે પ્રોડક્ટ્સ પ્રાઇવેટ લિમિટેડ", "Bombay Products Private Limited"),
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "Ram Marketing Private Limited"),
        ("मॉडर्न फाइनेंस", "Modern Finance"),
        ("महाराष्ट्र", "Maharashtra"), ("পশ্চিমবঙ্গ", "West Bengal"),
        ("ಕರ್ನಾಟಕ", "Karnataka"), ("తెలంగాణ", "Telangana"),
        ("தமிழ்நாடு", "Tamil Nadu"), ("ગુજરાત", "Gujarat"),
    ]
    print(f"{'indic':26s} {'romanized':30s} {'skel(indic)':20s} {'skel(latin)':20s} shared")
    tot = hits = 0
    for a, b in pairs:
        pa, pb = parse_name(a), parse_name(b)
        sa, sb = set(pa.skels) - {""}, set(pb.skels) - {""}
        n = len(sa & sb)
        tot += 1
        hits += n > 0
        print(f"{a[:24]:26s} {romanize(a)[:28]:30s} {'+'.join(pa.skels)[:18]:20s} "
              f"{'+'.join(pb.skels)[:18]:20s} {n}")
    print(f"\ncross-script skeleton hit rate: {hits}/{tot}")
