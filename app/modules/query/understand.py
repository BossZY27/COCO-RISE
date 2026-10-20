"""Read what a question asks before any evidence is fetched.

pipeline.analyze finds a fund by its full name, and people rarely type one. They say ทุนสถาบัน, page
charge or ทุนหลังปริญญาเอก, ask about ปีหน้า, compare two funds, or ask which fund pays the most. This
module holds that vocabulary. Every mapping is a reviewed entry in data/aliases.json or a word list here,
never a model's guess, so a wrong mapping can be found in one place and fixed.
"""
import json
import re

from app.modules.knowledge import corpus as store

# Spellings people type for the same word. เเ is two sara e typed for แ, which looks the same on screen,
# and ไหร่ and มั้ย are the spoken forms of ไร and ไหม. The word lists below are written in the right-hand
# form, so "ปิดรับเมื่อไหร่" reaches the same answer as "ปิดรับเมื่อไร".
SPELLINGS = (("​", ""), ("เเ", "แ"), ("ไหร่", "ไร"), ("มั๊ย", "ไหม"), ("มั้ย", "ไหม"))
_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
# How Thai asks yes or no after ได้. One list, because a form missing from one of three copies once sent
# "คนนอกขอทุนได้หรือเปล่า" to the amount answer while "ได้ไหม" reached the eligibility one. บ่ is the Isan
# form, common in Khon Kaen; บ่อย (often) is not it.
YES_NO = ("ไหม", "มั้ย", "มั๊ย", "หรือไม่", "หรือเปล่า", "รึเปล่า", "หรือป่าว", "รึป่าว", "ป่าว", "เปล่า",
          "ป่ะ", "ปะ", "บ่")
_YES_NO = "(?:" + "|".join(re.escape(w) + ("(?!อ)" if w == "บ่" else "") for w in YES_NO) + ")"
CAN_ASK_RE = re.compile(r"ได้\s*" + _YES_NO)

COMPARE_WORDS = ("ต่างกัน", "แตกต่าง", "เทียบ", "มากกว่า", "น้อยกว่า", "ดีกว่า", "เยอะกว่า",
                 "สูงกว่า", "ต่ำกว่า")
SUPERLATIVE_HIGH = ("มากที่สุด", "สูงที่สุด", "เยอะที่สุด", "มากสุด", "สูงสุด", "เยอะสุด")
SUPERLATIVE_LOW = ("น้อยที่สุด", "ต่ำที่สุด", "น้อยสุด", "ต่ำสุด")
TIER_LIST_WORDS = ("แต่ละชั้น", "ทุกชั้น", "กี่ชั้น", "แต่ละระดับ", "ทุกระดับ", "กี่ระดับ", "กี่แบบ",
                   "มีชั้นอะไร", "มีระดับอะไร", "แบ่งเป็น")
DETAIL_WORDS = ("อธิบาย", "เงื่อนไขอะไร", "มีเงื่อนไข", "เงื่อนไขของ", "ขอเงื่อนไข",
                "เงื่อนไขในการรับ", "เงื่อนไขการรับ", "ขึ้นกับอะไร", "ขึ้นอยู่กับอะไร", "รายละเอียด",
                "หลักเกณฑ์อะไร", "ต้องทำอะไร", "ต้องมีอะไร", "คิดยังไง", "คิดอย่างไร")
HYPOTHETICAL_RE = re.compile(r"(?:ถ้า|หาก|สมมติ)(?:ว่า)?.{0,20}?(?:เคย)?ได้(?:รับ)?.+แล้ว.+ได้(?:อีก)?" + _YES_NO)
PERSONAL_WORDS = ("ผม", "ฉัน", "หนู", "ดิฉัน", "กระผม", "เรา", "ตัวเอง")
THIRD_PERSON_WORDS = ("เพื่อน", "เขา", "เธอ", "บุคคลนี้", "คนนี้", "คนนั้น", "ผู้สมัครรายนี้")
# People asked about by role. The applicant definition names three employment types; every other role
# gets the definition and the condition, never a verdict the document does not state.
PERSON_WORDS = ("พระสงฆ์", "พระ", "นักศึกษา", "นิสิต", "อาจารย์พิเศษ", "อาจารย์", "ข้าราชการ",
                "พนักงานมหาวิทยาลัย", "พนักงานราชการ", "พนักงาน", "ลูกจ้างมหาวิทยาลัย", "ลูกจ้าง",
                "บุคลากร", "บุคคลภายนอก", "คนนอก", "ศิษย์เก่า", "ผู้ช่วยวิจัย", "นักวิจัย", "เจ้าหน้าที่",
                "สายสนับสนุน", "ผู้เกษียณ")
# Words that ask whether someone may apply without saying what for. What is left once these, the
# pronouns and the roles are removed is the purpose, such as ซื้อรถ.
_ASKING = ("ได้หรือเปล่า", "ได้รึเปล่า", "ได้หรือป่าว", "ได้รึป่าว", "ได้หรือไม่", "ได้ไหม", "ได้มั้ย", "ได้ป่าว",
           "ได้ป่ะ", "ได้ปะ", "ได้บ่", "ได้บ้าง", "หรือเปล่า", "รึเปล่า", "หรือไม่", "มีสิทธิ์", "มีสิทธิ",
           "สามารถ", "อยากจะ", "อยาก", "ขอรับ", "ขอ", "ทุน", "จะ", "ใคร", "บ้าง", "ครับ", "ค่ะ", "คะ", "นะ",
           "หน่อย", "เงิน", "ได้", "ไหม", "มั้ย", "ไป", "ของ", "ที่", "เพื่อ", "สำหรับ", "การ", "ตัวไหน",
           "อะไร", "ไหน", "ใด", "เรื่อง", "บ้าง")
# Words every fund is about. "ขอทุนวิจัยได้ไหม" asks whether one may apply at all. Read as a purpose it got
# "ไม่พบทุนที่ระบุเรื่องวิจัย", which told a researcher the college funds no research.
GENERIC_PURPOSES = ("ทำงานวิจัย", "งานวิจัย", "ทำวิจัย", "การวิจัย", "วิจัย")
# No announcement names a country, so "ไปญี่ปุ่น" shares no wording with any fund. Every one of these is
# abroad, which the announcements do name, so the suggestion search reads them as ต่างประเทศ.
PLACES_ABROAD = ("ญี่ปุ่น", "จีน", "เกาหลี", "ไต้หวัน", "ฮ่องกง", "สิงคโปร์", "มาเลเซีย", "อินโดนีเซีย",
                 "เวียดนาม", "ลาว", "กัมพูชา", "เมียนมา", "พม่า", "ฟิลิปปินส์", "อินเดีย", "อเมริกา",
                 "สหรัฐ", "แคนาดา", "อังกฤษ", "สหราชอาณาจักร", "ฝรั่งเศส", "เยอรมัน", "อิตาลี", "สเปน",
                 "เนเธอร์แลนด์", "สวีเดน", "ออสเตรเลีย", "นิวซีแลนด์", "ยุโรป", "เมืองนอก", "ต่างแดน")
LEISURE_TRAVEL = ("ไปเที่ยว", "ท่องเที่ยว", "เที่ยวเล่น", "พักผ่อน", "ฮันนีมูน", "ช้อปปิ้ง")
OUTSIDE_ROLES = ("พระสงฆ์", "พระ", "นักศึกษา", "นิสิต", "บุคคลภายนอก", "คนนอก", "ศิษย์เก่า", "ผู้เกษียณ")
# "ทุนเกี่ยวกับการสอนมีอะไรบ้าง" asks for funds by subject: whatever follows the linking word, with the
# asking words trimmed off its end. สำหรับ is left out, because "ทุนสำหรับนักศึกษา" asks who, not what.
TOPIC_RE = re.compile(r"(?:ทุน|เงินรางวัล|ค่าตอบแทน).{0,14}?(?:ที่|ซึ่ง)?(?:เกี่ยวข้องกับ|เกี่ยวกับ|ด้าน)(.+)")
TOPIC_TAIL = ("มีทุนอะไรบ้าง", "มีทุนอะไร", "มีทุนไหม", "มีทุนบ้าง", "มีอะไรบ้าง", "มีอะไร", "มีไหม", "มีบ้าง",
              "หรือเปล่า", "หรือไม่", "บ้าง", "อะไร", "ไหม", "ครับ", "ค่ะ", "คะ", "นะ", "มี", "?")
# Messages that are not questions about a fund. Each gets a reply of its own, because "ระบบยังไม่แน่ใจว่า
# คำถามนี้หมายถึงทุนใด" in answer to a greeting reads as if the system misheard it.
GREETINGS = ("สวัสดี", "หวัดดี", "hello", "hi")
THANKS = ("ขอบคุณ", "ขอบใจ", "thank you", "thanks", "thank")
CAPABILITY = ("ทำอะไรได้บ้าง", "ช่วยอะไรได้บ้าง", "ถามอะไรได้บ้าง", "ใช้งานยังไง", "ใช้ยังไง", "คือใคร", "เป็นใคร")
_CHATTER = ("สำหรับข้อมูล", "สำหรับคำตอบ", "ระบบนี้", "ระบบ", "ครับ", "คับ", "ค่ะ", "คะ", "จ้า", "จ้ะ", "นะ",
            "มากๆ", "มาก", "เลย", "คุณ", "บอท", "น้อง", "ๆ")
_THAI = re.compile(r"[฀-๿]")
TWO_DIGIT_YEAR = re.compile(r"ปี(?:งบ(?:ประมาณ)?)?\s*(?:พ\.?\s*ศ\.?\s*)?([67]\d)(?![\d,.])")
NEXT_YEAR = ("ปีงบประมาณหน้า", "ปีงบหน้า", "ปีหน้า", "ปีถัดไป")
LAST_YEAR = ("ปีงบประมาณที่แล้ว", "ปีงบที่แล้ว", "ปีที่แล้ว", "ปีก่อน", "ปีที่ผ่านมา", "ปีงบก่อน")
THIS_YEAR = ("ปีงบประมาณนี้", "ปีงบนี้", "ปีนี้", "ปีปัจจุบัน")


def normalize(text: str) -> str:
    """One spelling for words typed several ways, Arabic digits, and single spaces."""
    text = text.translate(_THAI_DIGITS)
    for typed, meant in SPELLINGS:
        text = text.replace(typed, meant)
    return " ".join(text.split())


def mentions(text: str) -> list[dict]:
    """Every fund the text names, in reading order, as {"fund", "alternatives", "exact"}.

    Longer names claim their span first, so ทุนวิจัยสถาบัน is not also read as ทุนสถาบัน. A fund named
    in full or by a unique alias makes an ambiguous mention of the same fund redundant, so it is dropped.
    """
    lowered = text.lower()
    claimed: list[tuple[int, int]] = []
    found: list[tuple[int, dict]] = []

    def claim(needle: str, mention: dict):
        start = lowered.find(needle)
        while start != -1:
            end = start + len(needle)
            if all(end <= s or start >= e for s, e in claimed):
                claimed.append((start, end))
                found.append((start, mention))
            start = lowered.find(needle, end)

    for fund in store.view().FUNDS:                                   # FUNDS is sorted longest first
        claim(fund.lower(), {"fund": fund, "alternatives": [], "exact": True})
    for alias, fund in store.view().ALIASES:
        claim(alias, {"fund": fund, "alternatives": [], "exact": False})
    for alias, candidates in store.view().AMBIGUOUS:
        chosen = next((f for f, hints in candidates.items() if any(h in text for h in hints)), None)
        fund = chosen or next(iter(candidates))
        others = [] if chosen else [f for f in candidates if f != fund]
        claim(alias, {"fund": fund, "alternatives": others, "exact": False})

    ordered = [m for _, m in sorted(found, key=lambda pair: pair[0])]
    certain = {m["fund"] for m in ordered if not m["alternatives"]}
    result, seen = [], set()
    for m in ordered:
        if m["alternatives"] and (certain & ({m["fund"]} | set(m["alternatives"]))):
            continue
        if m["fund"] not in seen:
            seen.add(m["fund"])
            result.append(m)
    return result


def relative_year(text: str, base: int) -> int | None:
    """ปีหน้า, ปีที่แล้ว, ปีนี้ and two-digit years such as ปีงบ 69, measured from the base fiscal year."""
    two_digit = TWO_DIGIT_YEAR.search(text)
    if two_digit:
        return 2500 + int(two_digit.group(1))
    if any(w in text for w in NEXT_YEAR):
        return base + 1
    if any(w in text for w in LAST_YEAR):
        return base - 1
    if any(w in text for w in THIS_YEAR):
        return base
    return None


def asks_if_possible(text: str) -> bool:
    """ได้ followed by a yes-or-no ending: "ขอได้ไหม", "ได้หรือเปล่า", "ได้บ่"."""
    return bool(CAN_ASK_RE.search(text))


def is_personal(text: str) -> bool:
    return any(w in text for w in PERSONAL_WORDS)


def person_asked_about(text: str) -> str | None:
    explicit_self_role = next((role for role in PERSON_WORDS if re.search(
        rf"(?:ผม|ฉัน|หนู|ดิฉัน|กระผม|เรา|ตัวเอง).{{0,8}}?(?:เป็น|ทำงานเป็น|มีสถานะเป็น|สังกัด).{{0,8}}?{re.escape(role)}",
        text,
    )), None)
    if explicit_self_role:
        return explicit_self_role
    # Role words can be part of a grant's official name (for example
    # "ทุนนักวิจัยอาวุโส"). They describe the grant, not another applicant.
    without_fund_names = text
    grant_names = set(store.view().FUNDS)
    grant_names.update(alias for alias, _fund in store.view().ALIASES)
    grant_names.update(alias for alias, _funds in store.view().AMBIGUOUS)
    for grant_name in sorted(grant_names, key=len, reverse=True):
        without_fund_names = without_fund_names.replace(grant_name, " ")
    return next((w for w in PERSON_WORDS if w in without_fund_names), None)


def about_the_asker(text: str) -> bool:
    """Thai drops the subject, so "ขอทุนได้ไหม" asks about the asker as much as "ผมขอทุนได้ไหม" does.
    A question that names a role, or asks ใคร, is about other people."""
    # "เพื่อนผมเป็นลูกจ้าง" contains ผม, but the applicant is the friend.  The old
    # personal-word-first check silently used the selected demo profile for that other
    # person and could therefore return the opposite eligibility verdict.
    if any(word in text for word in THIRD_PERSON_WORDS):
        return False
    role = person_asked_about(text)
    if role:
        # A role typed into the question is not the authenticated/demo profile. Until
        # declared attributes are parsed and verified separately, answer the role in
        # general terms instead of silently borrowing the selected person's record.
        return False
    return is_personal(text) or "ใคร" not in text


def _strip(text: str, words: tuple[str, ...]) -> str:
    for word in sorted(words, key=len, reverse=True):
        text = text.replace(word, " ")
    return " ".join(text.split())


def purpose_of(text: str) -> str:
    """What the asker wants the money for: what follows ทุน, once the asking words, pronouns and roles
    are removed. Words before ทุน describe the person, so "นักศึกษาปริญญาเอกขอทุนได้ไหม" has none.

    "ขอทุนซื้อรถได้ไหม" leaves ซื้อรถ; "นักศึกษาขอทุนได้ไหม" leaves nothing, because it asks who; and
    "ขอทุนวิจัยได้ไหม" leaves nothing either, because every fund is for research.
    """
    rest = _strip(text.split("ทุน", 1)[1] if "ทุน" in text else text, _ASKING + PERSONAL_WORDS + PERSON_WORDS)
    if len(_THAI.findall(_strip(rest, GENERIC_PURPOSES))) < 3:
        return ""
    return rest


def topic_of(text: str) -> str:
    """The subject of "ทุนเกี่ยวกับ X" or "ทุนด้าน X", or "" when the question asks for none. A tail is
    trimmed only while three Thai letters remain, so "ทุนด้านเคมี" keeps the มี that ends เคมี."""
    match = TOPIC_RE.search(text)
    topic = match.group(1).strip() if match else ""
    trimmed = True
    while trimmed:
        trimmed = False
        for tail in TOPIC_TAIL:
            rest = topic[:-len(tail)].rstrip()
            if topic.endswith(tail) and len(_THAI.findall(rest)) >= 3:
                topic, trimmed = rest, True
    return topic if len(_THAI.findall(topic)) >= 3 else ""


def small_talk(text: str) -> str | None:
    """"greeting", "thanks" or "capability" when that is all the message says, otherwise None."""
    lowered = text.lower()
    kind = next((kind for kind, words in (("capability", CAPABILITY), ("thanks", THANKS), ("greeting", GREETINGS))
                 if any(w in lowered for w in words)), None)
    if kind is None:
        return None
    rest = _strip(lowered, GREETINGS + THANKS + CAPABILITY + _CHATTER)
    return kind if len(re.findall(r"[a-z฀-๿]", rest)) < 3 else None


def with_places_abroad(text: str) -> str:
    """The text, with ต่างประเทศ added when it names a place abroad."""
    # A country alone may describe research travel, but an explicitly leisure
    # trip is outside the source documents.  Do not turn that wording into the
    # research term "ต่างประเทศ" and accidentally recommend academic grants.
    if any(term in text for term in LEISURE_TRAVEL):
        return text
    return f"{text} ต่างประเทศ" if any(place in text for place in PLACES_ABROAD) else text


def wants_comparison(text: str) -> bool:
    return any(w in text for w in COMPARE_WORDS)


def wants_ranking(text: str) -> str | None:
    """"high" or "low" when the question asks which fund pays the most or the least."""
    if "ทุน" not in text:
        return None
    if any(w in text for w in SUPERLATIVE_HIGH):
        return "high"
    if any(w in text for w in SUPERLATIVE_LOW):
        return "low"
    return None


def wants_tier_list(text: str) -> bool:
    return any(w in text for w in TIER_LIST_WORDS)


def wants_details(text: str) -> bool:
    return any(w in text for w in DETAIL_WORDS)


def is_hypothetical(text: str) -> bool:
    return bool(HYPOTHETICAL_RE.search(text))


def __getattr__(name):
    if name in {"ALIASES", "AMBIGUOUS"}:
        return getattr(store.view(), name)
    raise AttributeError(name)
