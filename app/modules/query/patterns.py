"""Shared patterns for distinguishing catalogue, eligibility, and history questions."""

import re

# Broad personal-eligibility questions occur in both Thai word orders: a person may
# ask "ขอทุนอะไรได้บ้าง" or "ทุนอะไรบ้างที่ฉันขอได้".  Keep these separate from
# catalogue questions such as "มีทุนอะไรบ้าง", which contain no application verb.
_GENERIC_RESEARCH = r"(?:(?:ทำ)?(?:งาน)?วิจัย|การวิจัย)?"
_WHICH_FUND = r"(?:อะไร|ไหน|ตัวไหน|ตัวใด|ใด)"
_APPLICATION_VERB = r"(?:ขอ|สมัคร|ยื่น(?:ขอ)?)"
GENERIC_CATALOGUE_RE = re.compile(
    rf"(?:มี\s*ทุน\s*{_GENERIC_RESEARCH}\s*{_WHICH_FUND}"
    rf"|ทุน\s*{_GENERIC_RESEARCH}\s*มี\s*{_WHICH_FUND})"
)
BROAD_ELIGIBILITY_RE = re.compile(
    rf"(?:มีสิทธิ์?|สามารถ)\s*.{{0,12}}?{_APPLICATION_VERB}\s*ทุน\s*"
    rf"{_GENERIC_RESEARCH}\s*{_WHICH_FUND}"
    rf"|{_APPLICATION_VERB}\s*ทุน\s*{_GENERIC_RESEARCH}\s*{_WHICH_FUND}\s*(?:บ้าง\s*)?ได้"
    rf"|ทุน\s*{_GENERIC_RESEARCH}\s*{_WHICH_FUND}"
    rf"(?:(?!(?:เคย|มาแล้ว|ไปแล้ว|เอกสาร|วงเงิน)).){{0,40}}?{_APPLICATION_VERB}\s*ได้"
)
PERSONAL_HISTORY_RE = re.compile(
    rf"(?:เคย\s*(?:{_APPLICATION_VERB}|ได้รับ|ได้)\s*(?:ทุน|เงินทุน|เงินรางวัล)"
    rf"|{_APPLICATION_VERB}\s*ทุน.{{0,40}}?(?:มาแล้ว|ไปแล้ว)"
    rf"|(?:ได้รับ|เคย\s*ได้)\s*ทุน\s*{_GENERIC_RESEARCH}\s*{_WHICH_FUND}"
    rf"|(?:ได้รับ|ได้)\s*(?:ทุน|เงินทุน|เงินรางวัล).{{0,60}}?"
    rf"(?:แล้ว(?:หรือยัง)?|หรือยัง|ไหม|หรือไม่|หรือเปล่า|รึเปล่า|ป่าว|เปล่า)"
    rf"|ทุน\s*{_GENERIC_RESEARCH}\s*{_WHICH_FUND}.{{0,40}}?เคย\s*{_APPLICATION_VERB})"
)
REPEAT_ELIGIBILITY_RE = re.compile(
    rf"(?:แล้ว.{{0,60}}?{_APPLICATION_VERB}\s*(?:(?:ทุน|เงินทุน|เงินรางวัล)|ซ้ำ|อีก|เพิ่ม)"
    rf"|เคย.{{0,100}}?(?:แล้ว.{{0,60}}?)?{_APPLICATION_VERB}\s*"
    rf"(?:(?:ทุน|เงินทุน|เงินรางวัล)|ซ้ำ|อีก|เพิ่ม)).{{0,80}}?ได้"
)
