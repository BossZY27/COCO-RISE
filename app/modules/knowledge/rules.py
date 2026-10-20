"""Render approved rule conditions without evaluating a visitor's identity."""
from app.modules.knowledge.formatting import cite


def describe_rules(rules):
    lines, citations = [], []
    for rule in rules:
        for condition in rule["conditions"]:
            field, op, value = condition["field"], condition["op"], condition["value"]
            label = {"staff_type": "ประเภทบุคลากร", "department": "สังกัด"}.get(field)
            if label and op == "in":
                lines.append(f"{label}ต้องเป็นหนึ่งใน: " + ", ".join(value))
            elif label and op == "eq":
                lines.append(f"{label}ต้องเป็น{value}")
            elif op == "lt" and field.startswith("grants_this_year."):
                lines.append(f"จำนวนครั้งที่ได้รับ{field.split('.', 1)[1]}ในปีงบประมาณต้องน้อยกว่า {value}")
            elif op == "not_received":
                lines.append("ต้องยังไม่ได้รับทุนหรือรางวัลดังต่อไปนี้ในปีงบประมาณ: " + ", ".join(value))
            else:
                return None
        citations.append(cite(rule["source"]))
    if not lines:
        return None
    return "ตามกฎที่ตรวจแล้วของ" + rules[0]["fund"] + ": " + "; ".join(lines) + " ระบบยังไม่ตัดสินสิทธิ์รายบุคคล", citations
