# Pipeline

## เตรียมข้อมูล

PDF → PostgreSQL job → worker อ่าน text/OCR และแบ่ง chunks → ผู้ดูแลแก้และตรวจทุกหน้า → review → publish เพิ่ม corpus revision → worker สร้าง BGE-M3 generation ใน ChromaDB ข้อมูล facts/rules ต้องผูกหลักฐานและอนุมัติแยก ไม่ได้เกิดเป็นข้อมูลอนุมัติจาก OCR อัตโนมัติ

## ถามตอบ

`POST /api/query` → validation/rate limit → corpus snapshot → วิเคราะห์คำถามกับ context → LLM slot fallback เมื่อจำเป็น → ค้น BM25+BGE-M3 และ RRF พร้อม structured lookup → verified template → LLM rephrasing เมื่อมีหลักฐาน → candidate verification → corpus revision check → คำตอบและ citations

หาก index/embedding ไม่พร้อมใช้ BM25 พร้อมแจ้งเหตุผล หากโมเดลเขียนคำตอบไม่ได้หรือ candidate ไม่ผ่าน ใช้ template ที่ตรวจแล้ว หาก template/อ้างอิงไม่ผ่านหรือ corpus เปลี่ยนจะบล็อก การตรวจใช้แหล่งข้อมูลจริง ไม่มีการเติมคำตอบที่สร้างเองกลับเป็น evidence ไม่ส่ง token หรือ preview ที่ยังตรวจไม่เสร็จ
