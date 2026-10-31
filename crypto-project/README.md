# crypto-project

**Live demo:** [https://block-chain-bigdata-bjyf5p9jt7nkghlee9gaoe.streamlit.app/](https://block-chain-bigdata-bjyf5p9jt7nkghlee9gaoe.streamlit.app/)

รายละเอียดโปรเจกต์ ผลลัพธ์ และวิธีรัน อยู่ใน [README หลัก](../README.md)

Quick start (รันจากโฟลเดอร์นี้):

```bash
pip install -r requirements.txt
python scripts/download_data.py      # ต้องมี ~/.kaggle/kaggle.json
# รัน notebooks/1.0-data-prep.ipynb (สร้าง data/processed/)
python scripts/train_models.py       # สร้าง models/*_graph.pkl ที่แอปใช้ (ผลซ้ำได้ ไม่ต้องใช้ Git LFS)
# (ทางเลือก) notebooks/2.1 → 2.2 สำหรับ evaluation และ drift
streamlit run app/main.py
```

ทดสอบ: `python -m pytest tests`
