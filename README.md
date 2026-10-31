# Bitcoin AML: ตรวจจับธุรกรรมผิดกฎหมายด้วย Machine Learning

ระบบตรวจจับธุรกรรม Bitcoin ที่เกี่ยวข้องกับกิจกรรมผิดกฎหมาย (ฟอกเงิน, ransomware, dark market) จาก **Elliptic Dataset**
พร้อม Dashboard ที่ออกแบบจากมุมมองของทีม Compliance / AML: อธิบายได้ว่า *ทำไม* ธุรกรรมถูกตั้งธง (SHAP), ปรับ threshold ตามกำลังคนที่ตรวจได้, ติดตามว่าโมเดลเสื่อมตามเวลาหรือไม่ และดูเครือข่ายการโอนเงินรอบธุรกรรมที่น่าสงสัย

<!-- TODO: ใส่ภาพหน้าจอ dashboard ที่ docs/screenshots/ แล้วเปิดบรรทัดด้านล่าง
![Dashboard](docs/screenshots/dashboard.png)
-->

## ปัญหา

ทีม Compliance ตรวจธุรกรรมได้จำนวนจำกัด แต่ธุรกรรมผิดกฎหมายมีสัดส่วนน้อยมาก (~10% ของธุรกรรมที่มี label) การสุ่มตรวจหรือใช้กฎตายตัวจึงเสียแรงไปกับธุรกรรมปกติจำนวนมาก
โปรเจกต์นี้สร้างโมเดลจัดอันดับความเสี่ยง เพื่อให้ทีมตรวจรายการที่น่าสงสัยที่สุดก่อน และอธิบายเหตุผลของการแจ้งเตือนได้

## Dataset

[Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set): ธุรกรรม Bitcoin จริง 203,769 รายการ, การโอนเงินระหว่างธุรกรรม 234,355 เส้น, 49 ช่วงเวลา (time step ละประมาณ 2 สัปดาห์)

| | จำนวน |
|---|---|
| Illicit (ผิดกฎหมาย) | 4,545 |
| Licit (ปกติ) | 42,019 |
| Unknown (ไม่มี label, ไม่ใช้ train) | 157,205 |

แต่ละธุรกรรมมี 166 features (94 local + 72 aggregated จากธุรกรรมข้างเคียง) และเราเพิ่ม **graph features** (in-degree, out-degree) จาก edge list

## วิธีการ

1. **Data prep** (`1.0-data-prep.ipynb`): รวม features กับ labels, ตัด unknown, สร้าง directed graph ด้วย NetworkX แล้วเพิ่ม in/out-degree
2. **Time-based split**: train = time step 1–34, test = 35–49
   ไม่ใช้การสุ่มแบ่ง เพราะในงานจริงโมเดลต้องทำนาย *อนาคต* การสุ่มจะทำให้ข้อมูลช่วงเวลาเดียวกันรั่วไปอยู่ทั้ง train และ test จนผลดูดีเกินจริง
3. **SMOTE เฉพาะ train set** เพื่อแก้ class imbalance โดยไม่ให้ข้อมูลสังเคราะห์ปนเข้า test
4. **Models**: Logistic Regression, Random Forest, XGBoost (`2.0-model-training_v1.ipynb`)
5. **Evaluation** (`2.1-model-evaluation_v1.ipynb`): precision, recall, F1, ROC-AUC ของ class illicit บน test set
6. **Threshold & drift analysis** (`2.2-threshold-and-drift.ipynb`): PR curve, threshold sweep, Precision@K และ recall ราย time step
7. **Explainability**: SHAP ทั้งภาพรวม และรายธุรกรรม

## ผลลัพธ์

Test set = time step 35–49 (16,670 ธุรกรรม, illicit 1,083), threshold 0.5, ค่าของ class **illicit**

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **XGBoost** | **0.905** | 0.733 | **0.810** | 0.941 |
| Random Forest | 0.926 | 0.718 | 0.809 | 0.945 |
| Logistic Regression (scaled, C=0.01) | 0.168 | 0.902 | 0.284 | 0.881 |

**ข้อสังเกต**
- Tree-based models ดีกว่า Logistic Regression ชัดเจน LR ได้ recall สูง แต่ precision ต่ำมาก (แจ้งเตือน 5 ครั้งเป็นของจริงแค่ 1) ซึ่งใช้งานจริงไม่ได้เพราะทีมตรวจจะจมกับ false alarm
- Graph features (in/out-degree) ช่วยได้เล็กน้อย: XGBoost F1 0.804 → 0.810 เมื่อเทียบกับไม่มี graph features
- Recall ~73% ที่ threshold 0.5 ยังพลาดธุรกรรมผิดกฎหมายราว 1 ใน 4 → ดู notebook 2.2 สำหรับการเลือก threshold ตามนโยบาย และการวิเคราะห์ว่าโมเดลแย่ลงหลัง time step 43 หรือไม่ (paper ต้นฉบับรายงานว่า dark market ขนาดใหญ่ถูกปิดในช่วงนั้น ทำให้รูปแบบเปลี่ยน)

## Dashboard (Streamlit)

| ส่วน | ใช้ทำอะไร |
|---|---|
| Model Performance | เทียบ 3 โมเดลบน test set แบบสด (คำนวณจากโมเดลจริง ไม่ hard-code) |
| **Alert Threshold** (sidebar) | ปรับเกณฑ์แจ้งเตือน ทุกตัวเลขในหน้าเปลี่ยนตาม |
| **PR Curve + Review Budget** | "ถ้าทีมตรวจได้แค่ K รายการ จะเจอของจริงกี่รายการ" (Precision@K) |
| **Performance Over Time** | precision/recall ราย time step เพื่อจับ concept drift |
| SHAP | ภาพรวม feature ที่สำคัญ และเหตุผลรายธุรกรรม |
| Fraud Prediction | ทำนายธุรกรรมใน test set พร้อมเทียบกับ label จริง และ export CSV |
| **Transaction Network** | กราฟธุรกรรมข้างเคียง (เงินเข้า/ออก) แยกสี illicit / licit / unknown |

## วิธีรัน

```bash
git clone https://github.com/suphakit085/Block-Chain-BigData.git
cd Block-Chain-BigData/crypto-project

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**1. ดาวน์โหลดข้อมูล** ไปไว้ที่ `crypto-project/data/raw/` (ต้องมี `elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`)

```bash
# ต้องตั้งค่า Kaggle API (~/.kaggle/kaggle.json) ก่อน
pip install kaggle
python scripts/download_data.py
```
หรือดาวน์โหลดเองจาก [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

**2. รัน notebooks ตามลำดับ** (ใน `crypto-project/notebooks/`)

| Notebook | Output |
|---|---|
| `1.0-data-prep.ipynb` | `data/processed/elliptic_bitcoin_dataset_v0.csv`, `..._v1_graph.csv` |
| `2.0-model-training_v1.ipynb` | `models/*_smote_v1_graph.pkl` |
| `2.1-model-evaluation_v1.ipynb` | ตารางผลลัพธ์ + feature importance |
| `2.2-threshold-and-drift.ipynb` | PR curve, threshold sweep, Precision@K, drift |

(`2.0-model-training.ipynb` / `2.1-model-evaluation.ipynb` คือเวอร์ชัน V0 ที่ไม่มี graph features ใช้เทียบ)

**3. เปิด Dashboard** (รันจากโฟลเดอร์ `crypto-project/`)

```bash
streamlit run app/main.py
```

## โครงสร้าง

```
crypto-project/
├── app/
│   ├── main.py            # Streamlit dashboard
│   └── utils.py           # load data/models, metrics, SHAP, transaction graph
├── notebooks/             # 1.0 data prep → 2.0 train → 2.1 evaluate → 2.2 threshold & drift
├── scripts/
│   └── download_data.py   # Kaggle downloader
├── data/                  # (ไม่อยู่ใน git) raw/ และ processed/
├── models/                # (ไม่อยู่ใน git) ไฟล์ .pkl จาก notebook 2.0
└── requirements.txt
```

## ข้อจำกัดและสิ่งที่จะทำต่อ

- Features ของ Elliptic ถูก anonymize จึงอธิบายความหมายทางธุรกิจของแต่ละ feature ไม่ได้ (SHAP บอกได้แค่ว่า `feat_x` สำคัญ)
- ยังไม่ได้ใช้ข้อมูลจาก 157k ธุรกรรมที่ไม่มี label (เช่น semi-supervised)
- ลอง Graph Neural Network (GCN) ที่ใช้โครงสร้างกราฟเต็มรูปแบบ แทน degree features
- ระบบจริงควร retrain เป็นระยะ และ monitor recall รายช่วงเวลาตามที่เห็นใน drift analysis

## ทีมพัฒนา

- **Suphakit Kaewkaemthong**: model training (time-based split, SMOTE), evaluation, Streamlit app, threshold & drift analysis
- **Annop Sangsila**: data preparation & graph features, Kaggle data pipeline, SHAP integration

โปรเจกต์วิชา Big Data / Blockchain, วิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น
