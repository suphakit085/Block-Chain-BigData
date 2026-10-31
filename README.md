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
| XGBoost | 0.898 | 0.729 | 0.805 | 0.943 |
| Random Forest | 0.912 | 0.720 | 0.805 | 0.945 |
| Logistic Regression (scaled, C=0.01) | 0.168 | 0.902 | 0.284 | 0.881 |

*หมายเหตุ: ตารางและ dashboard นับเป็น fraud เมื่อ P ≥ threshold ส่วน `model.predict()` ของ sklearn ปัดกรณี P = 0.5 พอดีเป็น licit — RF มี 15 ธุรกรรมที่ได้ 0.5 พอดี จึงได้ precision 0.926 / F1 0.809 ใน notebook 2.1 แทน 0.912 / 0.805 (XGBoost และ LR ไม่ได้รับผล)*

**ข้อสังเกต**
- Tree-based models ดีกว่า Logistic Regression ชัดเจน LR ได้ recall สูง แต่ precision ต่ำมาก (แจ้งเตือน 6 ครั้งเป็นของจริงแค่ 1) ซึ่งใช้งานจริงไม่ได้เพราะทีมตรวจจะจมกับ false alarm
- Graph features (in/out-degree) **ไม่ได้ช่วย**: XGBoost ไม่มี degree ได้ F1 0.811 เทียบกับมี 0.805 (อยู่ในช่วงความคลาดเคลื่อน) ฟีเจอร์ที่ช่วยจริงคือ aggregated features จากเพื่อนบ้าน (local-only 93 ตัว F1 0.70 → ทั้งหมด 0.80) ดูหัวข้อ "ตรวจสอบความถูกต้อง"
- Recall ~73% ที่ threshold 0.5 ยังพลาดธุรกรรมผิดกฎหมายราว 1 ใน 4 → ดู notebook 2.2 สำหรับการเลือก threshold ตามนโยบาย และการวิเคราะห์ว่าโมเดลแย่ลงหลัง time step 43 หรือไม่ (paper ต้นฉบับรายงานว่า dark market ขนาดใหญ่ถูกปิดในช่วงนั้น ทำให้รูปแบบเปลี่ยน)

### Concept drift: โมเดลพังหลัง time step 43

Recall เฉลี่ยราย time step (threshold 0.5) ก่อนและหลัง time step 43:

| Model | t35–43 | t44–49 | ลดลง |
|---|---|---|---|
| Logistic Regression (scaled) | 0.885 | 0.781 | −0.104 |
| Random Forest | 0.747 | **0.093** | −0.654 |
| XGBoost | 0.765 | **0.093** | −0.672 |

- หลัง time step 43 โมเดล tree-based จับธุรกรรมผิดกฎหมายได้ไม่ถึง 10% ตรงกับที่ paper ต้นฉบับ (Weber et al., 2019) รายงานว่า dark market ขนาดใหญ่ถูกปิดช่วงนั้น ทำให้รูปแบบธุรกรรมผิดกฎหมายเปลี่ยนไปจากที่โมเดลเคยเห็น
- Logistic Regression ทนต่อ drift ได้ดีกว่า (recall ยังเหลือ 0.78) แต่แลกกับ precision ที่ต่ำมาก (0.17) โมเดลที่ซับซ้อนกว่าจึงเรียนรู้รูปแบบเฉพาะของช่วง train ได้แม่นกว่า แต่ก็ปรับตัวกับรูปแบบใหม่ได้แย่กว่า
- **นัยต่องานจริง:** ตัวเลข F1 รวมของทั้ง test set (0.81) ซ่อนปัญหานี้ไว้ ระบบ AML ที่ใช้งานจริงต้อง monitor recall รายช่วงเวลา, retrain เมื่อมี label ใหม่ และไม่ควรพึ่งโมเดลเดียวโดยไม่มีกฎหรือคนตรวจประกอบ
- ผลของการ retrain แบบ walk-forward และการวัดแยกช่วงด้วย precision/PR-AUC อยู่ใน "ตรวจสอบความถูกต้อง" ด้านล่าง

## การตรวจสอบเพิ่มเติม (`scripts/validate_threshold.py`)

แบ่งตามเวลา: train 1–24, validation 25–34, test 35–49 (ใช้ test รายงานผลอย่างเดียว) ผลเต็มอยู่ใน `reports/threshold_validation.csv`

- **เลือก threshold บน validation ใช้กับ test ไม่ได้ผล**: F1 บน validation สูง 0.94–0.96 แต่บน test เหลือ 0.74–0.80 (concept drift) XGBoost ที่ใช้ threshold จาก validation (0.28) ได้ F1 0.758 ต่ำกว่าใช้ 0.5 (0.779) จึงรายงานผลที่ threshold 0.5 คู่กับกราฟ drift
- **SMOTE ไม่ได้แย่กว่า class weight**: PR-AUC บน test ของ RF 0.785 เทียบ 0.773 และ XGBoost 0.784 เทียบ 0.764
- **Isotonic calibration ช่วยให้ probability ตรงขึ้น** (Brier ของ RF 0.033 → 0.023) แต่ไม่ได้ช่วยแยกธุรกรรมผิดกฎหมายได้ดีขึ้น
- **Degree features ไม่รั่วข้อมูลอนาคต**: edge ทั้ง 234,355 เส้นอยู่ภายใน time step เดียวกัน
- Logistic Regression ต้อง scale ฟีเจอร์ (PR-AUC 0.19 → 0.36) แต่ยังอ่อนกว่าโมเดล tree มาก

## ตรวจสอบความถูกต้อง (`scripts/audit_data.py`, `scripts/audit_models.py`)

**ข้อมูล** — ผ่านทุกข้อ: จำนวนแถว 46,564 ตรงกับ raw ที่มี label, txId ไม่ซ้ำ, label=1 ตรงกับ class "1" (illicit) ทุกแถว, feature ตรงกับไฟล์ raw, in/out-degree ตรงกับ edge list และ edge ทุกเส้นอยู่ใน time step เดียวกัน (ไม่รั่วข้อมูลอนาคต)

**โมเดลทำนายได้จริง ไม่ใช่ความบังเอิญ**
- สลับ label แล้วฝึกด้วย pipeline เดิม → PR-AUC 0.060 (ใกล้ prevalence 0.065) ยืนยันว่าไม่มี leakage ใน pipeline
- Random scorer ได้ PR-AUC 0.066 / ROC-AUC 0.50 เทียบกับ XGBoost 0.814 / 0.943
- Bootstrap 95% CI ของ F1 (test set): XGBoost [0.784, 0.824], Random Forest [0.785, 0.824] → สองโมเดลไม่ต่างกันอย่างมีนัยสำคัญ
- ผลใกล้เคียง baseline Random Forest ใน paper ต้นฉบับของ Elliptic (illicit F1 ≈ 0.80)

**ข้อจำกัดสำคัญ: โมเดลมองไม่เห็น fraud หลัง time step 43**

| ช่วง | ธุรกรรม | illicit | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|---|
| step 35–42 | 9,983 | 914 | 0.954 | 0.861 | 0.905 | 0.927 |
| step 43–49 | 6,687 | 169 | 0.055 | 0.018 | 0.027 | 0.057 |

ตัวเลข F1 0.805 รวมทั้งช่วงจึงดีกว่าความจริงหลัง step 43 มาก PR-AUC หลัง step 43 (0.057) ใกล้ prevalence ของช่วงนั้น (~0.025) แปลว่า *การจัดอันดับก็ใช้ไม่ได้* ไม่ใช่แค่ threshold ผิด การลด threshold จึงไม่ช่วย ต้อง retrain ด้วยข้อมูลใหม่หรือใช้โมเดลที่ทนต่อ drift ในระบบจริงต้อง monitor recall รายช่วงเวลา (ดู "Performance Over Time" ใน dashboard)

**retrain ช่วยได้แค่ไหน** (`scripts/drift_retraining.py`, walk-forward: ทำนายแต่ละ step ด้วยโมเดลที่ฝึกจาก step ก่อนหน้า, label มาช้า 1 step, XGBoost, step 43–49)

| กลยุทธ์ | PR-AUC | Recall @0.5 | Recall @0.1 | Precision @0.1 |
|---|---|---|---|---|
| static (ฝึกครั้งเดียว step 1–34) | 0.044 | 0.024 | 0.047 | 0.028 |
| expanding (ฝึกใหม่ด้วยข้อมูลทั้งหมดที่รู้ label) | **0.434** | 0.154 | **0.462** | 0.520 |
| sliding window 5 step | 0.401 | 0.112 | 0.402 | 0.576 |
| sliding window 10 step | 0.265 | 0.077 | 0.260 | 0.386 |

- การ retrain คืน *ความสามารถในการจัดอันดับ* ได้มาก (PR-AUC 0.04 → 0.43) แต่โมเดลยังระวังเกินไปที่ threshold 0.5 ต้องลด threshold (ที่ 0.1 ได้ recall ~46%, precision ~52%) ซึ่งตัวเลขตรงนี้เลือก threshold จากช่วงทดสอบเอง จึงเป็นภาพที่ดีเกินจริงเล็กน้อย
- window ยาวดีกว่า window สั้น (ข้อมูล illicit สะสมมากกว่า) และถ้า label มาช้า 3 step ผลลดลงมาก (expanding PR-AUC 0.43 → 0.16) → **ความเร็วของ label feedback สำคัญพอ ๆ กับตัวโมเดล**
- ยังไม่ถึงระดับก่อน step 43 (F1 ~0.90) ถึงอย่างนั้นก็ดีกว่าไม่ retrain อย่างชัดเจน

**Ablation** (F1 บน test): ทุกฟีเจอร์ 0.805 · ไม่มี degree 0.811 · local-only 93 ตัว 0.70 · degree อย่างเดียว 0.16

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
- ทดลองรับมือ drift: retrain แบบ rolling window, ensemble กับ Logistic Regression, หรือแจ้งเตือนอัตโนมัติเมื่อ distribution ของ score เปลี่ยน

## ทีมพัฒนา

- **Suphakit Kaewkaemthong**: model training (time-based split, SMOTE), evaluation, Streamlit app, threshold & drift analysis
- **Annop Sangsila**: data preparation & graph features, Kaggle data pipeline, SHAP integration

โปรเจกต์วิชา Big Data / Blockchain, วิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น
