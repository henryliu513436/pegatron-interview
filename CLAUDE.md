# CLAUDE.md — Smart Factory Alert Agent 專案規格

本文件是給 coding agent 的唯一權威規格來源。實作、測試、debug、refactor 時若與本文件衝突，以本文件為準；若本文件本身有遺漏，先在 `docs/ai_usage_log.md` 記錄你的假設與理由，再繼續實作。

---

## 1. 專案目標

模擬工廠設備感測器資料，開發一個異常告警 AI Agent：

- 生成 100–500 筆帶時間戳的模擬感測器資料（temp / pressure / vibration + label）
- 用「規則式（rule-based）」與「機器學習（IsolationForest novelty detection）」兩軌並行偵測異常
- 合併兩軌結果產生分級告警（severity）
- 呼叫本機 Ollama（`gemma4:e4b`）為每筆告警生成可執行建議，LLM 不可用時降級為模板建議
- CLI 即時（可模擬重播）輸出告警，同步寫入 `alerts.log`
- 提供 `--evaluate` 模式，量化評估 rule / ML / ensemble 三組偵測器的 precision / recall / F1 與混淆矩陣

---

## 2. 資料流（文字圖）

```
generate_data.py
    │  覆寫產生 data/raw_data.csv
    │  (timestamp, temp, pressure, vibration, label；三個感測值含 ~2% 缺失值)
    ▼
preprocess.load_raw()
    │  讀取、依 timestamp 排序、型別轉換
    ▼
preprocess.handle_missing_values()
    │  對 temp/pressure/vibration 線性內插 + bfill，之後不得再有 NaN
    ▼
features.add_rolling_features()
    │  在「完整、未切分」的時間序列上一次計算因果 rolling 特徵
    │  (temp/pressure/vibration 各自的 rolling_mean / rolling_std / diff)
    ▼
preprocess.temporal_split()
    │  依時間順序切成三段（互斥、依序）：
    │    Block A (前 70%，僅 normal 列) → train_fit
    │    Block B (接續 15%，僅 normal 列) → cal
    │    Block C (最後 15%，normal+abnormal) → test
    ▼
preprocess.fit_scaler(train_fit) → scaler
preprocess.transform_features(train_fit/cal/test, scaler)
    │  在 train_fit 的 ML_FEATURE_COLUMNS 上 fit，三段皆 transform
    │  transform 結果以 `{col}_scaled` 新欄位附加，原始欄位保留
    ▼
    ┌───────────────────────────────┬───────────────────────────────┐
    ▼                               ▼
rule_detector.detect(test)     ml_detector.fit(train_fit_scaled)
（吃 test 的原始未縮放值）        ml_detector.calibrate_threshold(cal_scaled)
                                ml_detector.detect(test_scaled)
    └───────────────┬───────────────┘
                     ▼
              ensemble.combine(test)
       （合併 rule_flag + ml_flag → severity + triggered_sensors,
         回傳完整 DataFrame，包含所有列）
                     ▼
        ┌────────────┴────────────┐
        ▼                         ▼
llm_advisor.generate_suggestion   （--evaluate 模式在此分支，
（僅對 is_anomaly_final == True   改呼叫 evaluate.py，不經過此路徑）
 的列呼叫，或 fallback 模板）
        ▼
   output.render_alert() + output.append_to_log()
   （CLI 印出 + 寫入 logs/alerts.log，可用 --replay-speed 模擬逐筆重播）
```

---

## 3. 鐵律（Non-negotiable Rules）— 違反視為驗收失敗

1. **label 只有兩種合法用途**：(a) `temporal_split()` 中用來篩選 Block A/B 的 normal 列；(b) `evaluate.py` 中作為 ground truth。除此之外，**任何偵測器、特徵工程、標準化流程都不得讀取或使用 label**，label 也絕不能成為模型輸入特徵。
2. **rolling 特徵必須因果**：只能使用截至當下（含）為止的歷史資料。`rolling()` 一律 `min_periods=1`，**禁止 `center=True`**。必須在切分前對完整時間序列一次計算完成——不可先切分再分段各自計算，否則 Block B/C 開頭列會失去 Block A 的歷史脈絡。
3. **scaler 與 IsolationForest 只能 fit 在 train_fit（Block A 的 normal 列）上**。cal（Block B）與 test（Block C）一律只能被 `transform()` / `predict()` / `decision_function()`，不得參與任何 `fit()`。
4. **cal 必須與 train_fit 互斥且時間上完全晚於 train_fit**。threshold 必須用 cal 的 `decision_function()` 分布推導（見 §4 `ml_detector.calibrate_threshold`），**不得用 train_fit 自己的 decision_function 算閾值**（會導致閾值偏樂觀、高估模型表現）。
5. **標準化順序固定**：缺值處理 → 特徵工程（rolling）→ 時間切分 → fit scaler（僅 train_fit）→ transform 全部三段。禁止「先標準化再算 rolling」。
6. **LLM 呼叫的任何失敗（連線失敗／逾時／空回應／格式不合法）都不得讓 `agent.py` / `main.py` 拋出未捕捉例外**，一律 fallback 到模板建議，並在輸出中標記 `fallback=True`。
7. `rule_detector` 與 `ml_detector` 的 `detect()` **只對 test（Block C）呼叫**；train_fit 只用於 `fit()`，cal 只用於 `calibrate_threshold()`，兩者都不需要跑異常偵測。

---

## 4. 模組介面契約

專案為單層結構，所有程式碼模組平放於專案根目錄，執行期產出的資料/紀錄放在 `data/`、`logs/`、`docs/` 子目錄下（ these are output directories, not program modules, they don't affect the single-layer structure）。

```
smart_factory_alert_agent/
├── config.py
├── generate_data.py
├── preprocess.py
├── features.py
├── rule_detector.py
├── ml_detector.py
├── ensemble.py
├── llm_advisor.py
├── evaluate.py
├── output.py
├── agent.py
├── main.py
├── tests/
├── requirements.txt
├── README.md
├── data/           # raw_data.csv 輸出於此
├── logs/           # alerts.log 輸出於此
└── docs/           # confusion_matrices.png, ai_usage_log.md 輸出於此
```

### `config.py`

集中定義以下**全部常數**，其他模組一律 `from config import ...`，不得在各模組中硬編碼重複的數值：

```python
# ---------- Data generation ----------
N_ROWS = 300
ANOMALY_RATIO = 0.10
MISSING_RATIO = 0.02          # 僅套用於 temp/pressure/vibration
INTERVAL_MINUTES = 1
START_TIME = "2024-06-03 19:00:00"
RANDOM_SEED = 42

THRESHOLDS = {
    "temp":      {"normal_min": 45.0, "normal_max": 50.0, "abnormal_high": 52.0, "abnormal_low": 43.0},
    "pressure":  {"normal_min": 1.00, "normal_max": 1.05, "abnormal_high": 1.08, "abnormal_low": 0.97},
    "vibration": {"normal_max": 0.04, "abnormal_high": 0.07},
}

# ---------- Temporal split（比例總和須為 1.0，依時間順序切分）----------
TRAIN_RATIO = 0.70   # Block A：fit scaler + IsolationForest（僅 normal）
CAL_RATIO = 0.15     # Block B：threshold 校準（僅 normal，時間須晚於 Block A）
# 剩餘 ~0.15 為 Block C（test，normal+abnormal 混合）

# ---------- General Constraints ----------
MIN_REQUIRED_SAMPLES = 10

# ---------- Generation Ranges (derived from THRESHOLDS to ensure no-gap/no-overlap) ----------
# Format: {sensor: { "normal": (min, max), "abnormal_high": (min, max), "abnormal_low": (min, max) }}
# Note: Vibration only has abnormal_high.
GEN_RANGES = {
    "temp": {
        "normal": (45.0, 50.0),
        "abnormal_high": (52.5, 56.0), # > 52.0
        "abnormal_low": (38.0, 42.5),  # < 43.0
    },
    "pressure": {
        "normal": (1.00, 1.05),
        "abnormal_high": (1.09, 1.15), # > 1.08
        "abnormal_low": (0.85, 0.96),  # < 0.97
    },
    "vibration": {
        "normal": (0.02, 0.04),
        "abnormal_high": (0.08, 0.12), # > 0.07
    },
}

# ---------- Preprocessing ----------
MISSING_VALUE_STRATEGY = "linear_interpolate_then_bfill"
SCALER_TYPE = "standard"      # standard | robust | minmax

# ---------- Feature engineering ----------
ROLLING_WINDOW = 5

RAW_SENSOR_COLUMNS = ["temp", "pressure", "vibration"]
# features.py 產出的欄位命名須與此一致：
# {sensor}_rolling_mean, {sensor}_rolling_std, {sensor}_diff
ML_FEATURE_COLUMNS = (
    RAW_SENSOR_COLUMNS
    + [f"{c}_rolling_mean" for c in RAW_SENSOR_COLUMNS]
    + [f"{c}_rolling_std" for c in RAW_SENSOR_COLUMNS]
    + [f"{c}_diff" for c in RAW_SENSOR_COLUMNS]
)

# ---------- ML detector (IsolationForest) ----------
IF_N_ESTIMATORS = 100
IF_CONTAMINATION = 0.01       # 僅作內部正則化，非實際告警閾值來源
IF_RANDOM_STATE = 42
THRESHOLD_PERCENTILE = 5      # 取 cal 集 decision_function 分布的第 5 百分位作為 ml threshold

# ---------- LLM advisor ----------
OLLAMA_MODEL = "gemma4:e4b"
OLLAMA_HOST = "http://localhost:11434"
LLM_TIMEOUT_SECONDS = 6
LLM_MAX_RETRIES = 1

# ---------- CLI ----------
DEFAULT_REPLAY_SPEED = 0.0    # 秒；0 = 不模擬即時，>0 = 逐筆間隔秒數

# ---------- Paths ----------
RAW_DATA_PATH = "data/raw_data.csv"
ALERTS_LOG_PATH = "logs/alerts.log"
CONFUSION_MATRIX_PATH = "docs/confusion_matrices.png"
AI_USAGE_LOG_PATH = "docs/ai_usage_log.md"
```

### `generate_data.py`

```python
def generate_dataset(
    n_rows: int = N_ROWS,
    anomaly_ratio: float = ANOMALY_RATIO,
    missing_ratio: float = MISSING_RATIO,
    start_time: str = START_TIME,
    interval_minutes: int = INTERVAL_MINUTES,
    seed: int = RANDOM_SEED,
    output_path: str = RAW_DATA_PATH,
) -> pd.DataFrame:
```
- 欄位：`timestamp`(str, `'YYYY-MM-DD HH:MM:SS'`)、`temp`/`pressure`/`vibration`(float)、`label`(`'normal'|'abnormal'`)。
- 異常事件必須均勻分佈於整段時間軸（例如隨機分佈起始點），不得因配額提前用完而導致序列後段僅有正常資料。
- 異常類型設計：
    - Spike 與 Drift：代表機台故障，三個感測器同時聯動異常。
    - Stuck：代表感測器訊號卡住，僅影響單一感測器。
- 標記邏輯：
    - Spike 與 Stuck：依四捨五入後的數值是否超出 `normal` 區間判定 `label='abnormal'`。
    - Drift：整個事件段落（從起始點到結束點）一律標記為 `label='abnormal'`，不依逐點數值判定。
- 依 `anomaly_ratio` 決定異常列總量；異常列的三個感測值中至少一項落在對應 `abnormal_*` 區間，其餘落在 `normal_*`  l區間；normal 列三個感測值均落在 normal 區間內。抽樣需以 `seed` 保證可重現。
- 產生後對 `temp`/`pressure`/`vibration` 依 `missing_ratio` 隨機注入 `NaN`；`timestamp`、`label` 不得為 `NaN`。
- **固定行為為覆寫 `output_path`**（不做檔案存在性檢查）；因 `seed` 固定，同一份 config 下每次執行結果相同，不需要保留舊檔。
- 回傳依 `timestamp` 遞增排序的 DataFrame。
- 例外：`n_rows < MIN_REQUIRED_SAMPLES` 時 `raise ValueError`；輸出目錄不存在時自動建立。
- **驗收條件**：經過 `temporal_split()` 切分後的 test 段（最後 15%）必須含有 abnormal 列。

### `preprocess.py`

```python
def load_raw(path: str = RAW_DATA_PATH) -> pd.DataFrame:
```
讀取 CSV，`timestamp` 轉為 `pandas.Timestamp` 並依其排序。`timestamp`/`label` 含 `NaN` 時 `raise ValueError`。

```python
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
```
對 `RAW_SENSOR_COLUMNS` 三欄位依 `MISSING_VALUE_STRATEGY`：先 `df[col].interpolate(method="linear")`，序列開頭仍缺值者再 `bfill()`。處理後三欄位不得再有 `NaN`，否則 `raise ValueError`。**必須在 `features.add_rolling_features()` 之前呼叫**。

```python
def temporal_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    cal_ratio: float = CAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
```
輸入須為已依 `timestamp` 排序、已完成 `add_rolling_features()` 的完整資料。依列順序切成三段：
- Block A `[0, train_ratio)`：僅保留 `label=='normal'` → 回傳 `train_fit`
- Block B `[train_ratio, train_ratio+cal_ratio)`：僅保留 `label=='normal'` → 回返 `cal`
- Block C `[train_ratio+cal_ratio, 1.0]`：全部保留 → 回傳 `test`

Block A/B 中 `label=='abnormal'` 的列一律捨棄，不進入任何後續流程（不併入 test）。例外：`train_fit`、`cal` 或 `test` 篩出的列數 `< MIN_REQUIRED_SAMPLES` 時 `raise ValueError`。

```python
def fit_scaler(train_fit_df: pd.DataFrame) -> object:
```
只在 `train_fit_df[ML_FEATURE_COLUMNS]` 上 fit `SCALER_TYPE` 指定的 scaler，回傳 fitted scaler 物件。

```python
def transform_features(df: pd.DataFrame, scaler: object) -> pd.DataFrame:
```
用已 fit 的 scaler transform `df[ML_FEATURE_COLUMNS]`，結果以新欄位 `f"{col}_scaled"` 附加到 `df` 的複本上，**原始欄位保留不變**；回傳新 DataFrame。`train_fit`/`cal`/`test` 皆呼叫此函式。

### `features.py`

```python
def add_rolling_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
```
輸入為已排序、已補值、**尚未切分**的完整資料。對 `RAW_SENSOR_COLUMNS` 各自計算：
- `{col}_rolling_mean` = `df[col].rolling(window=window, min_periods=1).mean()`
- `{col}_rolling_std` = `df[col].rolling(window=window, min_periods=1).std()`，結果 `fillna(0)`
- `{col}_diff` = `df[col].diff().fillna(0)`

禁止 `center=True`。回傳新增欄位後的 DataFrame，原始欄位不變。

### `rule_detector.py`

```python
def detect(df: pd.DataFrame, thresholds: dict = THRESHOLDS) -> pd.DataFrame:
```
僅使用**原始未縮放**的 `temp/pressure/vibration`。新增欄位：
- `rule_flag`(bool)：任一感測值落在對應 `abnormal_*` 區間即為 `True`
- `rule_reason`(str)：以 `;` 分隔列出觸發的感測器與方向，如 `"temp_high;vibration_high"`；未觸發則為空字串

不得讀取或使用 `label`。只對 test（Block C）呼叫。

### `ml_detector.py`

```python
def fit(train_fit_scaled: pd.DataFrame) -> object:
```
在 `train_fit_scaled[[f"{c}_scaled" for c in ML_FEATURE_COLUMNS]]` 上 fit `IsolationForest(n_estimators=IF_N_ESTIMATORS, contamination=IF_CONTAMINATION, random_state=IF_RANDOM_STATE)`。

```python
def calibrate_threshold(model: object, cal_scaled: pd.DataFrame, train_fit_scaled: pd.DataFrame, percentile: float = THRESHOLD_PERCENTILE) -> float:
```
內部先以 [f"{c}_scaled" for c in ML_FEATURE_COLUMNS] 篩選輸入 `cal_scaled` 欄位。對篩選後的資料算 `model.decision_function(...)`，取其第 `percentile` 百分位數作為 threshold 並回傳。若 `cal_scaled` 與 `train_fit_scaled` 有共同 index，`raise ValueError`（防呆檢查，實際互斥保證由 `temporal_split` 負責）。

```python
def detect(df_scaled: pd.DataFrame, model: object, threshold: float) -> pd.DataFrame:
```
內部先以[f"{c}_scaled" for c in ML_FEATURE_COLUMNS] 篩選輸入 `df_scaled` 欄位。新增欄位 `ml_score`(float，`decision_function` 值) 與 `ml_flag`(bool，`ml_score < threshold`)。只對 test（Block C）呼叫。

### `ensemble.py`

```python
def combine(df: pd.DataFrame) -> pd.DataFrame:
```
輸入需含 `rule_flag`、`rule_reason`、`ml_flag`、`ml_score`，以及原始 `temp/pressure/vibration` 與其 `_scaled` 欄位。新增欄位：
- `is_anomaly_final`(bool) = `rule_flag or ml_flag`
- `severity`：
  - `rule_flag and ml_flag` → `"CRITICAL"`
  - `rule_flag and not ml_flag` → `"HIGH"`
  - `(not rule_flag) and ml_flag` → `"MEDIUM"`
  - 兩者皆 `False` → `None`（非異常，不產生警報）
- `triggered_sensors`(str)：
  - 若 `rule_flag == True`：直接沿用 `rule_reason` 拆解出的感測器名稱
  - 若僅 `ml_flag == True`（MEDIUM）：取 `RAW_SENSOR_COLUMNS` 對應之 `_scaled` 欄位中絕對值最大的那一個，對應回原始感測器名稱

**回傳包含所有列的完整 DataFrame（非異常列 `severity` 為 `None`）**。其篩選行為由 `agent.run_pipeline` 處理，僅將 `is_anomaly_final == True` 的列交給 `llm_advisor`/`output`。

### `llm_advisor.py`

```python
def generate_suggestion(record: dict, use_llm: bool = True) -> tuple[str, bool]:
```
`record` 需含 `timestamp`、`triggered_sensors`、`severity`、三個原始感測值。回傳 `(suggestion_text, is_fallback)`。
- `use_llm=False`（對應 `main.py --no-llm`）：直接呼叫 `template_suggestion`，不嘗試連線 Ollama，回傳 `(text, True)`。
- `use_llm=True`：呼叫 Ollama（`OLLAMA_MODEL`, `OLLAMA_HOST`），`timeout=LLM_TIMEOUT_SECONDS`，失敗（連線錯誤／逾時／空回應）重試至多 `LLM_MAX_RETRIES` 次，仍失敗則 fallback 到 `template_suggestion`，回傳 `(text, True)`；成功則回傳 `(text, False)`。**不得讓例外往外拋出**。

```python
def template_suggestion(record: dict) -> str:
```
依 `severity` 與 `triggered_sensors` 組合固定句型模板（不依賴外部服務），回傳建議文字。

### `evaluate.py`

```python
def evaluate(test_df_with_predictions: pd.DataFrame, label_col: str = "label") -> dict:
```
輸入需含 `label`（僅在此函式作為 ground truth 使用）、`rule_flag`、`ml_flag`、`is_anomaly_final`。對三者分別以 `label=='abnormal'` 為正類，計算 precision/recall/f1 與 confusion matrix。回傳 `{"rule": {...}, "ml": {...}, "ensemble": {...}}`，並呼叫 `plot_confusion_matrices()`。

```python
def plot_confusion_matrices(metrics: dict, output_path: str = CONFUSION_MATRIX_PATH) -> None:
```
用 matplotlib 畫 rule/ml/ensemble 三個並排混淆矩陣子圖，存成一張 PNG 至 `output_path`，目錄不存在時自動建立。

### `output.py`

```python
def render_alert(alert_row: dict, suggestion: str, is_fallback: bool) -> str:
```
格式化單筆警報字串，格式：
`"[{timestamp}] {severity} | sensors={triggered_sensors} | suggestion={suggestion} | fallback={is_fallback}"`
`print()` 到終端機，並回傳同樣的字串供寫檔。

```python
def append_to_log(formatted_alert: str, log_path: str = ALERTS_LOG_PATH) -> None:
```
以 append 模式寫入純文字 log；目錄不存在時自動建立。

```python
def replay(alerts_df: pd.DataFrame, suggestions: list[tuple[str, bool]], replay_speed: float = DEFAULT_REPLAY_SPEED) -> None:
```
依序對每筆呼叫 `render_alert` + `append_to_log`；`replay_speed > 0` 時每筆之間 `time.sleep(replay_speed)`。

### `agent.py`

```python
def run_pipeline() -> dict:
```
Orchestrator，依序呼叫：
`generate_data.generate_dataset()` → `preprocess.load_raw()` → `preprocess.handle_missing_values()` → `features.add_rolling_features()`（完整序列）→ `preprocess.temporal_split()` → `preprocess.fit_scaler(train_fit)` → `preprocess.transform_features()`（train_fit/cal/test 皆呼叫）→ `rule_detector.detect(test)` → `ml_detector.fit(train_fit_scaled)` → `ml_detector.calibrate_threshold(model, cal_scaled, train_fit_scaled)` → `ml_detector.detect(test_scaled, model, threshold)`（rule 和 ml 都往同一張 test 表上加欄位,不做兩張表 merge，依相同 index join）→ `ensemble.combine(test)`。

在此階段產出兩份資料集：
1. `test_df`：`ensemble.combine` 回傳的完整結果，供 `evaluate.py` 使用。
2. `alerts_df`：從 `test_df` 中篩選出 `is_anomaly_final == True` 的列，供 `llm_advisor` 使用。

回傳 `dict`，至少含：`"alerts_df"`、`"test_df"`、`"scaler"`、`"ml_model"`、`"ml_threshold"`。**此函式不呼叫 `llm_advisor` 或 `output`**，純資料處理，由 `main.py` 依模式決定後續動作。

### `main.py`

CLI 唯一入口，argparse 支援：

| flag | 行為 |
|---|---|
| （無 flag，預設） | `run_pipeline()` → 對 `alerts_df` 逐列呼叫 `llm_advisor.generate_suggestion(use_llm=True)` 並收集結果 `list[tuple[str, bool]]` $\to$ `output.replay(alerts_df, suggestions, replay_speed=DEFAULT_REPLAY_SPEED)` |
| `--evaluate` | `run_pipeline()` → `evaluate.evaluate(test_df)`，印出三組指標、輸出 `docs/confusion_matrices.png`；**不呼叫 `llm_advisor`，不進入告警重播** |
| `--replay-speed N` | float，覆寫 `DEFAULT_REPLAY_SPEED`，傳給 `output.replay` |
| `--no-llm` | `generate_suggestion(..., use_llm=False)` |

`--evaluate` 與其他 flag 互斥執行（`--evaluate` 優先，其餘 flag 於評估模式下忽略）。

---

## 5. 驗收標準

1. `pytest` 全綠：`tests/` 下所有測試通過；`llm_advisor` 相關測試一律 mock 掉 Ollama 呼叫（不得依賴本機真的裝有 Ollama service）。
2. `main.py` 至少以下三種呼叫方式須成功執行、無未捕捉例外：
   - `python main.py`
   - `python main.py --evaluate`
   - `python main.py --no-llm --replay-speed 0.5`
3. `--evaluate` 模式須在終端機印出 rule/ml/ensemble 三組 precision/recall/f1，並產出 `docs/confusion_matrices.png`。
4. `logs/alerts.log` 內容須與預設模式下的 CLI 輸出一致（逐行 append）。

---

## 6. 附加規則：AI 使用紀錄

coding agent 每個 session 結束時，在 `docs/ai_usage_log.md` **附加一行**（不覆寫既有內容），格式：

```
- [YYYY-MM-DD HH:MM] 完成事項: <一句話摘要> | 使用者指令數: <N> | 人工程式修改: <M>
- 註明：M 只計使用者親手編輯程式碼或手動修 bug；prompt、決策、review 不計；未被告知手動修改則記 0。
```
