# AI 工具使用紀錄

本檔為開發過程的原始流程紀錄，依時間遞增排列。內容為當下實況，包含失敗與誤判，未作事後美化。

## 工具分工

| 工具 | 角色 |
|---|---|
| Claude Sonnet（claude.ai 免費版） | 主要 brainstorming 與技術諮詢 |
| Claude Code + Ollama `gemma4:31b-cloud` | Coding agent，實作各模組與測試 |
| Claude Code + Ollama `nemotron-3-super:cloud` | Code review |
| Gemini（免費網頁版） | 簡易查找與規格交叉審計（cross-audit） |
| Perplexity | 免費工具調查與即時資訊查找 |
| 本機 Ollama `gemma4:e4b` | 系統執行期生成維修建議 |

---

## 逐日紀錄

### 2026-07-19

- **審計實驗（第一輪，工具未載明）**：對 AI 生成規格跑批判模式審計。已知 2 處矛盾抓到 1（CRITICAL，後果分析精確）、漏 1 且誤判「驗證通過」；另自行發現 2 個真問題（經人工判定採納）。結論：AI 審計有效但不完備，需人工持答案把關。| 人工程式修改: 0

- **工具**: Gemini 免費網頁版 | **任務**: 對修正前原版 CLAUDE.md 跑批判模式審計（cross-audit）| **結果**: 已知 2 缺陷抓到 1（combine 矛盾，修法與已採納方案一致）；calibrate_threshold 仍漏（其報告第 4 點直接分析該函式參數仍未發現）。另提 GEN_RANGES（採納）、join 冗餘與模板進 config（駁回）。**註**: 疑與前次審計同對話，非完全獨立。| 人工程式修改: 0

- **方法教訓**: 第一次 cross-audit 誤貼修正後版本，由報告內容比對（引用僅修正版才有的 MIN_REQUIRED_SAMPLES、未提已修矛盾）判定無效後重做。教訓: 審計實驗必須鎖定受測版本。

- **階段總結**: calibrate_threshold 缺陷經 gemma×2、Gemini×2 四輪 AI 審計皆未命中，僅人工介面追蹤發現——AI 審計對顯性矛盾有效、對隱性不可實作性不完備，人工把關不可省。

- **階段總結**: 隨後改用 Claude Sonnet，成功捕捉 calibrate_threshold 缺陷，並同時發現 CLAUDE.md 中參數未加 `_scaled` 後綴的缺失。

- [2026-07-19 03:04] 完成事項: 審計並修正 CLAUDE.md 規格漏洞，實作 config.py 及測試 | 使用者指令數: 8 | 人工程式修改: 0

- [2026-07-19 22:45] 完成事項: 實作從 generate_data 到 output 的所有核心模組及其對應測試 | 使用者指令數: 11 | 人工程式修改: 0

### 2026-07-20

- [2026-07-20] 完成事項: 實作 agent.py 與 main.py；修正 evaluate.py 於單一類別下 confusion_matrix 的 IndexError；重寫資料生成邏輯改為單一閾值治理（正常區間外即異常，不留生成斷層）；調整感測值精度（temp 一位、pressure/vibration 兩位）；output.py 改用 rich 呈現分級告警 | 使用者指令數: 未計數 | 人工程式修改: 0

### 2026-07-21

- [2026-07-21] 完成事項: 以 nemotron-3-super:cloud 執行 code review 並逐條人工查核（12 項僅採納 3 項，查出 4 項事實錯誤指控）；產出 architecture / workflow 圖說；撰寫技術報告 report.md 並建立 build_report.py 轉檔工具 | 使用者指令數: 79 | 人工程式修改: 0

---

## 說明

「人工程式修改」僅計使用者親手編輯程式碼或手動修 bug；prompt、決策、review 不計。全程計數為 0 —— 所有程式碼皆由 AI 生成，人工投入集中於規格設計、審計驗證與決策把關。
