# TCM-Classics-Bench — 源文本約束型測評協議

**TCM-Classics-Bench: A Source-Grounded Benchmark for Large Language Models on
Classical Chinese Medicine Texts**

面向中醫古籍大模型的源文本約束型測評基準。

本協議將通用古籍測評流程改寫為**以中醫笈成（Jicheng-TCM）收錄的繁體中醫古籍為主
語料源**，以古籍原文、底本、朝代、作者、版本品質與章節結構為約束，利用 LLM 從源
文本「倒推」生成試題。所有試題必須能**回指原文**，否則不得進入正式測評集。

---

## 一、核心原則

1. **源文本約束（source-grounded）**：每道題的答案與證據必須能在中醫笈成原文中
   找到對應片段（evidence span）。生成模型只能基於「原文 + 元數據」作答，不得
   引入現代教材知識作為事實依據，不得虛構方劑、藥物、劑量、病名。
2. **可復現（reproducible）**：固定語料快照 `book-20180111`，每條記錄寫入
   `version_commit`，使後續版本變化不破壞既有題目。
3. **繁簡雙存**：保留 `raw_text_trad`（古籍原貌、版本考證、正式答案）與
   `raw_text_simp`（適配大陸/通用中文模型），另設 `normalized_text` 供檢索去重。
4. **品質分層**：依中醫笈成「品質」字段分層；但 `book-20180111` 中約 95% 書目
   品質標為 `0%`（站方未校讀標記，而非真實低質），故 `0%`／未標注一律歸入
   `unscored` 層，核心經典即使品質低也不直接排除，改以人工校勘或跨版本交叉驗證。
5. **底本入證**：evidence 不只寫書名，必須帶 `base_text`（底本/參本）與
   章節路徑，避免「出處真實但版本不明」。

---

## 二、數據源

| 字段 | 內容 |
| --- | --- |
| 數據源名稱 | 中醫笈成 / Jicheng-TCM |
| 下載頁 | https://jicheng.tw/tcm/download.html |
| 語料快照 | https://jicheng.tw/files/jcw/book-20180111.7z （`version_commit=book-20180111`） |
| 獲取方式 | `scripts/download_corpus.sh`（curl + 7z 解壓） |
| 文本形態 | 繁體中文、古籍原文、含標點、章節標題、底本信息 |
| 授權 | 公共領域文本之編排/標點/附注以 **CC0** 釋放；站方自創文本 **CC BY 4.0**；個別頁面/多媒體需逐頁核查 |
| 適合任務 | 句讀、翻譯、術語注釋、實體關係抽取、方劑解析、方證推理、證據檢索、幻覺引用檢測 |

語料快照規模（實測 `book-20180111`）：**803 部古籍**。

### 笈成原始格式（解析依據）

每部書是一個目錄：

```
書名/index.txt                       <book> 元數據 +（有時）全文
書名/menu.txt + 書名/1.txt 2.txt …    元數據在 index.txt，正文按卷分檔
```

`<book>` 元數據塊（鍵為中文）：

```
<book>
書名=金匱要略方論
作者=張仲景
朝代=漢
年份=219
分類=金匱
品質=90%
參本=知音五版《金匱要略》
</book>
```

正文使用輕量 wiki 標記，本協議的解析器（`tcm_bench/markup.py`）按以下約定處理：

| 標記 | 含義 | 處理 |
| --- | --- | --- |
| `======標題======` | 標題，`=` 數量編碼層級（越多越淺：6=卷, 5=篇, 4=方名） | 拆分章節層級 |
| `<F> … </F>` | 方劑（方）塊 | 結構化抽取方名/組成/服法 |
| `herb<l>劑量</l>` | 藥後小字，多為劑量 | 拆出 `herb / dose / preparation` |
| `(( … ))` | 校注 | 去除（不入答案） |
| `[[book:foo:]]` | 跨書引用 | 還原為書名 |
| `**粗體**` | 強調，常為方名 | 去標記留字 |
| `<#/> <&/> <~/>` | 軟分句/分段 | 去除 |

> 注：`<l>` 劑量標記使得**仲景方書的方劑結構可被確定性抽取**（無需 LLM），是 T6
> 任務高可信度題目的來源。

---

## 三、語料分層（依笈成「分類」映射）

中醫笈成 `分類` 字段直接映射到本基準的語料層（`tcm_bench/taxonomy.py`）：

| 語料層 (level-1) | 笈成分類 (level-2) | 代表書目 | 主要任務 |
| --- | --- | --- | --- |
| 醫經類 | 內經、經論 | 黃帝內經素問、靈樞 | T2/T3/T7/T10 |
| 難經類 | 難經 | 八十一難經、難經本義 | T3/T7/T10 |
| 傷寒金匱類 | 傷寒、金匱 | 傷寒論、金匱要略方論 | T6/T8/T9/T11 |
| 溫病類 | 溫病 | 溫病條辨、溫熱論 | T7/T8/T10 |
| 本草類 | 本草、炮製 | 神農本草經、本草綱目、雷公炮炙論 | T3/T4/T6/T11 |
| 醫方類 | 方書 | 太平惠民和劑局方、醫方集解、湯頭歌訣 | T4/T5/T6/T8/T9 |
| 醫論/醫案/診法/針灸/臨證各科/養生/綜合 | … | … | T3/T4/T5/T10 等 |

> T1（句讀恢復）適用於任何帶標點的散文，故對所有分類默認可生成。

### 品質分層規則

| `品質` | 層 (`quality_tier`) | 用途 |
| --- | --- | --- |
| ≥80% | `test` | 可進入正式測試集 |
| 50–79% | `candidate` | 候選池，需專家複核 |
| <50%（非0） | `explore` | 訓練/探索用 |
| 0%／未標注 | `unscored` | 人工抽檢後再決定（**不等於低質**） |

---

## 四、數據入庫 Schema

完整定義見 `schemas/corpus_record.schema.json`。每條記錄（一個 passage）：

```json
{
  "source_id": "jicheng_tcm",
  "book_id": "金匱要略方論",
  "book_title_trad": "金匱要略方論",
  "book_title_simp": "金匮要略方论",
  "author": "張仲景",
  "dynasty": "漢",
  "year": "219",
  "category_level_1": "傷寒金匱類",
  "category_level_2": "金匱",
  "base_text": "知音五版《金匱要略》",
  "quality_score_source": "90%",
  "quality_tier": "test",
  "heading_path": ["金匱要略方論", "臟腑經絡先後病脈證第一"],
  "chapter": "金匱要略方論",
  "section_title": "臟腑經絡先後病脈證第一",
  "raw_text_trad": "問曰：上工治未病，何也？…",
  "raw_text_simp": "问曰：上工治未病，何也？…",
  "normalized_text": "問曰上工治未病何也…",
  "formulas": [{"formula_name": "…", "ingredients": [{"herb": "…", "dose": "…", "preparation": "…"}]}],
  "candidate_tasks": ["T1", "T2", "T3", "T6", "…"],
  "punctuation_status": "source_punctuated",
  "source_url": "https://jicheng.tw/tcm/book/金匱要略方論/",
  "source_license": "CC0 for public-domain editing; verify per page",
  "version_commit": "book-20180111",
  "ingestion_date": "2026-06-26"
}
```

新增關鍵字段：`book_title_trad/simp`（繁簡雙存）、`base_text`（底本考據）、
`quality_score_source` + `quality_tier`（質量分層）、`version_commit`（可復現）、
`heading_path` + `formulas`（章節/方劑自動切分結果）。

---

## 五、生成流程

```
中醫笈成語料快照 (book-20180111) 固定
  ↓  download_corpus.sh
典籍目錄與 <book> 元數據解析            tcm_bench/parsing.py
  ↓
按 分類/朝代/底本/品質 分層抽樣          tcm_bench/taxonomy.py
  ↓
繁簡轉換與異體字映射                     opencc t2s
  ↓
章節/條文/方劑/藥物段落切分             markup.py（= 層級 + <F>/<l> 塊）
  ↓
任務適配分類（候選任務）                 candidate_tasks
  ↓
LLM 基於源文本倒推候選題                 prompts.py + generate.py
  ↓
證據回指校驗（自動）                     validate.py
  ↓
跨模型審核 → 專家審核 → 多模型預測試篩題
```

確定性任務（T1 句讀、T6 `<F>` 方劑解析）無需 LLM，源約束由構造保證；其餘任務走
LLM 路徑並必須通過 `validate.py`。

**LLM 接入（provider-agnostic）**：`tcm_bench/llm.py` 提供統一 `complete()` 介面，
內建四個 provider —— `anthropic`、`azure`（Azure OpenAI）、`poe`（OpenAI 相容端點）、
`litellm`（通用路由，覆蓋 100+ 模型）。命令行用 `--provider/--model` 選擇，跨模型
審核（cross-model review）即由不同 provider 生成/互審實現。

**簡易模式（`simple`）**：一條命令產出均衡且已校驗的 N 道測試題（默認 5000）。默認僅
用確定性生成器（T1+T6），無需 API key，數秒完成；對 (任務, 書目) 做輪轉均衡抽樣，
覆蓋全部 pilot 書目。加 `--llm` 可混入模型生成任務（每題一次 API 調用，命中 N 即停）。
注意：`--llm` 僅在 `--tasks` 含**非確定性任務**（T2–T12）時才會真正調用模型；
`T1/T6` 為確定性任務，`--tasks T1 T6 --llm` 不會發起任何 API 調用（CLI 會警告）。

**加強難度（`--difficulty Medium/Hard/Expert`，默認 Hard）**：LLM 生成的題目被導向
需要多步推理、辨析相近概念、病機—治法—方藥綜合判斷的**難題**，避免可直接複製原文作答
的表層題。其中 **T7–T9、T11、T12** 以**四選一單選題**形式產出，干擾項須高度迷惑且
每項附**基於原文**的排除理由；排除理由若需外部教材知識則標 `external_required` 並剔除。

---

## 六、任務體系 T1–T12

| 任務 | 名稱 | 數據來源 | 生成方式 |
| --- | --- | --- | --- |
| T1 | 標點/句讀恢復 | 帶標點原文 | 刪標點反推（確定性） |
| T2 | 文白翻譯 | 醫經/傷寒/金匱/醫案 | LLM 生成，專家校正 |
| T3 | 術語注釋 | 內經/難經/傷寒/金匱 | 抽取術語 |
| T4 | 實體識別 | 全部類別 | 病/症/證/治/方/藥/量 |
| T5 | 關係抽取 | 方書/本草/醫案/經典 | 方-藥、病-症、證-治、方-證 |
| T6 | 方劑結構解析 | 醫方/傷寒/金匱 | `<F>` 塊確定性 + LLM |
| T7 | 理論分類 | 內經/難經/傷寒/溫病 | 六經/臟腑/衛氣營血/三焦 |
| T8 | 方證對應 | 傷寒/金匱/醫案 | 症狀→病機→治法→方劑 |
| T9 | 類方鑑別 | 傷寒注本/方論/方歌 | 相似方證干擾項 |
| T10 | 證據溯源問答 | 全庫檢索 | 問題→原文證據→出處 |
| T11 | 安全禁忌判斷 | 本草/炮製/醫方 | 毒性/禁忌/劑量/炮製 |
| T12 | 幻覺引用檢測 | 全庫 | 真引用/錯出處/篡改/偽造 |

---

## 七、Pilot 語料組合

不一開始全庫跑，先做高品質 pilot（`tcm_bench/taxonomy.PILOT_CORPORA`）：

- **Pilot 1 — 經典理論與基礎古文**：黃帝內經素問、靈樞、難經 → T1/T2/T3/T7。
- **Pilot 2 — 仲景方證推理**：傷寒論（宋本）、金匱要略方論、金匱要略條文版、
  傷寒論類方、長沙方歌括、金匱方歌括 → T6/T8/T9，含確定性方劑解析。
- **Pilot 3 — 方藥結構化**：太平惠民和劑局方、醫方集解、神農本草經、本草綱目、
  本草備要、雷公炮炙論、湯頭歌訣 → T4/T6/T11。

---

## 八、生成 Prompt（源約束版）

完整模板見 `tcm_bench/prompts.py`。共同 system 約束：

1. 只能基於給定原文、書名、篇章、底本、朝代、作者與品質字段判斷。
2. 不得引入現代教材知識作為事實依據。
3. 不得補充原文中沒有出現的方劑、藥物、劑量、病名。
4. 凡推理須標 `inference_level = direct / implicit / external_required`。
5. `external_required` 不得進入正式 benchmark。
6. 涉及犀角、麝香、朱砂、雄黃、烏頭、附子等須在 `safety_note` 標「僅作古籍理解
   測評，不構成用藥建議」。
7. 答案必須可由 `evidence_span` 回指原文，輸出合法 JSON。

任務路由 prompt（`task_router_prompt`）輸出 `suitable_tasks / detected_entities /
evidence_spans / difficulty / quality_warning`；方劑解析 prompt
（`formula_parse_prompt`）強制 方名/主治/組成/劑量/炮製/製法/服法 來自原文，缺失填
`null`，並保留「童子小便」「生薑自然汁」「研飛」「酒煮」「醋淬」等古法信息。

---

## 九、證據回指校驗（自動，`tcm_bench/validate.py`）

```python
def validate_item(item, source_text):
    # 1. evidence.book_title_trad 非空、source_id == jicheng_tcm
    # 2. 每個 evidence span 必須回指 source_text（容標點差異）
    # 3. inference_level == external_required → 逐出正式集
    # 4. T6：formula_name、herb、dose 必須出現於原文
    # 5. 毒性/禁限藥材出現而 safety_note 為空 → 警告
```

選擇題（T9 類方鑑別）另用 `validate_mcq`：正確方劑須由當前條文直接支持；每個干擾
項須有明確排除理由；排除理由若需外部教材知識，降級為訓練題；若兩方均可成立則刪題。

---

## 十、NER 測試子集（T4，確定性）

利用 `<F>` 方劑塊已抽出的 方名/中藥/劑量/炮製 span，為每個方劑段落生成一道 NER
題（`generate_ner`）：實體即原文逐字片段，gold 標註天然源約束，無需 LLM。輸出於
`data/bench_ner/`，每題經 `validate.py` 校驗每個實體出現於原文。

## 十一、模型評測（`tcm_bench evaluate`）

`tcm_bench/evaluate.py` 在基準上評測任意模型（provider：anthropic/azure/poe/litellm）。

- **Prompt 模式**：`zero_shot` / `few_shot`（同類示例前置）/ `cot`（思維鏈）。
- **自動評分**：單選題（T7–T9/T11/T12）→ accuracy；NER（T4）→ F1；方劑（T6）→ 藥味 F1；
  句讀（T1）→ 句讀邊界 F1。開放題（T2/T3/T5/T10）標為不可自動評分，排除於總分。
- **並發 / 斷點續跑 / 實時寫入 / 進度+ETA**；輸出每任務分數與總分（macro）。

## 十二、命名

- 英文：**TCM-Classics-Bench: A Source-Grounded Benchmark for Large Language
  Models on Classical Chinese Medicine Texts**
- 中文：**TCM-Classics-Bench：面向中醫古籍大模型的源文本約束型測評基準**
- 子集標識：`TCM-Classics-Bench-Jicheng v1`
