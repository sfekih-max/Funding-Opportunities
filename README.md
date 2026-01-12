# IOM Devex-PRIMA Analyzer 

A Streamlit application for IOM Resource Mobilization that processes Devex funding opportunities, matches them to PRIMA projects, automates Funding Highlights bulletins, and provides AI-powered relevance explanations with built-in hallucination safeguards.

## Features

### 📊 Tab 1: Overview Dashboard
- Summary metrics (total opportunities, PRIMA projects, open status)
- Interactive charts by Region and Thematic Area
- Top donors list
- **Download regional Excel files** (Africa, Americas, Asia-Pacific, Europe, MENA, Global)

### 🔍 Tab 2: Browse & Select
- Filter opportunities by **Region**, **Thematic Area**, **Type**, and **Status**
- Search in title/description
- View **Top 5** matching opportunities
- Select any opportunity for detailed AI analysis
- **Add opportunities to export list**
- **Export selected matches to Excel** with PRIMA project details

### 📰 Tab 3: Funding Highlights
- **Auto-fetch** donor news via Azure OpenAI Web Search
- **Manual paste** option for news items
- Relevance scoring based on migration keywords
- Duplicate detection across bulletins
- Focal point assignment
- **Source URL extraction** and clickable links
- **AI hallucination detection** and verification warnings
- Export to Text or Excel format

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              WORKFLOW                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. UPLOAD FILES                                                            │
│     ├── Devex XML export (funding opportunities)                            │
│     └── PRIMA Excel/CSV (IOM project database)                              │
│                                                                             │
│  2. FAST ANALYSIS (< 1 second)                                              │
│     ├── Parse 1000+ opportunities                                           │
│     ├── Classify by thematic area (keyword-based)                           │
│     └── Group by IOM region                                                 │
│                                                                             │
│  3. BROWSE & SELECT                                                         │
│     ├── Filter by region/theme/type/status                                  │
│     ├── View Top 5 results                                                  │
│     ├── Select one for deep analysis                                        │
│     └── Add to export list for Excel download                               │
│                                                                             │
│  4. AI EXPLANATION (on-demand)                                              │
│     ├── Compare to PRIMA projects                                           │
│     ├── Find similar past IOM work                                          │
│     ├── Calculate similarity scores                                         │
│     └── Generate recommendation                                             │
│                                                                             │
│  5. FUNDING HIGHLIGHTS                                                      │
│     ├── Auto-fetch or manual paste donor news                               │
│     ├── Score relevance to migration/IOM                                    │
│     ├── Detect duplicates from previous bulletins                           │
│     ├── Flag hallucinations & verify sources                                │
│     └── Export for RM Bulletin                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run application.py
```

## Input Files

### Devex XML Export
Export from [Devex Funding](https://www.devex.com/funding) containing:
- Opportunity ID, Title, Description
- Countries, Regions, Donors
- Type, Status, Deadline
- Devex URL

### PRIMA Data (Excel/CSV)
IOM project database with columns:
- `PRIMA / Project ID` or `Project ID`
- `Title`
- `Reporting Area`
- `Project Summary` or `Summary`
- `Benefiting Country`
- `Contracting Funding Source`
- `Budget Amount In Project Currency`

## Thematic Classification

Opportunities are automatically classified into 10 thematic areas:

| Theme | Keywords |
|-------|----------|
| Health | health, medical, disease, mhpss, psychosocial |
| Border & Immigration | border, visa, passport, identity, biometric |
| Protection & Anti-Trafficking | trafficking, smuggling, protection, gbv |
| Emergency & Humanitarian | emergency, humanitarian, disaster, crisis, displaced |
| Return & Reintegration | return, reintegration, returnee, repatriation |
| Resettlement | resettlement, relocation, refugee, asylum |
| Labor & Diaspora | labor, employment, diaspora, remittance, skills |
| Climate & Environment | climate, environment, drought, flood, resilience |
| Policy & Research | policy, research, data, study, assessment |
| Capacity Building | capacity, training, workshop, technical assistance |

## AI-Powered Analysis

When you select an opportunity, the app provides:

```
# 🎯 Opportunity Analysis

## 📋 DEVEX OPPORTUNITY

| Field | Value |
|-------|-------|
| ID | 51909 |
| Title | Emergency Health Response for Displaced Populations |
| Type | grant |
| Region | MENA |
| Countries | Sudan |
| Donors | AICS |
| Thematic Area | Health |
| Deadline | 2026-01-25 |

### Description
[Full opportunity description]

---

## 🔗 MATCHING PRIMA PROJECTS

### Match 1: Community Stabilization in River Nile
| Field | Value |
|-------|-------|
| Project ID | CH1NP0559 |
| Reporting Area | 1104 # Community Stabilization |
| Country | Sudan |
| Budget | $200,000 |
| Similarity Score | **20%** |

#### PRIMA Project Summary
[Full project summary]

#### 🔑 Common Themes
`conflict • displacement • humanitarian • protection`

---

## 📊 RELEVANCE ASSESSMENT

### ✅ HIGH RELEVANCE (20%)

Strong alignment with IOM's past work.

### 💡 Recommendation
**Action: PURSUE** - This opportunity strongly aligns with IOM's mandate.
```

## Funding Highlights: Hallucination Detection

The system automatically flags unreliable AI-generated content:

### Detected Patterns

| Warning Type | Trigger Phrases | Risk Level |
|--------------|-----------------|------------|
| No source URL | Missing verification link | ⚠️ Medium |
| Speculative language | "expected to", "likely to", "may affect", "could impact" | ⚠️ Medium |
| Pending details | "details pending", "details not yet announced" | ⚠️ Medium |
| Inferred context | "amid growing domestic", "amid political constraints" | ⚠️ Medium |
| Uncertain attribution | "reportedly planning", "said to be", "believed to" | ⚠️ Medium |
| Specific claims without URL | Impact statements without source link | 🚨 High |

### Example: Good Output ✅
```
Germany: On November 28, 2025, the German parliament approved the 
federal budget for 2026, allocating EUR10.05 billion (US$11.6 billion) 
to BMZ, a decrease of EUR251 million compared to 2025. 
Source: Bundestag (https://bundestag.de/...)
```
✅ Specific figures | ✅ Exact date | ✅ Source URL | ✅ No speculation

### Example: Flagged Output ⚠️
```
Belgium: Belgium is among several EU countries planning development 
aid cuts in 2026, amid growing domestic constraints. While details 
are pending, reductions are expected to affect funding to UN partners.
Source: Web Search
```
| Issue | Flag |
|-------|------|
| No source URL | ⚠️ No source URL - cannot verify claims |
| "details are pending" | ⚠️ Speculative language detected |
| "expected to affect" | ⚠️ Speculative language detected |
| "amid growing domestic" | ⚠️ Contains inferred context |

### Visual Indicators
- **🔍 VERIFY** badge on items needing fact-checking
- **⚠️ Verification Warnings** listed in item details
- **Export warning banner** summarizing items to verify
- **Excel columns**: `Verified` and `Confidence Warnings`

## Azure OpenAI Configuration (Optional)

### For AI Explanations (Browse & Select)
1. Check "🤖 Enable AI Explanations" in sidebar
2. Enter your Azure OpenAI credentials:
   - **Endpoint**: `https://your-resource.openai.azure.com/`
   - **API Key**: From Azure Portal
   - **Deployment Name**: Your model deployment (e.g., `gpt-4o-mini`)
   - **API Version**: `2024-08-01-preview`

### For Web Search (Funding Highlights)
- **Model**: `gpt-4.1` or later required
- **Feature**: `web_search_preview` must be enabled
- Contact Azure admin if web search is blocked

Without Azure OpenAI, the app uses keyword-based matching (still works well!).

## Export Formats

### Matches Excel (Browse & Select)

| Column | Description |
|--------|-------------|
| Opportunity ID | Devex opportunity ID |
| Opportunity Title | Full title |
| Type | grant, tender, program, etc. |
| Status | open, closed, forecast |
| Thematic Area | Classified theme |
| Region | IOM region(s) |
| Countries | Target countries |
| Donors | Funding sources |
| Deadline | Application deadline |
| Devex URL | Link to full opportunity |
| Relevance Level | HIGH, MEDIUM, LOW |
| Match Rank | 1, 2, or 3 |
| PRIMA Project ID | Matched project ID |
| PRIMA Title | Project title |
| PRIMA Reporting Area | Thematic area |
| PRIMA Country | Implementation country |
| PRIMA Budget | Project budget |
| Match Score | Similarity percentage |
| Common Keywords | Shared terms |

### Funding Highlights Excel

| Column | Description |
|--------|-------------|
| Selected | ✓ if included in bulletin |
| Donor | Country/organization |
| Region | Geographic region |
| Category | News type (Budget Announcement, Political Change, etc.) |
| Title | Headline |
| Summary | News summary |
| Relevance Score | Migration relevance % |
| Keywords Found | Matched migration terms |
| Source | Publication name |
| Source URL | Link to original article |
| Verified | ✅ or ⚠️ Verify |
| Confidence Warnings | Detected hallucination flags |
| Date | Publication date |
| Duplicate | Yes/No |
| Focal Point | Assigned reviewer |

### Regional Excel (Overview)

| Column | Description |
|--------|-------------|
| ID | Devex opportunity ID |
| Title | Opportunity title |
| Type | grant, tender, program, etc. |
| Status | open, closed, forecast |
| Thematic Area | Classified theme |
| Region | IOM region(s) |
| Countries | Target countries |
| Donors | Funding sources |
| Deadline | Application deadline |
| Devex URL | Link to full opportunity |

## Supported Donors (Funding Highlights)

### Bilateral (21)
Australia, Belgium, Canada, Finland, France, Germany, Italy, Japan, Kuwait, Netherlands, Norway, Qatar, Republic of Ireland, Republic of Korea, Saudi Arabia, Spain, Sweden, Switzerland, United Arab Emirates, United Kingdom, United States

### Multilateral (6)
European Union, World Bank, African Development Bank, Asian Development Bank, Inter-American Development Bank, Green Climate Fund

## News Categories

| Category | Description |
|----------|-------------|
| Budget/Funding Announcement | ODA budget changes, new allocations |
| Political Change/Cabinet Reshuffle | Minister changes, elections |
| Policy Update | New laws, regulations, strategies |
| Multilateral Commitment | UN, World Bank pledges |
| Bilateral Agreement | Country-to-country agreements |
| Election Result | Election outcomes affecting aid |
| Strategy/Framework Release | New development strategies |
| Humanitarian Response | Crisis response funding |

## Project Structure

```
devex-prima-analyzer/
├── app.py              # Main Streamlit application (2188 lines)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
openpyxl>=3.1.0
openai>=1.0.0
requests>=2.31.0
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| 401 Unauthorized | Check Azure endpoint URL and API key |
| 404 Not Found | Verify deployment name matches exactly |
| Web search blocked | Contact Azure admin to enable `web_search_preview` |
| No news items parsed | Check raw response in expander; try manual paste |
| All items flagged | Normal for some searches; manually verify critical items |
| Timeout error | Web search can take 2+ minutes; try again |

## Version History

| Version | Changes |
|---------|---------|
| v4.0 | Hallucination safeguards, source URL tracking, matches export |
| v3.0 | Funding Highlights module, web search integration |
| v2.0 | PRIMA matching, AI explanations |
| v1.0 | Initial Devex XML parser |

## License

Internal IOM tool - Not for public distribution.
