# IOM Devex-PRIMA Analyzer

A Streamlit application for IOM Resource Mobilization that processes Devex funding opportunities, matches them to PRIMA projects, and provides AI-powered relevance explanations.

## Features

### 📊 Tab 1: Overview Dashboard
- Summary metrics (total opportunities, PRIMA projects, open status)
- Interactive charts by Region and Thematic Area
- Top donors list
- **Download regional Excel files** (Africa, Americas, Asia-Pacific, Europe, MENA, Global)

### 🔍 Tab 2: Browse & Select
- Filter opportunities by **Region**, **Thematic Area**, and **Status**
- Search in title/description
- View **Top 5** matching opportunities
- Select any opportunity for detailed AI analysis

### 💬 Tab 3: Ask AI
- Chat interface to ask about opportunities
- Get AI-powered explanations of why opportunities match IOM's work
- Compare Devex opportunities to similar PRIMA projects

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
│     ├── Filter by region/theme/status                                       │
│     ├── View Top 5 results                                                  │
│     └── Select one for deep analysis                                        │
│                                                                             │
│  4. AI EXPLANATION (on-demand)                                              │
│     ├── Compare to PRIMA projects                                           │
│     ├── Find similar past IOM work                                          │
│     ├── Calculate similarity scores                                         │
│     └── Generate recommendation                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
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

### 🤖 Analysis
- **Geographic Match**: IOM has active operations in Sudan
- **Crisis Context**: Both involve conflict situations
- **Migration Focus**: Shared focus on displacement

### 💡 Recommendation
**Action: PURSUE** - This opportunity strongly aligns with IOM's mandate.
```

## Azure OpenAI Configuration (Optional)

Enable AI for richer explanations:

1. Check "🤖 Enable AI Explanations" in sidebar
2. Enter your Azure OpenAI credentials:
   - **Endpoint**: `https://your-resource.openai.azure.com/`
   - **API Key**: From Azure Portal
   - **Deployment Name**: Your model deployment name
   - **API Version**: `2024-08-01-preview`

Without Azure OpenAI, the app uses keyword-based matching (still works well!).

## Regional Excel Output

Each regional file contains:

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

## Project Structure

```
devex-prima-analyzer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # This file

```

## Requirements

- Python 3.8+
- streamlit >= 1.28.0
- pandas >= 2.0.0
- openpyxl >= 3.1.0
- openai >= 1.0.0 (for Azure OpenAI features)

## License

Internal IOM tool - Not for public distribution.


