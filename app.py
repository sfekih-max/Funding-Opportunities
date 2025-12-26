"""
Devex Funding Opportunities Processor v3
=========================================
- Fast initial analysis (no matching)
- Browse opportunities with filters
- AI explanation on-demand (when user selects one)
"""

import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
from io import BytesIO
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from openai import AzureOpenAI

# =============================================================================
# CONFIGURATION
# =============================================================================

REGION_MAPPING = {
    "Sub-Saharan Africa": "Africa", "West Africa": "Africa", "East Africa": "Africa",
    "Eastern Africa": "Africa", "Central Africa": "Africa", "Southern Africa": "Africa",
    "North America": "Americas", "Central America": "Americas", "South America": "Americas",
    "Latin America": "Americas", "Latin America and Caribbean": "Americas", "Caribbean": "Americas",
    "East Asia": "Asia_Pacific", "East Asia and Pacific": "Asia_Pacific", "Southeast Asia": "Asia_Pacific",
    "South Asia": "Asia_Pacific", "Central Asia": "Asia_Pacific", "Pacific": "Asia_Pacific", "Oceania": "Asia_Pacific",
    "Eastern Europe": "Europe", "Western Europe": "Europe", "Southern Europe": "Europe",
    "Northern Europe": "Europe", "Balkans": "Europe", "Europe": "Europe",
    "Middle East": "MENA", "North Africa": "MENA", "North Africa and Middle East": "MENA", "Gulf States": "MENA",
    "Global": "Global", "Worldwide": "Global", "Multiple Regions": "Multi_Region",
}

# Thematic keywords for quick classification (no AI needed)
THEMATIC_KEYWORDS = {
    "Health": ["health", "medical", "disease", "hospital", "clinic", "mhpss", "psychosocial", "mental health"],
    "Border & Immigration": ["border", "visa", "passport", "identity", "biometric", "immigration", "customs"],
    "Protection & Anti-Trafficking": ["trafficking", "smuggling", "protection", "gbv", "exploitation", "victim"],
    "Emergency & Humanitarian": ["emergency", "humanitarian", "disaster", "crisis", "relief", "shelter", "idp", "displaced"],
    "Return & Reintegration": ["return", "reintegration", "returnee", "repatriation", "avrr"],
    "Resettlement": ["resettlement", "relocation", "refugee", "asylum"],
    "Labor & Diaspora": ["labor", "employment", "diaspora", "remittance", "skills", "tvet", "vocational"],
    "Climate & Environment": ["climate", "environment", "drought", "flood", "resilience"],
    "Policy & Research": ["policy", "research", "data", "study", "assessment", "governance"],
    "Capacity Building": ["capacity", "training", "workshop", "technical assistance"],
}

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PRIMAProject:
    project_id: str
    title: str
    reporting_area: str
    summary: str
    country: str
    funding_source: str
    budget: float
    project_type: str

@dataclass
class FundingOpportunity:
    id: int
    title: str
    description: str
    type: str
    status: str
    countries: List[str]
    regions: List[str]
    iom_regions: List[str]
    donors: List[str]
    deadline: Optional[str]
    devex_url: str
    value: Optional[str]
    # Quick classification (no AI)
    thematic_area: str = "Other"


# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def parse_devex_xml(xml_content: bytes) -> List[FundingOpportunity]:
    """Parse Devex XML - fast, no matching."""
    root = ET.fromstring(xml_content)
    opportunities = []
    
    for report in root.findall('.//devex-funding-report'):
        try:
            opp_id = int(report.findtext('id', '0'))
            title = report.findtext('title', '')
            description = report.findtext('description', '') or ''
            
            # Clean HTML
            description = re.sub(r'<[^>]+>', ' ', description)
            description = re.sub(r'\s+', ' ', description).strip()
            
            # Extract regions
            regions = []
            regions_elem = report.find('regions')
            if regions_elem is not None:
                for elem in regions_elem.iter():
                    if elem.tag in ('n', 'name') and elem.text and elem.text.strip():
                        regions.append(elem.text.strip())
            
            iom_regions = list(set(REGION_MAPPING.get(r, "Other") for r in regions)) or ["Global"]
            
            # Extract countries
            countries = []
            countries_elem = report.find('countries')
            if countries_elem is not None:
                for elem in countries_elem.iter():
                    if elem.tag in ('n', 'name') and elem.text and elem.text.strip():
                        countries.append(elem.text.strip())
            
            # Extract donors
            donors = []
            donors_elem = report.find('donors')
            if donors_elem is not None:
                for elem in donors_elem.iter():
                    if elem.tag in ('n', 'name') and elem.text and elem.text.strip():
                        donors.append(elem.text.strip())
            
            # Quick thematic classification (keyword-based, very fast)
            text_lower = f"{title} {description}".lower()
            thematic_area = "Other"
            max_matches = 0
            for theme, keywords in THEMATIC_KEYWORDS.items():
                matches = sum(1 for kw in keywords if kw in text_lower)
                if matches > max_matches:
                    max_matches = matches
                    thematic_area = theme
            
            opportunities.append(FundingOpportunity(
                id=opp_id,
                title=title,
                description=description[:2000],
                type=report.findtext('type', ''),
                status=report.findtext('status', ''),
                countries=countries,
                regions=regions,
                iom_regions=iom_regions,
                donors=donors,
                deadline=report.findtext('deadline'),
                devex_url=report.findtext('devex-url', ''),
                value=report.findtext('value'),
                thematic_area=thematic_area
            ))
        except:
            continue
    
    return opportunities


def parse_prima_data(file) -> List[PRIMAProject]:
    """Parse PRIMA data."""
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    
    projects = []
    for _, row in df.iterrows():
        # Get project ID
        project_id = ''
        for col in ['PRIMA / Project ID', 'Project ID', 'PRIMA']:
            if col in df.columns and pd.notna(row.get(col)):
                project_id = str(row.get(col, ''))
                break
        if not project_id:
            continue
        
        # Get summary
        summary = ''
        for col in ['Project Summary', 'Summary', 'Description']:
            if col in df.columns and pd.notna(row.get(col)):
                summary = str(row.get(col, ''))
                break
        
        # Get budget
        budget = 0.0
        for col in ['Budget Amount In Project Currency', 'Budget', 'Total Amount']:
            if col in df.columns and pd.notna(row.get(col)):
                try:
                    budget_str = str(row.get(col, '0')).replace(',', '').replace('$', '')
                    budget = float(budget_str)
                except:
                    pass
                break
        
        projects.append(PRIMAProject(
            project_id=project_id,
            title=str(row.get('Title', '')),
            reporting_area=str(row.get('Reporting Area', '')),
            summary=summary[:1500],
            country=str(row.get('Benefiting Country', '')),
            funding_source=str(row.get('Contracting Funding Source', '')),
            budget=budget,
            project_type=str(row.get('Project Type', ''))
        ))
    
    return projects


# =============================================================================
# ON-DEMAND AI EXPLANATION
# =============================================================================

def explain_opportunity(
    opp: FundingOpportunity,
    prima_projects: List[PRIMAProject],
    client: AzureOpenAI = None,
    deployment: str = None
) -> str:
    """Generate detailed AI explanation for ONE opportunity - called on demand."""
    
    # Filter PRIMA projects with summaries
    prima_with_summaries = [p for p in prima_projects if p.summary and len(p.summary) > 50]
    
    if client and deployment:
        # Use AI for rich explanation
        return explain_with_ai(opp, prima_with_summaries[:30], client, deployment)
    else:
        # Use keyword matching fallback
        return explain_with_keywords(opp, prima_with_summaries)


def explain_with_keywords(opp: FundingOpportunity, prima_projects: List[PRIMAProject]) -> str:
    """Detailed keyword-based explanation with side-by-side comparison."""
    
    opp_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', f"{opp.title} {opp.description}".lower()))
    stopwords = {'this', 'that', 'with', 'from', 'will', 'their', 'which', 'through', 'project', 'about', 'have', 'been', 'also', 'more', 'other'}
    opp_words -= stopwords
    
    matches = []
    for prima in prima_projects:
        if not prima.summary or len(prima.summary) < 50:
            continue
            
        prima_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', f"{prima.title} {prima.summary}".lower()))
        prima_words -= stopwords
        
        common = opp_words & prima_words
        if len(common) >= 3:
            similarity = len(common) / len(opp_words | prima_words)
            
            # Boost score for important migration-related terms
            important_terms = {'migration', 'migrant', 'refugee', 'displacement', 'displaced', 
                             'humanitarian', 'protection', 'trafficking', 'returnee', 'idp',
                             'border', 'asylum', 'resettlement', 'reintegration'}
            important_matches = common & important_terms
            boosted_similarity = similarity + (len(important_matches) * 0.05)
            
            matches.append((prima, boosted_similarity, list(common)))
    
    matches.sort(key=lambda x: -x[1])
    top_matches = matches[:3]
    
    # Calculate overall relevance
    if top_matches:
        avg_score = sum(s for _, s, _ in top_matches) / len(top_matches)
    else:
        avg_score = 0
    
    # Determine relevance level
    if avg_score >= 0.15:
        relevance_emoji = "✅"
        relevance_level = "HIGH"
        relevance_color = "green"
        relevance_explanation = "Strong alignment with IOM's past work. IOM has significant experience in similar projects."
    elif avg_score >= 0.08:
        relevance_emoji = "🟡"
        relevance_level = "MEDIUM"
        relevance_color = "orange"
        relevance_explanation = "Moderate alignment. IOM has some related experience that could be leveraged."
    else:
        relevance_emoji = "🔴"
        relevance_level = "LOW"
        relevance_color = "red"
        relevance_explanation = "Limited direct match with past projects. May require new capacity or partnerships."
    
    # Build response
    response = f"""# 🎯 Opportunity Analysis

---

## 📋 DEVEX OPPORTUNITY

| Field | Value |
|-------|-------|
| **ID** | {opp.id} |
| **Title** | {opp.title} |
| **Type** | {opp.type} |
| **Status** | {opp.status} |
| **Region** | {', '.join(opp.iom_regions)} |
| **Countries** | {', '.join(opp.countries[:5])} |
| **Donors** | {', '.join(opp.donors[:3])} |
| **Thematic Area** | {opp.thematic_area} |
| **Deadline** | {opp.deadline or 'Not specified'} |

### Description
{opp.description}

---

## 🔗 MATCHING PRIMA PROJECTS

"""
    
    if top_matches:
        for i, (prima, score, common_words) in enumerate(top_matches, 1):
            # Get top meaningful common words (exclude very generic ones)
            meaningful_common = [w for w in common_words if len(w) > 4][:8]
            
            response += f"""### Match {i}: {prima.title}

| Field | Value |
|-------|-------|
| **Project ID** | {prima.project_id} |
| **Reporting Area** | {prima.reporting_area} |
| **Country** | {prima.country} |
| **Funding Source** | {prima.funding_source[:60]}{'...' if len(prima.funding_source) > 60 else ''} |
| **Budget** | ${prima.budget:,.0f} |
| **Similarity Score** | **{score:.0%}** |

#### PRIMA Project Summary
{prima.summary}

#### 🔑 Common Themes
`{' • '.join(meaningful_common)}`

---

"""
        
        # AI-like explanation based on keyword analysis
        response += f"""## 📊 RELEVANCE ASSESSMENT

### {relevance_emoji} **{relevance_level} RELEVANCE** ({avg_score:.0%})

{relevance_explanation}

### 🤖 Analysis

"""
        # Generate intelligent explanation based on matches
        all_common = set()
        for _, _, common in top_matches:
            all_common.update(common)
        
        # Categorize the common themes
        geo_terms = all_common & {'sudan', 'africa', 'ethiopia', 'kenya', 'uganda', 'somalia', 'yemen', 'syria', 'afghanistan', 'ukraine', 'libya', 'niger', 'chad', 'mali'}
        crisis_terms = all_common & {'conflict', 'crisis', 'emergency', 'disaster', 'war', 'violence', 'armed'}
        migration_terms = all_common & {'displacement', 'displaced', 'migration', 'migrant', 'refugee', 'idp', 'returnee', 'internally'}
        health_terms = all_common & {'health', 'medical', 'hospital', 'clinic', 'disease', 'vaccination', 'mhpss'}
        protection_terms = all_common & {'protection', 'trafficking', 'vulnerable', 'victim', 'gbv', 'child'}
        humanitarian_terms = all_common & {'humanitarian', 'assistance', 'relief', 'shelter', 'food', 'water', 'nfi'}
        
        explanation_parts = []
        
        if geo_terms:
            explanation_parts.append(f"**Geographic Match**: IOM has active operations in {', '.join(geo_terms).title()}")
        
        if crisis_terms:
            explanation_parts.append(f"**Crisis Context**: Both involve {', '.join(crisis_terms)} situations")
        
        if migration_terms:
            explanation_parts.append(f"**Migration Focus**: Shared focus on {', '.join(migration_terms)}")
        
        if health_terms:
            explanation_parts.append(f"**Health Component**: Both address {', '.join(health_terms)} needs")
        
        if protection_terms:
            explanation_parts.append(f"**Protection Elements**: Common themes of {', '.join(protection_terms)}")
        
        if humanitarian_terms:
            explanation_parts.append(f"**Humanitarian Response**: Both involve {', '.join(humanitarian_terms)}")
        
        if explanation_parts:
            response += "\n".join(f"- {part}" for part in explanation_parts)
        else:
            response += "- General thematic alignment based on vocabulary overlap"
        
        # Recommendation
        response += f"""

### 💡 Recommendation

"""
        if relevance_level == "HIGH":
            response += """**PURSUE**: This opportunity strongly aligns with IOM's mandate and experience. 
Consider leveraging the matched PRIMA projects as reference for proposal development.
The organization has demonstrated capacity in similar contexts."""
        elif relevance_level == "MEDIUM":
            response += """**REVIEW**: This opportunity has potential alignment with IOM's work.
Recommend further assessment to identify specific entry points.
Consider partnerships if internal capacity is limited in certain areas."""
        else:
            response += """**EVALUATE CAREFULLY**: Limited direct experience in this specific area.
Consider whether this fits strategic priorities.
May require capacity building or consortium approach if pursuing."""
    
    else:
        response += """### ⚠️ No Strong Matches Found

This opportunity doesn't closely match existing PRIMA projects. Possible reasons:
- Outside IOM's typical programming areas
- New geographic or thematic focus
- Different terminology used

**Recommendation**: Manual review needed to assess strategic fit.
"""
    
    response += f"""

---

🔗 [View Full Opportunity on Devex]({opp.devex_url})
"""
    
    return response


def explain_with_ai(
    opp: FundingOpportunity,
    prima_projects: List[PRIMAProject],
    client: AzureOpenAI,
    deployment: str
) -> str:
    """Rich AI-powered explanation with side-by-side comparison."""
    
    # Find top matches first using keywords to narrow down
    opp_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', f"{opp.title} {opp.description}".lower()))
    stopwords = {'this', 'that', 'with', 'from', 'will', 'their', 'which', 'through', 'project', 'about', 'have', 'been'}
    opp_words -= stopwords
    
    matches = []
    for prima in prima_projects:
        if not prima.summary or len(prima.summary) < 50:
            continue
        prima_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', f"{prima.title} {prima.summary}".lower()))
        prima_words -= stopwords
        common = opp_words & prima_words
        if len(common) >= 2:
            similarity = len(common) / len(opp_words | prima_words)
            matches.append((prima, similarity, list(common)))
    
    matches.sort(key=lambda x: -x[1])
    top_matches = matches[:5]  # Send top 5 to AI for analysis
    
    # Build PRIMA context for AI
    prima_context = ""
    for i, (prima, score, common) in enumerate(top_matches, 1):
        prima_context += f"""
--- PRIMA PROJECT {i} ---
Project ID: {prima.project_id}
Title: {prima.title}
Reporting Area: {prima.reporting_area}
Country: {prima.country}
Funding Source: {prima.funding_source}
Budget: ${prima.budget:,.0f}
Keyword Similarity: {score:.0%}

Full Summary:
{prima.summary}
"""
    
    prompt = f"""You are an IOM Resource Mobilization expert. Analyze this Devex funding opportunity and compare it to the PRIMA projects to assess relevance.

=== DEVEX FUNDING OPPORTUNITY ===
ID: {opp.id}
Title: {opp.title}
Type: {opp.type}
Status: {opp.status}
Region: {', '.join(opp.iom_regions)}
Countries: {', '.join(opp.countries)}
Donors: {', '.join(opp.donors)}
Thematic Area: {opp.thematic_area}
Deadline: {opp.deadline or 'Not specified'}

Full Description:
{opp.description}

=== CANDIDATE PRIMA PROJECTS ===
{prima_context}

=== YOUR TASK ===
Provide a detailed analysis in this exact format:

## 📊 RELEVANCE ASSESSMENT

**Overall Relevance: [HIGH/MEDIUM/LOW] ([percentage]%)**

[2-3 sentences explaining the overall relevance]

## 🔍 DETAILED COMPARISON

### Best Match: [Project ID] - [Title]
**Why it matches:**
[Explain specific connections between the Devex opportunity and this PRIMA project. Be specific about:
- Thematic alignment
- Geographic overlap  
- Similar target populations
- Comparable activities]

**Key similarities:**
- [Point 1]
- [Point 2]
- [Point 3]

### Second Best Match: [Project ID] - [Title]
**Why it matches:**
[Explanation]

## 💡 RECOMMENDATION

**Action: [PURSUE / REVIEW / EVALUATE CAREFULLY]**

[Provide specific recommendation for IOM Resource Mobilization team, including:
- Whether to pursue this opportunity
- What expertise/capacity IOM can leverage
- Any concerns or gaps
- Suggested next steps]"""

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are an expert IOM Resource Mobilization advisor. Provide clear, actionable analysis comparing funding opportunities to past IOM projects."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        ai_response = response.choices[0].message.content
        
        # Build full response with opportunity details
        header = f"""# 🎯 Opportunity Analysis (AI-Powered)

---

## 📋 DEVEX OPPORTUNITY

| Field | Value |
|-------|-------|
| **ID** | {opp.id} |
| **Title** | {opp.title} |
| **Type** | {opp.type} |
| **Status** | {opp.status} |
| **Region** | {', '.join(opp.iom_regions)} |
| **Countries** | {', '.join(opp.countries[:5])} |
| **Donors** | {', '.join(opp.donors[:3])} |
| **Thematic Area** | {opp.thematic_area} |
| **Deadline** | {opp.deadline or 'Not specified'} |

### Description
{opp.description}

---

## 🔗 MATCHING PRIMA PROJECTS

"""
        # Add PRIMA project details
        for i, (prima, score, common) in enumerate(top_matches[:3], 1):
            header += f"""### Match {i}: {prima.title}

| Field | Value |
|-------|-------|
| **Project ID** | {prima.project_id} |
| **Reporting Area** | {prima.reporting_area} |
| **Country** | {prima.country} |
| **Budget** | ${prima.budget:,.0f} |
| **Similarity Score** | **{score:.0%}** |

#### PRIMA Project Summary
{prima.summary}

---

"""
        
        return header + ai_response + f"\n\n---\n\n🔗 [View Full Opportunity on Devex]({opp.devex_url})"
        
    except Exception as e:
        error_msg = str(e)
        
        # Provide helpful troubleshooting based on error type
        if "401" in error_msg:
            troubleshoot = """
⚠️ **Authentication Error (401)**

Please check in Azure Portal → Azure OpenAI:
1. **Endpoint**: Must be `https://YOUR-RESOURCE.openai.azure.com/`
2. **API Key**: Copy from "Keys and Endpoint" section
3. **Deployment**: Check "Model Deployments" for the exact name
4. **API Version**: Try `2024-08-01-preview` or `2023-05-15`

---
**Keyword-based analysis (still works!):**

"""
        elif "404" in error_msg:
            troubleshoot = """
⚠️ **Deployment Not Found (404)**

Your deployment name might be wrong. In Azure Portal:
1. Go to Azure OpenAI → Model Deployments
2. Copy the exact "Deployment name" (not the model name)

---
**Keyword-based analysis:**

"""
        else:
            troubleshoot = f"""
⚠️ **AI Error**: {error_msg}

---
**Keyword-based analysis:**

"""
        
        return troubleshoot + explain_with_keywords(opp, prima_projects)


# =============================================================================
# EXCEL GENERATION
# =============================================================================

def generate_excel(opps: List[FundingOpportunity], region: str = None) -> BytesIO:
    """Generate Excel file."""
    
    if region:
        filtered = [o for o in opps if region in o.iom_regions]
    else:
        filtered = opps
    
    data = []
    for opp in filtered:
        data.append({
            'ID': opp.id,
            'Title': opp.title,
            'Type': opp.type,
            'Status': opp.status,
            'Thematic Area': opp.thematic_area,
            'Region': ', '.join(opp.iom_regions),
            'Countries': ', '.join(opp.countries[:5]),
            'Donors': ', '.join(opp.donors[:3]),
            'Deadline': opp.deadline or '',
            'Devex URL': opp.devex_url,
        })
    
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name=region or 'All')
    output.seek(0)
    return output


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.set_page_config(page_title="Devex-PRIMA Analyzer", page_icon="🌍", layout="wide")
    
    st.title("🌍 Devex Funding Opportunities Analyzer")
    st.markdown("Fast analysis → Browse → Select → AI explains why it matches IOM's work")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    st.sidebar.markdown("---")
    use_ai = st.sidebar.checkbox("🤖 Enable AI Explanations", value=False)
    
    if not use_ai:
        st.sidebar.caption("☝️ Check this to configure Azure OpenAI")
        st.sidebar.info("💡 Without AI, keyword matching still works!")
    
    azure_client = None
    azure_deployment = None
    
    if use_ai:
        st.sidebar.markdown("**Azure OpenAI Settings**")
        st.sidebar.caption("Find in: Azure Portal → Azure OpenAI → Keys and Endpoint")
        
        azure_endpoint = st.sidebar.text_input(
            "Endpoint",
            placeholder="https://your-resource.openai.azure.com/",
            help="Must include https:// and end with /"
        )
        
        azure_key = st.sidebar.text_input(
            "API Key", 
            type="password",
            help="From Azure Portal (Key 1 or Key 2)"
        )
        
        azure_deployment = st.sidebar.text_input(
            "Deployment Name",
            placeholder="gpt-4o-mini",
            help="Your deployment name from Model Deployments"
        )
        
        azure_api_version = st.sidebar.text_input(
            "API Version",
            value="2024-08-01-preview",
            help="Latest: 2024-08-01-preview"
        )
        
        if azure_endpoint and azure_key and azure_deployment:
            # Fix endpoint format
            if not azure_endpoint.startswith('https://'):
                azure_endpoint = 'https://' + azure_endpoint
            if not azure_endpoint.endswith('/'):
                azure_endpoint = azure_endpoint + '/'
            
            try:
                azure_client = AzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=azure_key,
                    api_version=azure_api_version
                )
                st.sidebar.success("✓ Client ready")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")
    
    # File uploads
    col1, col2 = st.columns(2)
    with col1:
        devex_file = st.file_uploader("📄 Upload Devex XML", type=['xml'])
    with col2:
        prima_file = st.file_uploader("📊 Upload PRIMA Data", type=['xlsx', 'xls', 'csv'])
    
    # Session state
    if 'opportunities' not in st.session_state:
        st.session_state.opportunities = None
    if 'prima_projects' not in st.session_state:
        st.session_state.prima_projects = None
    if 'selected_opp' not in st.session_state:
        st.session_state.selected_opp = None
    
    # Process files
    if devex_file and prima_file:
        if st.button("🚀 Analyze", type="primary"):
            with st.spinner("Parsing Devex XML..."):
                opportunities = parse_devex_xml(devex_file.read())
            
            with st.spinner("Loading PRIMA data..."):
                prima_projects = parse_prima_data(prima_file)
            
            st.session_state.opportunities = opportunities
            st.session_state.prima_projects = prima_projects
            st.session_state.selected_opp = None
            
            st.success(f"✓ Loaded {len(opportunities)} opportunities and {len(prima_projects)} PRIMA projects")
    
    # Display results
    if st.session_state.opportunities:
        opportunities = st.session_state.opportunities
        prima_projects = st.session_state.prima_projects
        
        # =====================================================================
        # TAB 1: OVERVIEW
        # =====================================================================
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Browse & Select", "💬 Ask AI"])
        
        with tab1:
            st.header("📊 Analysis Overview")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Opportunities", len(opportunities))
            col2.metric("PRIMA Projects", len(prima_projects))
            col3.metric("Open Status", len([o for o in opportunities if o.status == 'open']))
            col4.metric("With Deadlines", len([o for o in opportunities if o.deadline]))
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("By Region")
                region_counts = {}
                for o in opportunities:
                    for r in o.iom_regions:
                        region_counts[r] = region_counts.get(r, 0) + 1
                region_df = pd.DataFrame({
                    'Region': list(region_counts.keys()),
                    'Count': list(region_counts.values())
                }).sort_values('Count', ascending=True)
                st.bar_chart(region_df.set_index('Region'))
            
            with col2:
                st.subheader("By Thematic Area")
                theme_counts = {}
                for o in opportunities:
                    theme_counts[o.thematic_area] = theme_counts.get(o.thematic_area, 0) + 1
                theme_df = pd.DataFrame({
                    'Theme': list(theme_counts.keys()),
                    'Count': list(theme_counts.values())
                }).sort_values('Count', ascending=True)
                st.bar_chart(theme_df.set_index('Theme'))
            
            # Top donors
            st.subheader("Top Donors")
            donor_counts = {}
            for o in opportunities:
                for d in o.donors[:1]:
                    donor_counts[d] = donor_counts.get(d, 0) + 1
            top_donors = sorted(donor_counts.items(), key=lambda x: -x[1])[:10]
            donor_df = pd.DataFrame(top_donors, columns=['Donor', 'Opportunities'])
            st.dataframe(donor_df, hide_index=True, use_container_width=True)
            
            # Download buttons
            st.subheader("📥 Download Regional Files")
            regions = ['Africa', 'Americas', 'Asia_Pacific', 'Europe', 'MENA', 'Global']
            cols = st.columns(len(regions))
            for i, region in enumerate(regions):
                count = len([o for o in opportunities if region in o.iom_regions])
                with cols[i]:
                    st.download_button(
                        f"{region} ({count})",
                        data=generate_excel(opportunities, region),
                        file_name=f"Devex_{region}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        # =====================================================================
        # TAB 2: BROWSE & SELECT
        # =====================================================================
        with tab2:
            st.header("🔍 Browse Opportunities")
            st.markdown("Filter, browse, and **select one to get AI explanation**")
            
            # Filters (3 columns: Region, Thematic Area, Status)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_region = st.selectbox("Region", ["All"] + list(set(
                    r for o in opportunities for r in o.iom_regions
                )))
            
            with col2:
                filter_theme = st.selectbox("Thematic Area", ["All"] + list(set(
                    o.thematic_area for o in opportunities
                )))
            
            with col3:
                filter_status = st.selectbox("Status", ["All", "open", "closed", "forecast"])
            
            # Search
            search = st.text_input("🔎 Search in title/description")
            
            # Apply filters
            filtered = opportunities
            if filter_region != "All":
                filtered = [o for o in filtered if filter_region in o.iom_regions]
            if filter_theme != "All":
                filtered = [o for o in filtered if o.thematic_area == filter_theme]
            if filter_status != "All":
                filtered = [o for o in filtered if o.status == filter_status]
            if search:
                search_lower = search.lower()
                filtered = [o for o in filtered if search_lower in o.title.lower() or search_lower in o.description.lower()]
            
            st.markdown(f"**Showing Top 5 of {len(filtered)} opportunities**")
            
            # Display as table (Top 5 only)
            if filtered:
                table_data = []
                for o in filtered[:5]:
                    table_data.append({
                        'ID': o.id,
                        'Title': o.title[:60] + '...' if len(o.title) > 60 else o.title,
                        'Type': o.type,
                        'Theme': o.thematic_area,
                        'Region': ', '.join(o.iom_regions),
                        'Status': o.status,
                        'Deadline': o.deadline or '-',
                    })
                
                df = pd.DataFrame(table_data)
                
                # Show table
                st.dataframe(
                    df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "ID": st.column_config.NumberColumn("ID", width="small"),
                        "Title": st.column_config.TextColumn("Title", width="large"),
                    }
                )
                
                # Selection
                st.markdown("---")
                st.subheader("🎯 Select an Opportunity for AI Analysis")
                
                opp_options = {f"{o.id}: {o.title[:50]}...": o.id for o in filtered[:5]}
                selected = st.selectbox("Choose opportunity", ["-- Select --"] + list(opp_options.keys()))
                
                if selected != "-- Select --":
                    selected_id = opp_options[selected]
                    selected_opp = next((o for o in opportunities if o.id == selected_id), None)
                    
                    if selected_opp:
                        if st.button("🤖 Explain Why This Matches IOM's Work", type="primary"):
                            st.session_state.selected_opp = selected_opp
                            
                            with st.spinner("Analyzing opportunity against PRIMA projects..."):
                                explanation = explain_opportunity(
                                    selected_opp,
                                    prima_projects,
                                    client=azure_client,
                                    deployment=azure_deployment
                                )
                            
                            st.markdown(explanation)
        
        # =====================================================================
        # TAB 3: CHAT
        # =====================================================================
        with tab3:
            st.header("💬 Ask About Opportunities")
            
            if 'chat_messages' not in st.session_state:
                st.session_state.chat_messages = []
            
            # Show chat history
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask about opportunities (e.g., 'Explain opportunity 840928')"):
                st.session_state.chat_messages.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    # Parse intent
                    prompt_lower = prompt.lower()
                    
                    # Check for opportunity ID
                    id_match = re.search(r'\b(\d{5,})\b', prompt)
                    
                    if id_match:
                        opp_id = int(id_match.group(1))
                        opp = next((o for o in opportunities if o.id == opp_id), None)
                        
                        if opp:
                            with st.spinner(f"Analyzing opportunity {opp_id}..."):
                                response = explain_opportunity(
                                    opp, prima_projects,
                                    client=azure_client,
                                    deployment=azure_deployment
                                )
                        else:
                            response = f"I couldn't find an opportunity with ID {opp_id}."
                    
                    elif any(w in prompt_lower for w in ['summary', 'overview', 'how many']):
                        response = f"""## 📊 Summary

| Metric | Value |
|--------|-------|
| Total Opportunities | {len(opportunities)} |
| PRIMA Projects | {len(prima_projects)} |
| Open Opportunities | {len([o for o in opportunities if o.status == 'open'])} |

**By Region:**
"""
                        for r in ['Africa', 'Americas', 'Asia_Pacific', 'Europe', 'MENA', 'Global']:
                            count = len([o for o in opportunities if r in o.iom_regions])
                            response += f"- {r}: {count}\n"
                        
                        response += "\n💡 *To analyze a specific opportunity, ask: 'Explain opportunity [ID]'*"
                    
                    else:
                        response = """I can help you analyze specific opportunities!

**Try asking:**
- "Explain opportunity 840928"
- "Give me a summary"
- "What's opportunity 123456 about?"

**Or use the Browse tab to:**
1. Filter by region/theme
2. Select an opportunity
3. Click "Explain" for AI analysis
"""
                    
                    st.markdown(response)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
            
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_messages = []
                st.rerun()


if __name__ == "__main__":
    main()
