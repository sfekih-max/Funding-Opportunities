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
from datetime import datetime, timedelta
import hashlib
import requests  # For Azure Responses API

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
# FUNDING HIGHLIGHTS CONFIGURATION
# =============================================================================

# Donor countries and organizations for Funding Highlights
DONORS = {
    # Bilateral donors
    "Australia": {"region": "Asia-Pacific", "focal_point": "", "keywords": ["australia", "australian", "dfat", "canberra"]},
    "Belgium": {"region": "Europe", "focal_point": "", "keywords": ["belgium", "belgian", "brussels", "dgd"]},
    "Canada": {"region": "Americas", "focal_point": "", "keywords": ["canada", "canadian", "gac", "ottawa", "ircc"]},
    "Finland": {"region": "Europe", "focal_point": "", "keywords": ["finland", "finnish", "helsinki", "mfa finland"]},
    "France": {"region": "Europe", "focal_point": "", "keywords": ["france", "french", "afd", "paris", "quai d'orsay"]},
    "Germany": {"region": "Europe", "focal_point": "", "keywords": ["germany", "german", "bmz", "giz", "berlin", "kfw"]},
    "Italy": {"region": "Europe", "focal_point": "", "keywords": ["italy", "italian", "aics", "rome", "farnesina"]},
    "Japan": {"region": "Asia-Pacific", "focal_point": "", "keywords": ["japan", "japanese", "jica", "tokyo", "mofa japan"]},
    "Kuwait": {"region": "MENA", "focal_point": "", "keywords": ["kuwait", "kuwaiti", "kuwait fund"]},
    "Netherlands": {"region": "Europe", "focal_point": "", "keywords": ["netherlands", "dutch", "the hague", "bz"]},
    "Norway": {"region": "Europe", "focal_point": "", "keywords": ["norway", "norwegian", "norad", "oslo"]},
    "Qatar": {"region": "MENA", "focal_point": "", "keywords": ["qatar", "qatari", "doha", "qatar fund"]},
    "Republic of Ireland": {"region": "Europe", "focal_point": "", "keywords": ["ireland", "irish", "dublin", "irish aid"]},
    "Republic of Korea": {"region": "Asia-Pacific", "focal_point": "", "keywords": ["korea", "korean", "koica", "seoul"]},
    "Saudi Arabia": {"region": "MENA", "focal_point": "", "keywords": ["saudi", "arabia", "riyadh", "ksrelief"]},
    "Spain": {"region": "Europe", "focal_point": "", "keywords": ["spain", "spanish", "aecid", "madrid"]},
    "Sweden": {"region": "Europe", "focal_point": "", "keywords": ["sweden", "swedish", "sida", "stockholm"]},
    "Switzerland": {"region": "Europe", "focal_point": "", "keywords": ["switzerland", "swiss", "sdc", "bern", "deza"]},
    "United Arab Emirates": {"region": "MENA", "focal_point": "", "keywords": ["uae", "emirates", "abu dhabi", "dubai"]},
    "United Kingdom": {"region": "Europe", "focal_point": "", "keywords": ["uk", "britain", "british", "fcdo", "london", "home office"]},
    "United States": {"region": "Americas", "focal_point": "", "keywords": ["usa", "usaid", "state department", "washington", "prm", "bha"]},
    # Multilateral donors
    "European Union": {"region": "Multilateral", "focal_point": "", "keywords": ["eu", "european union", "european commission", "dg intpa", "dg echo", "brussels"]},
    "World Bank": {"region": "Multilateral", "focal_point": "", "keywords": ["world bank", "ibrd", "ida", "ifc"]},
    "African Development Bank": {"region": "Multilateral", "focal_point": "", "keywords": ["afdb", "african development bank"]},
    "Asian Development Bank": {"region": "Multilateral", "focal_point": "", "keywords": ["adb", "asian development bank", "manila"]},
    "Inter-American Development Bank": {"region": "Multilateral", "focal_point": "", "keywords": ["idb", "iadb", "inter-american"]},
    "Green Climate Fund": {"region": "Multilateral", "focal_point": "", "keywords": ["gcf", "green climate fund"]},
}

# News categories for Funding Highlights
NEWS_CATEGORIES = {
    "budget_announcement": "Budget/Funding Announcement",
    "political_change": "Political Change/Cabinet Reshuffle",
    "policy_update": "Policy Update",
    "multilateral_commitment": "Multilateral Commitment",
    "bilateral_agreement": "Bilateral Agreement",
    "election_result": "Election Result",
    "strategy_release": "Strategy/Framework Release",
    "humanitarian_response": "Humanitarian Response",
}

# Relevance keywords for migration/IOM (Funding Highlights)
MIGRATION_KEYWORDS = [
    "migration", "migrant", "refugee", "asylum", "displacement", "displaced",
    "resettlement", "return", "reintegration", "trafficking", "smuggling",
    "border", "visa", "integration", "diaspora", "remittance",
    "idp", "internally displaced", "protection", "returnee",
    "humanitarian", "development assistance", "oda", "official development assistance",
    "climate mobility", "crisis", "emergency", "conflict", "war", "disaster",
    "funding", "grant", "million", "billion", "budget", "aid", "assistance",
    "allocation", "commitment", "contribution", "pledge",
    "unhcr", "iom", "unicef", "wfp", "ocha", "undp",
    "africa", "middle east", "latin america", "asia", "europe",
    "ukraine", "syria", "afghanistan", "sudan", "gaza", "yemen", "ethiopia", "somalia",
    "nigeria", "libya", "venezuela", "myanmar", "bangladesh", "rohingya",
    "immigration", "asylum seekers", "irregular", "regular pathways",
    "safe routes", "family reunion", "removal", "deportation",
]

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


@dataclass
class FundingNews:
    """Data class for Funding Highlights news items."""
    id: str
    title: str
    summary: str
    full_text: str
    source: str
    source_url: str
    date: str
    donor: str
    donor_region: str
    category: str
    relevance_score: float
    migration_keywords_found: List[str]
    ai_summary: str = ""
    focal_point: str = ""
    is_duplicate: bool = False
    duplicate_of: str = ""
    selected_for_bulletin: bool = False
    reviewer_notes: str = ""
    confidence_warnings: List[str] = field(default_factory=list)  # Hallucination warnings


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


def get_opportunity_matches(opp: FundingOpportunity, prima_projects: List[PRIMAProject]) -> Tuple[List[Tuple], str]:
    """Get keyword matches for an opportunity. Returns (matches, relevance_level)."""
    
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
            
            important_terms = {'migration', 'migrant', 'refugee', 'displacement', 'displaced', 
                             'humanitarian', 'protection', 'trafficking', 'returnee', 'idp',
                             'border', 'asylum', 'resettlement', 'reintegration'}
            important_matches = common & important_terms
            boosted_similarity = similarity + (len(important_matches) * 0.05)
            
            matches.append((prima, boosted_similarity, list(common)))
    
    matches.sort(key=lambda x: -x[1])
    top_matches = matches[:3]
    
    if top_matches:
        avg_score = sum(s for _, s, _ in top_matches) / len(top_matches)
    else:
        avg_score = 0
    
    if avg_score >= 0.15:
        relevance_level = "HIGH"
    elif avg_score >= 0.08:
        relevance_level = "MEDIUM"
    else:
        relevance_level = "LOW"
    
    return top_matches, relevance_level


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
# FUNDING HIGHLIGHTS FUNCTIONS
# =============================================================================

def generate_news_id(title: str, source: str) -> str:
    """Generate unique ID for news item."""
    text = f"{title.lower()}{source.lower()}"
    return hashlib.md5(text.encode()).hexdigest()[:12]


def detect_donor(text: str) -> Tuple[str, float]:
    """Detect which donor the news is about."""
    text_lower = text.lower()
    
    best_match = None
    best_score = 0
    
    for donor, info in DONORS.items():
        score = 0
        for keyword in info["keywords"]:
            if keyword.lower() in text_lower:
                score += 1
        
        if score > best_score:
            best_score = score
            best_match = donor
    
    confidence = min(best_score / 3, 1.0) if best_score > 0 else 0
    return best_match or "Unknown", confidence


def calculate_news_relevance(text: str) -> Tuple[float, List[str]]:
    """Calculate migration/IOM relevance score for news."""
    text_lower = text.lower()
    
    found_keywords = []
    for keyword in MIGRATION_KEYWORDS:
        if keyword.lower() in text_lower:
            found_keywords.append(keyword)
    
    # Base score from keyword matches (more generous)
    base_score = min(len(found_keywords) / 5, 1.0)
    
    # Boost for IOM-specific mentions
    if "iom" in text_lower:
        base_score = min(base_score + 0.3, 1.0)
    
    # Boost for core migration terms
    core_terms = ["migration", "migrant", "refugee", "asylum", "displacement", "displaced", "humanitarian"]
    core_matches = sum(1 for t in core_terms if t in text_lower)
    base_score = min(base_score + (core_matches * 0.1), 1.0)
    
    # Boost for funding-related terms
    funding_terms = ["million", "billion", "funding", "grant", "budget", "oda", "aid"]
    funding_matches = sum(1 for t in funding_terms if t in text_lower)
    base_score = min(base_score + (funding_matches * 0.08), 1.0)
    
    # Boost for policy terms
    policy_terms = ["reform", "policy", "law", "regulation", "legislation"]
    policy_matches = sum(1 for t in policy_terms if t in text_lower)
    base_score = min(base_score + (policy_matches * 0.05), 1.0)
    
    return base_score, found_keywords


def categorize_news(text: str) -> str:
    """Categorize the type of news."""
    text_lower = text.lower()
    
    if any(w in text_lower for w in ["budget", "funding", "million", "billion", "grant", "allocation"]):
        return "budget_announcement"
    elif any(w in text_lower for w in ["election", "elected", "vote", "ballot"]):
        return "election_result"
    elif any(w in text_lower for w in ["minister", "cabinet", "reshuffle", "appointed", "resigned"]):
        return "political_change"
    elif any(w in text_lower for w in ["policy", "reform", "law", "legislation", "regulation"]):
        return "policy_update"
    elif any(w in text_lower for w in ["strategy", "framework", "plan", "roadmap"]):
        return "strategy_release"
    elif any(w in text_lower for w in ["humanitarian", "crisis", "emergency", "response"]):
        return "humanitarian_response"
    elif any(w in text_lower for w in ["agreement", "mou", "partnership", "bilateral"]):
        return "bilateral_agreement"
    else:
        return "policy_update"


def parse_funding_news(text: str) -> List[FundingNews]:
    """Parse manually pasted news items for Funding Highlights."""
    news_items = []
    
    # Split by patterns that indicate a new entry
    entries = re.split(r'\n\s*(?=[A-Z][a-zA-Z\s]+:(?:\s|$))|\n\s*(?=•\s*[A-Z])|\n\s*(?=\*\s*[A-Z])', text)
    
    for entry in entries:
        entry = entry.strip()
        if len(entry) < 30:
            continue
        
        # Clean up bullet points
        entry = re.sub(r'^[•\*]\s*', '', entry)
        
        # Try to extract donor from start
        donor_match = re.match(r'^([A-Za-z][A-Za-z\s]+?):\s*', entry)
        if donor_match:
            potential_donor = donor_match.group(1).strip()
            if potential_donor in DONORS or any(potential_donor.lower() in d.lower() for d in DONORS):
                donor = potential_donor
                content = entry[donor_match.end():].strip()
            else:
                donor, _ = detect_donor(entry)
                content = entry
        else:
            donor, _ = detect_donor(entry)
            content = entry
        
        # Extract source if present
        source_match = re.search(r'Source:\s*(.+?)(?:\n|$)', content, re.IGNORECASE)
        source = source_match.group(1).strip() if source_match else "Manual Entry"
        
        # Clean content
        content_clean = re.sub(r'Source:\s*.+?(?:\n|$)', '', content, flags=re.IGNORECASE).strip()
        
        # Calculate relevance
        relevance, keywords = calculate_news_relevance(content_clean)
        
        # Generate title
        title_match = re.match(r'^(.+?[.!?])\s', content_clean)
        title = title_match.group(1) if title_match else content_clean[:100]
        
        news_items.append(FundingNews(
            id=generate_news_id(title, source),
            title=title[:150] + "..." if len(title) > 150 else title,
            summary=content_clean[:500],
            full_text=content_clean,
            source=source,
            source_url="",
            date=datetime.now().strftime("%Y-%m-%d"),
            donor=donor,
            donor_region=DONORS.get(donor, {}).get("region", "Unknown"),
            category=categorize_news(content_clean),
            relevance_score=relevance,
            migration_keywords_found=keywords,
        ))
    
    return news_items


def generate_funding_highlights_export(news_items: List[FundingNews]) -> BytesIO:
    """Generate text export for Funding Highlights bulletin."""
    selected = [n for n in news_items if n.selected_for_bulletin]
    selected.sort(key=lambda x: -x.relevance_score)
    
    output_text = "FUNDING HIGHLIGHTS\n"
    output_text += "=" * 50 + "\n\n"
    output_text += "Key updates about IOM's key donors from RMD, COPAs, and Country Offices in donor capitals.\n\n"
    
    for news in selected:
        summary = news.ai_summary if news.ai_summary else news.summary
        output_text += f"• {news.donor}: {summary} Source: {news.source}\n\n"
    
    output_text += f"\nGenerated: {datetime.now().strftime('%d %B %Y')}\n"
    
    return BytesIO(output_text.encode('utf-8'))


def generate_funding_highlights_excel(news_items: List[FundingNews]) -> BytesIO:
    """Generate Excel report with all funding news for review."""
    data = []
    for news in news_items:
        data.append({
            'Selected': '✓' if news.selected_for_bulletin else '',
            'Donor': news.donor,
            'Region': news.donor_region,
            'Category': NEWS_CATEGORIES.get(news.category, news.category),
            'Title': news.title,
            'Summary': news.ai_summary if news.ai_summary else news.summary,
            'Relevance Score': f"{news.relevance_score:.0%}",
            'Keywords Found': ', '.join(news.migration_keywords_found[:5]),
            'Source': news.source,
            'Source URL': news.source_url,
            'Verified': '✅' if news.source_url and not news.confidence_warnings else '⚠️ Verify',
            'Confidence Warnings': '; '.join(news.confidence_warnings) if news.confidence_warnings else '',
            'Date': news.date,
            'Duplicate': 'Yes' if news.is_duplicate else 'No',
            'Focal Point': news.focal_point,
        })
    
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Funding News')
    output.seek(0)
    return output


# =============================================================================
# EXCEL GENERATION
# =============================================================================

def generate_matches_excel(selected_matches: List[Tuple]) -> BytesIO:
    """Generate Excel file with selected opportunity matches."""
    data = []
    for opp, matches, relevance_level in selected_matches:
        for i, (prima, score, keywords) in enumerate(matches[:3], 1):
            data.append({
                'Opportunity ID': opp.id,
                'Opportunity Title': opp.title,
                'Type': opp.type,
                'Status': opp.status,
                'Thematic Area': opp.thematic_area,
                'Region': ', '.join(opp.iom_regions),
                'Countries': ', '.join(opp.countries[:5]),
                'Donors': ', '.join(opp.donors[:3]),
                'Deadline': opp.deadline or '',
                'Devex URL': opp.devex_url,
                'Relevance Level': relevance_level,
                'Match Rank': i,
                'PRIMA Project ID': prima.project_id,
                'PRIMA Title': prima.title,
                'PRIMA Reporting Area': prima.reporting_area,
                'PRIMA Country': prima.country,
                'PRIMA Budget': f"${prima.budget:,.0f}" if prima.budget else '',
                'Match Score': f"{score:.0%}",
                'Common Keywords': ', '.join(keywords[:10]),
            })
    
    df = pd.DataFrame(data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Matches')
    output.seek(0)
    return output


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
# WEB SEARCH FOR FUNDING HIGHLIGHTS
# =============================================================================

# Prompt template for fetching donor news
DONOR_NEWS_PROMPT = """Your role: highly experienced resource mobilization specialist and political analyst in the field of migration. Help prepare an internal newsletter about recent funding and political developments that might impact the migration field.

Search for news from the following donors: Australia, Belgium, Canada, Finland, France, Germany, Italy, Japan, Kuwait, Netherlands, Norway, Qatar, Republic of Ireland, Republic of Korea, Saudi Arabia, Spain, Sweden, Switzerland, United Arab Emirates, United Kingdom, United States, European Union, World Bank, African Development Bank, Asian Development Bank, Inter-American Development Bank, Green Climate Fund.

Search sources including:
- donortracker.org
- reliefweb.int
- oecd.org/dac
- devex.com
- fts.unocha.org
- Government Ministry and Agency press pages

Find the following information from the past {days} days:

1. Funding announcements from IOM donors related to humanitarian or development assistance that have direct or indirect influence on the migration field (changes to ODA budget or development cooperation strategy, new commitments to multilateral agencies). Prioritize unearmarked or multi-year funding.

2. Political changes (cabinet reshuffles, elections, ministerial changes) in those countries that might impact migration-related development or humanitarian funding.

Select up to 40 updates (up to 5 per donor government) based on how much they impact the migration field and IOM in particular.

Format each update EXACTLY like this (one per line):
Country: [Factual summary - ONLY include facts explicitly stated in your search results]. Source: [Exact source name](URL)

CRITICAL RULES TO AVOID HALLUCINATIONS:
- ONLY include facts explicitly stated in your search results
- Do NOT infer, extrapolate, or add context not directly from the source
- Do NOT speculate about impacts (e.g., "expected to affect migration programming") unless the source explicitly says this
- If specific details are unknown, say "details not yet announced" rather than inventing them
- Include specific figures (amounts, percentages, dates) ONLY if found in the source
- If you cannot find recent verified news for a donor, SKIP that donor entirely
- Always include the source URL - if no URL is available, do not include the item
- Focus on migration, refugees, humanitarian aid, development assistance
- Exclude news about UNHCR unless directly relevant to IOM
- Prioritize news with broader regional or global relevance"""


def fetch_donor_news_with_web_search(
    azure_endpoint: str,
    azure_key: str,
    model_deployment: str,
    days: int = 30,
    user_location: str = "CH"
) -> Tuple[str, List[dict]]:
    """
    Fetch donor news using Azure OpenAI web search preview.
    
    Uses the Responses API with web_search_preview tool.
    
    Returns:
        Tuple of (raw_text_response, list_of_citations)
    """
    # Build the API URL for Azure Responses API
    # Format: https://{resource}.openai.azure.com/openai/v1/responses
    base_url = azure_endpoint.rstrip('/')
    if not base_url.endswith('/openai/v1'):
        api_url = f"{base_url}/openai/v1/responses"
    else:
        api_url = f"{base_url}/responses"
    
    prompt = DONOR_NEWS_PROMPT.format(days=days)
    
    # Build request payload
    payload = {
        "model": model_deployment,
        "tools": [{
            "type": "web_search_preview",
            "user_location": {
                "type": "approximate",
                "country": user_location
            }
        }],
        "input": prompt
    }
    
    # Azure uses api-key header (not Authorization: Bearer)
    headers = {
        "Content-Type": "application/json",
        "api-key": azure_key
    }
    
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=180  # 3 minutes for web search
        )
        
        # Check for errors
        if response.status_code != 200:
            error_detail = response.text
            try:
                error_json = response.json()
                if 'error' in error_json:
                    error_detail = error_json['error'].get('message', error_detail)
            except:
                pass
            raise Exception(f"API Error {response.status_code}: {error_detail}")
        
        result = response.json()
        
        # Extract output text
        output_text = ""
        citations = []
        
        # The response structure has 'output' array
        if 'output' in result:
            for item in result['output']:
                if item.get('type') == 'message':
                    content = item.get('content', [])
                    for c in content:
                        if c.get('type') == 'output_text':
                            output_text += c.get('text', '')
                            # Extract annotations/citations
                            annotations = c.get('annotations', [])
                            for ann in annotations:
                                if ann.get('type') == 'url_citation':
                                    citations.append({
                                        'url': ann.get('url', ''),
                                        'title': ann.get('title', ''),
                                    })
        
        # Fallback: check for output_text at top level
        if not output_text and 'output_text' in result:
            output_text = result['output_text']
        
        return output_text, citations
        
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. Web search can take up to 2 minutes. Please try again.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {str(e)}")


def parse_web_search_results(text: str, citations: List[dict] = None) -> List[FundingNews]:
    """Parse the web search results into FundingNews objects."""
    news_items = []
    citations = citations or []
    
    # Normalize text
    text = text.replace('\r\n', '\n')
    
    # Split into blocks
    blocks = re.split(r'\n---+\n|\n\n+', text)
    
    for block in blocks:
        block = block.strip()
        if not block or len(block) < 30:
            continue
        
        if block.startswith('Below is') or block.startswith('*Compiled') or block.startswith('If you require'):
            continue
        if 'Macro Trend:' in block or 'Political Developments:' in block or 'IOM-Specific Updates:' in block:
            continue
            
        lines = block.split('\n')
        current_donor = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            country_match = re.match(r'^\*\*([A-Za-z][A-Za-z\s]+?):\*\*\s*(.*)', line) or \
                           re.match(r'^([A-Za-z][A-Za-z\s]+?):\s*(.*)', line)
            
            if country_match:
                if current_donor and current_content:
                    news_item = create_news_item(current_donor, ' '.join(current_content), citations)
                    if news_item:
                        news_items.append(news_item)
                
                potential_donor = country_match.group(1).strip()
                content_start = country_match.group(2).strip() if country_match.group(2) else ""
                
                if potential_donor in DONORS or any(potential_donor.lower() == d.lower() for d in DONORS):
                    current_donor = potential_donor
                    current_content = [content_start] if content_start else []
                else:
                    if current_content:
                        current_content.append(line)
            else:
                if current_donor:
                    current_content.append(line)
        
        if current_donor and current_content:
            news_item = create_news_item(current_donor, ' '.join(current_content), citations)
            if news_item:
                news_items.append(news_item)
    
    return news_items


def flag_potential_hallucinations(text: str, source_url: str) -> List[str]:
    """Flag potential hallucinations or unverified claims in news summaries."""
    warnings = []
    text_lower = text.lower()
    
    # No source URL is a major red flag
    if not source_url:
        warnings.append("⚠️ No source URL - cannot verify claims")
    
    # Speculative/hedging language often indicates fabrication
    hedge_phrases = [
        "expected to", "likely to", "may affect", "could impact", 
        "details pending", "details are pending", "details not yet",
        "is expected", "are expected", "will likely", "might impact",
        "potentially", "possibly", "appears to", "seems to",
        "reportedly planning", "said to be", "believed to"
    ]
    found_hedges = [phrase for phrase in hedge_phrases if phrase in text_lower]
    if found_hedges:
        warnings.append(f"⚠️ Speculative language detected: '{found_hedges[0]}' - verify against source")
    
    # Specific impact claims without source URL are high risk
    specific_claims = [
        "limiting budget flexibility", "affecting funding to", 
        "impact on migration", "reduce funding for",
        "cut funding to", "affecting multilateral",
        "limiting migration programming", "affecting un partners"
    ]
    if any(claim in text_lower for claim in specific_claims) and not source_url:
        warnings.append("🚨 Specific impact claims without source URL - HIGH hallucination risk")
    
    # Inferred connections not typically in news
    inference_patterns = [
        "amid growing domestic", "amid political", "amid budget",
        "while details are pending", "although specifics",
        "this could mean", "this may result", "this suggests"
    ]
    if any(pattern in text_lower for pattern in inference_patterns):
        warnings.append("⚠️ Contains inferred context - may not be from source")
    
    # Round numbers without source might be fabricated
    import re
    round_millions = re.findall(r'\$?\d+\s*(?:million|billion|m|bn)\b', text_lower)
    if round_millions and not source_url:
        warnings.append("⚠️ Contains funding figures but no source URL to verify")
    
    return warnings


def create_news_item(donor: str, content: str, citations: List[dict] = None) -> Optional[FundingNews]:
    """Create a FundingNews object from donor and content."""
    content = content.strip()
    if len(content) < 20:
        return None
    
    # Extract source and URL
    source = "Web Search"
    source_url = ""
    
    # Try [Source](url) format first
    source_url_match = re.search(r'Source:\s*\[([^\]]+)\]\(([^)]+)\)', content)
    if source_url_match:
        source = source_url_match.group(1).strip()
        source_url = source_url_match.group(2).strip()
    else:
        # Try [Source] format
        source_match = re.search(r'Source:\s*\[([^\]]+)\]', content)
        if source_match:
            source = source_match.group(1).strip()
        else:
            # Try plain Source: text
            source_match = re.search(r'Source:\s*([^\[\n\.]+)', content, re.IGNORECASE)
            if source_match:
                source = source_match.group(1).strip()
    
    # Look for inline URLs
    if not source_url:
        url_match = re.search(r'\[([^\]]+)\]\((https?://[^)]+)\)', content)
        if url_match:
            if not source or source == "Web Search":
                source = url_match.group(1).strip()
            source_url = url_match.group(2).strip()
        else:
            standalone_url = re.search(r'(https?://[^\s\)]+)', content)
            if standalone_url:
                source_url = standalone_url.group(1).strip()
    
    # Match with citations
    if not source_url and citations:
        content_lower = content.lower()
        for cite in citations:
            cite_title = cite.get('title', '').lower()
            if cite_title and any(word in content_lower for word in cite_title.split()[:3] if len(word) > 3):
                source_url = cite.get('url', '')
                if not source or source == "Web Search":
                    source = cite.get('title', source)
                break
    
    # Clean content
    content_clean = re.sub(r'Source:\s*\[[^\]]+\]\([^)]+\)', '', content)
    content_clean = re.sub(r'Source:\s*\[[^\]]+\]', '', content_clean)
    content_clean = re.sub(r'Source:\s*[^\[\n\.]+\.?', '', content_clean, flags=re.IGNORECASE)
    content_clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content_clean)
    content_clean = re.sub(r'https?://[^\s\)]+', '', content_clean)
    content_clean = content_clean.strip()
    
    if len(content_clean) < 20:
        return None
    
    relevance, keywords = calculate_news_relevance(content_clean)
    
    title_match = re.match(r'^(.+?[.!?])\s', content_clean)
    title = title_match.group(1) if title_match else content_clean[:100]
    
    return FundingNews(
        id=generate_news_id(title, source),
        title=title[:150] + "..." if len(title) > 150 else title,
        summary=content_clean[:500],
        full_text=content_clean,
        source=source,
        source_url=source_url,
        date=datetime.now().strftime("%Y-%m-%d"),
        donor=donor,
        donor_region=DONORS.get(donor, {}).get("region", "Unknown"),
        category=categorize_news(content_clean),
        relevance_score=relevance,
        migration_keywords_found=keywords,
        confidence_warnings=flag_potential_hallucinations(content_clean, source_url),
    )


def parse_single_news_entry(entry: str) -> Optional[FundingNews]:
    """Parse a single news entry into a FundingNews object. (Legacy - kept for compatibility)"""
    entry = entry.strip()
    if len(entry) < 30:
        return None
    
    # Remove bullet points
    entry = re.sub(r'^[\-\*•]\s*', '', entry)
    
    # Try to match **Country:** format (markdown bold)
    match = re.match(r'^\*\*([A-Za-z][A-Za-z\s]+?):\*\*\s*(.+)', entry)
    if not match:
        # Try plain Country: format
        match = re.match(r'^([A-Za-z][A-Za-z\s]+?):\s*(.+)', entry)
    
    if not match:
        return None
    
    potential_donor = match.group(1).strip()
    content = match.group(2).strip()
    
    # Check if it's a valid donor
    if potential_donor not in DONORS and not any(potential_donor.lower() in d.lower() for d in DONORS):
        return None
    
    return create_news_item(potential_donor, content)

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
    azure_endpoint = ""
    azure_key = ""
    
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
    if 'selected_matches' not in st.session_state:
        st.session_state.selected_matches = []
    
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
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Browse & Select", "📰 Funding Highlights"])
        
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
            
            # Filters
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                filter_region = st.selectbox("Region", ["All"] + list(set(
                    r for o in opportunities for r in o.iom_regions
                )))
            
            with col2:
                filter_theme = st.selectbox("Thematic Area", ["All"] + list(set(
                    o.thematic_area for o in opportunities
                )))
            
            with col3:
                filter_type = st.selectbox("Type", ["All"] + list(set(
                    o.type for o in opportunities if o.type
                )))
            
            with col4:
                filter_status = st.selectbox("Status", ["All", "open", "closed", "forecast"])
            
            # Search
            search = st.text_input("🔎 Search in title/description")
            
            # Apply filters
            filtered = opportunities
            if filter_region != "All":
                filtered = [o for o in filtered if filter_region in o.iom_regions]
            if filter_theme != "All":
                filtered = [o for o in filtered if o.thematic_area == filter_theme]
            if filter_type != "All":
                filtered = [o for o in filtered if o.type == filter_type]
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
                        col_analyze, col_add = st.columns([2, 1])
                        
                        with col_analyze:
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
                        
                        with col_add:
                            if st.button("➕ Add to Export List"):
                                matches, relevance_level = get_opportunity_matches(selected_opp, prima_projects)
                                existing_ids = [m[0].id for m in st.session_state.selected_matches]
                                if selected_opp.id not in existing_ids:
                                    st.session_state.selected_matches.append((selected_opp, matches, relevance_level))
                                    st.success(f"✓ Added ({len(st.session_state.selected_matches)} total)")
                                else:
                                    st.warning("Already in list")
                
                # Export section
                st.markdown("---")
                st.subheader("📥 Export Selected Matches")
                
                if st.session_state.selected_matches:
                    st.markdown(f"**{len(st.session_state.selected_matches)} opportunities ready for export**")
                    
                    for i, (opp_item, matches, rel) in enumerate(st.session_state.selected_matches):
                        rel_emoji = "✅" if rel == "HIGH" else ("🟡" if rel == "MEDIUM" else "🔴")
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"{rel_emoji} **{opp_item.id}**: {opp_item.title[:60]}... ({rel})")
                        with col2:
                            if st.button("❌", key=f"remove_{opp_item.id}"):
                                st.session_state.selected_matches = [m for m in st.session_state.selected_matches if m[0].id != opp_item.id]
                                st.rerun()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📊 Export Matches to Excel",
                            data=generate_matches_excel(st.session_state.selected_matches),
                            file_name=f"Devex_Matches_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            type="primary"
                        )
                    with col2:
                        if st.button("🗑️ Clear List"):
                            st.session_state.selected_matches = []
                            st.rerun()
                else:
                    st.info("Select opportunities and click '➕ Add to Export List' to build your export.")
        
        # =====================================================================
        # TAB 3: FUNDING HIGHLIGHTS
        # =====================================================================
        with tab3:
            st.header("📰 Funding Highlights")
            st.markdown("Automate the RM Bulletin news collection, screening, and export")
            
            # Initialize session state for funding highlights
            if 'funding_news' not in st.session_state:
                st.session_state.funding_news = []
            
            # Sub-tabs for Funding Highlights workflow
            fh_tab1, fh_tab2, fh_tab3 = st.tabs(["📥 Input News", "📊 Review & Select", "📤 Export"])
            
            # -----------------------------------------------------------------
            # FUNDING HIGHLIGHTS - INPUT TAB
            # -----------------------------------------------------------------
            with fh_tab1:
                st.subheader("📥 Input Funding News")
                
                # Two methods: Auto-fetch or Manual paste
                input_method = st.radio(
                    "Choose input method:",
                    ["🔍 Auto-Fetch with Web Search", "📋 Manual Paste"],
                    horizontal=True
                )
                
                # Previous bulletin donors for deduplication (used by both methods)
                prev_donors = st.text_input(
                    "Donors in last bulletin (comma-separated, for duplicate detection)",
                    placeholder="Canada, Finland, Netherlands"
                )
                
                st.markdown("---")
                
                if input_method == "🔍 Auto-Fetch with Web Search":
                    st.markdown("""
                    **Automatically fetch donor news using Azure OpenAI Web Search.**
                    
                    This will search:
                    - DonorTracker, ReliefWeb, OECD/DAC, Devex, FTS
                    - Government ministry and agency press pages
                    
                    ⚠️ **Requires**: Azure OpenAI with `gpt-4.1` or later and `web_search_preview` enabled
                    """)
                    
                    # Manual credentials for web search
                    st.markdown("### 🔑 Azure OpenAI Credentials for Web Search")
                    st.caption("Web search requires `gpt-4.1` or later. You can use different credentials than the sidebar.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        ws_endpoint = st.text_input(
                            "Azure Endpoint",
                            value=azure_endpoint if use_ai else "",
                            placeholder="https://your-resource.openai.azure.com/",
                            key="ws_endpoint",
                            help="Your Azure OpenAI endpoint URL"
                        )
                    with col2:
                        ws_api_key = st.text_input(
                            "API Key",
                            type="password",
                            value=azure_key if use_ai else "",
                            key="ws_api_key",
                            help="Your Azure OpenAI API key"
                        )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        ws_deployment = st.text_input(
                            "Model Deployment Name",
                            value="gpt-4.1",
                            placeholder="gpt-4.1",
                            key="ws_deployment",
                            help="Must be gpt-4.1 or later for web search"
                        )
                    with col2:
                        ws_location = st.selectbox(
                            "Search location bias",
                            ["CH", "US", "GB", "DE", "FR"],
                            help="Country code for search location"
                        )
                    
                    search_days = st.slider("Search news from past N days", 7, 60, 30, 7)
                    
                    st.markdown("---")
                    
                    # Validate credentials
                    credentials_ready = ws_endpoint and ws_api_key and ws_deployment
                    
                    if not credentials_ready:
                        st.warning("⚠️ Please fill in all Azure credentials above.")
                    
                    # Test connection button
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🧪 Test Connection", disabled=not credentials_ready):
                            api_url = ws_endpoint.rstrip('/') + "/openai/v1/responses"
                            st.info(f"Testing: `{api_url}`")
                            
                            # Simple test request
                            test_payload = {
                                "model": ws_deployment,
                                "input": "Say hello in one word."
                            }
                            headers = {
                                "Content-Type": "application/json",
                                "api-key": ws_api_key
                            }
                            
                            try:
                                response = requests.post(
                                    api_url,
                                    headers=headers,
                                    json=test_payload,
                                    timeout=30
                                )
                                
                                if response.status_code == 200:
                                    st.success("✅ Connection successful! Responses API is working.")
                                else:
                                    st.error(f"❌ Error {response.status_code}: {response.text[:500]}")
                            except Exception as e:
                                st.error(f"❌ Connection failed: {str(e)}")
                    
                    with col2:
                        if st.button("🔍 Auto-Fetch Donor News", type="primary", disabled=not credentials_ready):
                            # Show the endpoint being used
                            api_url = ws_endpoint.rstrip('/') + "/openai/v1/responses"
                            st.info(f"🔗 Calling: `{api_url}` with model `{ws_deployment}`")
                            
                            with st.spinner("Searching the web for donor news... This may take 1-2 minutes..."):
                                try:
                                    raw_text, citations = fetch_donor_news_with_web_search(
                                        azure_endpoint=ws_endpoint,
                                        azure_key=ws_api_key,
                                        model_deployment=ws_deployment,
                                        days=search_days,
                                        user_location=ws_location
                                    )
                                    
                                    # Show raw response in expander
                                    with st.expander("📄 Raw Web Search Response"):
                                        st.text(raw_text)
                                        if citations:
                                            st.markdown("**Citations:**")
                                            for cite in citations[:10]:
                                                st.markdown(f"- [{cite.get('title', 'Link')}]({cite.get('url', '')})")
                                    
                                    # Parse into news items
                                    news_items = parse_web_search_results(raw_text, citations)
                                    
                                    # Check for duplicates
                                    if prev_donors:
                                        prev_donor_list = [d.strip() for d in prev_donors.split(',')]
                                        for news in news_items:
                                            if any(news.donor.lower() == d.lower() for d in prev_donor_list):
                                                news.is_duplicate = True
                                                news.duplicate_of = "Previous bulletin"
                                    
                                    st.session_state.funding_news = news_items
                                    
                                    st.success(f"✓ Fetched and parsed {len(news_items)} news items!")
                                    
                                    # Quick summary
                                    if news_items:
                                        st.markdown("### Quick Summary")
                                        for news in sorted(news_items, key=lambda x: -x.relevance_score)[:15]:
                                            if news.relevance_score >= 0.7:
                                                icon = "🟢"
                                            elif news.relevance_score >= 0.4:
                                                icon = "🟡"
                                            else:
                                                icon = "🔴"
                                            dup = " ⚠️ DUP" if news.is_duplicate else ""
                                            st.markdown(f"{icon} **{news.donor}** ({news.relevance_score:.0%}){dup}: {news.summary}")
                                    else:
                                        st.warning("No news items could be parsed from the response. Try manual paste instead.")
                                
                                except Exception as e:
                                    st.error(f"❌ Web search failed: {str(e)}")
                                    st.markdown("""
💡 **Troubleshooting:**

**401 Error (Invalid key/endpoint):**
- Endpoint should be: `https://YOUR-RESOURCE.openai.azure.com/`
- Make sure there's no extra path (remove `/openai/deployments/...` if present)
- API Key: Copy from Azure Portal → Your OpenAI Resource → Keys and Endpoint

**404 Error (Not found):**
- Check your model deployment name matches exactly
- Responses API requires `gpt-4.1` or later

**Web search blocked:**
- Contact your Azure admin to enable `web_search_preview`
- Run: `az feature unregister --name OpenAI.BlockedTools.web_search --namespace Microsoft.CognitiveServices`

**Fallback:** Use the Manual Paste option with Copilot Researcher output
                                    """)
                
                else:  # Manual Paste
                    st.markdown("""
                    **Paste news from Copilot Researcher or manual sources.**
                    
                    Expected format:
                    ```
                    Country: Summary of the news update. Source: Source Name
                    ```
                    """)
                    
                    news_input = st.text_area(
                        "Paste news items here",
                        height=300,
                        placeholder="""Canada: On November 17, the first Federal Budget under Mark Carney's Liberal Government was introduced. The budget aims to stimulate the economy. International assistance funding will decrease by CAD 470M in 2026. Source: COPA Ottawa

Finland: The Government of Finland announced EUR 13.5 million in humanitarian assistance. The funds will go to Ukraine, Nigeria, Sudan, Gaza, Yemen and Afghanistan. Source: Finnish Government Press Release

United Kingdom: Home Secretary announced sweeping reforms to the asylum system. Key changes include regular reviews of refugee status and stricter family reunion rules. Source: COPA London"""
                    )
                    
                    if st.button("🔄 Process News", type="primary"):
                        if news_input:
                            with st.spinner("Processing news items..."):
                                news_items = parse_funding_news(news_input)
                                
                                # Check for duplicates
                                if prev_donors:
                                    prev_donor_list = [d.strip() for d in prev_donors.split(',')]
                                    for news in news_items:
                                        if any(news.donor.lower() == d.lower() for d in prev_donor_list):
                                            news.is_duplicate = True
                                            news.duplicate_of = "Previous bulletin"
                                
                                st.session_state.funding_news = news_items
                            
                            st.success(f"✓ Processed {len(news_items)} news items")
                            
                            # Quick summary
                            st.markdown("### Quick Summary")
                            for news in sorted(news_items, key=lambda x: -x.relevance_score):
                                if news.relevance_score >= 0.7:
                                    icon = "🟢"
                                elif news.relevance_score >= 0.4:
                                    icon = "🟡"
                                else:
                                    icon = "🔴"
                                dup = " ⚠️ DUP" if news.is_duplicate else ""
                                st.markdown(f"{icon} **{news.donor}** ({news.relevance_score:.0%}){dup}: {news.summary}")
                        else:
                            st.warning("Please paste some news content first")
            
            # -----------------------------------------------------------------
            # FUNDING HIGHLIGHTS - REVIEW TAB
            # -----------------------------------------------------------------
            with fh_tab2:
                st.subheader("📊 Review & Select")
                
                if not st.session_state.funding_news:
                    st.info("No news items yet. Go to 'Input News' tab to add some.")
                else:
                    news_items = st.session_state.funding_news
                    
                    # Summary stats
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Items", len(news_items))
                    col2.metric("High Relevance", len([n for n in news_items if n.relevance_score >= 0.7]))
                    col3.metric("Duplicates", len([n for n in news_items if n.is_duplicate]))
                    col4.metric("Selected", len([n for n in news_items if n.selected_for_bulletin]))
                    
                    # Auto-select settings
                    st.markdown("### ⚙️ Auto-Select Settings")
                    col1, col2 = st.columns(2)
                    with col1:
                        min_relevance = st.slider("Minimum Relevance Score", 0.0, 1.0, 0.3, 0.1, key="review_min_relevance")
                    with col2:
                        max_per_donor = st.slider("Max Items per Donor", 1, 10, 5, key="review_max_per_donor")
                    
                    # Auto-select button
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🎯 Auto-Select Top Items"):
                            # Sort by relevance, exclude duplicates
                            candidates = [n for n in news_items if not n.is_duplicate and n.relevance_score >= min_relevance]
                            candidates.sort(key=lambda x: -x.relevance_score)
                            
                            # Apply max per donor limit
                            donor_counts = {}
                            selected_count = 0
                            
                            for news in news_items:
                                news.selected_for_bulletin = False
                            
                            for news in candidates:
                                donor_counts[news.donor] = donor_counts.get(news.donor, 0)
                                if donor_counts[news.donor] < max_per_donor and selected_count < 8:
                                    news.selected_for_bulletin = True
                                    donor_counts[news.donor] += 1
                                    selected_count += 1
                            
                            st.success(f"✓ Auto-selected {selected_count} items")
                            st.rerun()
                    
                    with col2:
                        if st.button("🔄 Clear Selection"):
                            for news in news_items:
                                news.selected_for_bulletin = False
                            st.rerun()
                    
                    st.markdown("---")
                    
                    # Filter
                    filter_relevance = st.selectbox(
                        "Filter by Relevance",
                        ["All", "High (≥70%)", "Medium (40-70%)", "Low (<40%)"]
                    )
                    
                    # Apply filter
                    filtered = news_items
                    if filter_relevance == "High (≥70%)":
                        filtered = [n for n in filtered if n.relevance_score >= 0.7]
                    elif filter_relevance == "Medium (40-70%)":
                        filtered = [n for n in filtered if 0.4 <= n.relevance_score < 0.7]
                    elif filter_relevance == "Low (<40%)":
                        filtered = [n for n in filtered if n.relevance_score < 0.4]
                    
                    # Display news items
                    for news in sorted(filtered, key=lambda x: -x.relevance_score):
                        if news.relevance_score >= 0.7:
                            color = "🟢"
                        elif news.relevance_score >= 0.4:
                            color = "🟡"
                        else:
                            color = "🔴"
                        
                        dup_badge = " ⚠️ DUPLICATE" if news.is_duplicate else ""
                        selected_badge = " ✅" if news.selected_for_bulletin else ""
                        verify_badge = " 🔍 VERIFY" if news.confidence_warnings else ""
                        
                        with st.expander(f"{color} **{news.donor}**: {news.title[:50]}...{dup_badge}{verify_badge}{selected_badge}"):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**Category:** {NEWS_CATEGORIES.get(news.category, news.category)}")
                                st.markdown(f"**Relevance:** {news.relevance_score:.0%}")
                                st.markdown(f"**Keywords:** {', '.join(news.migration_keywords_found[:6])}")
                                if news.source_url:
                                    st.markdown(f"**Source:** [{news.source}]({news.source_url}) ✅")
                                else:
                                    st.markdown(f"**Source:** {news.source} ⚠️ *No direct link*")
                                
                                # Show confidence warnings if any
                                if news.confidence_warnings:
                                    st.markdown("---")
                                    st.markdown("**⚠️ Verification Warnings:**")
                                    for warning in news.confidence_warnings:
                                        st.markdown(f"- {warning}")
                                
                                st.markdown("---")
                                st.markdown("**Full Text:**")
                                st.write(news.full_text)
                            
                            with col2:
                                # Selection checkbox
                                selected = st.checkbox(
                                    "Include",
                                    value=news.selected_for_bulletin,
                                    key=f"sel_{news.id}"
                                )
                                news.selected_for_bulletin = selected
                                
                                # Focal point
                                focal = st.text_input(
                                    "Assign to",
                                    value=news.focal_point,
                                    key=f"fp_{news.id}",
                                    placeholder="Focal point"
                                )
                                news.focal_point = focal
            
            # -----------------------------------------------------------------
            # FUNDING HIGHLIGHTS - EXPORT TAB
            # -----------------------------------------------------------------
            with fh_tab3:
                st.subheader("📤 Export Funding Highlights")
                
                selected_items = [n for n in st.session_state.funding_news if n.selected_for_bulletin]
                
                if not selected_items:
                    st.info("No items selected. Go to 'Review & Select' tab to select items.")
                else:
                    # Check for items needing verification
                    items_needing_verification = [n for n in selected_items if n.confidence_warnings]
                    if items_needing_verification:
                        st.warning(f"""
                        ⚠️ **Verification Required**: {len(items_needing_verification)} of {len(selected_items)} selected items 
                        have potential accuracy concerns. AI-generated summaries may contain inferences not explicitly 
                        stated in sources. Please verify claims against source URLs before including in official communications.
                        
                        Items to verify: {', '.join([n.donor for n in items_needing_verification])}
                        """)
                    
                    st.markdown(f"**{len(selected_items)} items ready for export**")
                    
                    # Preview
                    st.markdown("### Preview")
                    st.markdown("---")
                    st.markdown("**FUNDING HIGHLIGHTS**")
                    st.markdown("*Key updates about IOM's key donors from RMD, COPAs, and Country Offices in donor capitals.*")
                    st.markdown("")
                    
                    for news in sorted(selected_items, key=lambda x: -x.relevance_score):
                        summary = news.ai_summary if news.ai_summary else news.summary
                        verify_note = " 🔍" if news.confidence_warnings else ""
                        if news.source_url:
                            st.markdown(f"**{news.donor}:**{verify_note} {summary} *Source: [{news.source}]({news.source_url})*")
                        else:
                            st.markdown(f"**{news.donor}:**{verify_note} {summary} *Source: {news.source}* ⚠️")
                        st.markdown("")
                    
                    st.markdown("---")
                    
                    # Edit summaries
                    st.markdown("### ✏️ Edit Summaries")
                    for news in sorted(selected_items, key=lambda x: -x.relevance_score):
                        with st.expander(f"Edit: {news.donor}"):
                            current = news.ai_summary if news.ai_summary else news.summary
                            edited = st.text_area(
                                "Summary (50-150 words)",
                                value=current,
                                height=100,
                                key=f"edit_{news.id}"
                            )
                            news.ai_summary = edited
                            
                            word_count = len(edited.split())
                            if 50 <= word_count <= 150:
                                st.success(f"Word count: {word_count} ✓")
                            else:
                                st.warning(f"Word count: {word_count} (target: 50-150)")
                    
                    st.markdown("---")
                    
                    # Export buttons
                    st.markdown("### Download")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        text_export = generate_funding_highlights_export(st.session_state.funding_news)
                        st.download_button(
                            "📄 Download Text File",
                            data=text_export,
                            file_name=f"Funding_Highlights_{datetime.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain"
                        )
                    
                    with col2:
                        excel_export = generate_funding_highlights_excel(st.session_state.funding_news)
                        st.download_button(
                            "📊 Download Excel Report",
                            data=excel_export,
                            file_name=f"Funding_News_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    # Focal point summary
                    focal_points = {}
                    for news in selected_items:
                        if news.focal_point:
                            if news.focal_point not in focal_points:
                                focal_points[news.focal_point] = []
                            focal_points[news.focal_point].append(news.donor)
                    
                    if focal_points:
                        st.markdown("### 📧 Focal Point Assignments")
                        for fp, donors in focal_points.items():
                            st.markdown(f"**{fp}:** {', '.join(donors)}")


if __name__ == "__main__":
    main()
