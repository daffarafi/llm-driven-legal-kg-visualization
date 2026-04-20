"""
Prompt builder for KG extraction.

Supports templating with {kg_schema} placeholder.
Schema can be loaded from:
  - Google Sheets (by SCHEMA_ID from a worksheet with SCHEMA_ID + SCHEMA columns)
  - Local JSON file (config/kg_schema.json)
"""

import json
import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# Default prompt template
# ============================================================

DEFAULT_PROMPT_TEMPLATE = """You are a Knowledge Graph extractor for Indonesian legal documents.
Given a chunk of legal text, extract entities (nodes) and relationships (edges) according to the ontology below.

{kg_schema}

## Critical Rules

### Deduplication
1. Every node MUST have a unique `id`, `type`, and `label`.
2. Use the format `{{Type}}_{{short_label}}` for ids (e.g., `Pasal_27`, `Sanksi_pidana_penjara_6_tahun`).
3. If the same entity appears multiple times in the text, reuse the SAME id — do NOT create duplicates.
4. Create only ONE `Regulasi` node for the document being processed.

### Quality over Quantity
5. Prefer fewer, high-quality nodes over many low-quality ones.
6. Only create `KonsepHukum` for terms that are **explicitly defined** with a definition in the text.
7. `PerbuatanHukum` must be a **specific, concrete action**, not a vague description.
8. Every `Sanksi` node must contain the FULL penalty text including duration and/or fine amount.

### Hierarchy
9. Maintain strict hierarchy: Regulasi → Bab → Bagian (if exists) → Pasal → Ayat.
10. Every Pasal should be connected to its parent Bab (or Bagian) via MEMUAT if known from the text.
11. If a Pasal has multiple ayat, create Ayat nodes and connect them via MEMILIKI_AYAT.

### Relationships
12. Each Pasal/Ayat that regulates an action MUST have a MENGATUR edge.
13. Each Pasal/Ayat that specifies a sanction MUST have both MENGATUR and MENETAPKAN_SANKSI.
14. MERUJUK edges should only be created for **explicit cross-references**.

## Output Format

Output MUST be valid JSON:
```json
{{
  "nodes": [
    {{"id": "Pasal_27", "type": "Pasal", "label": "Pasal 27", "content": "brief description"}}
  ],
  "edges": [
    {{"source": "Pasal_27", "target": "PerbuatanHukum_X", "type": "MENGATUR"}}
  ]
}}
```
"""


# ============================================================
# Schema loaders
# ============================================================

def load_schema_from_file(schema_path: str = None) -> str:
    """Load KG schema from a local JSON file and format as markdown.
    
    Returns:
        Schema as a markdown string ready for prompt injection.
    """
    if schema_path is None:
        schema_path = Path(__file__).parent.parent.parent / "config" / "kg_schema.json"
    
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    
    return _schema_dict_to_markdown(schema)


def load_schema_from_gsheets(
    schema_id: str,
    google_util,
    spreadsheet_id: str,
    worksheet_name: str = "KG_SCHEMA",
) -> str:
    """Load KG schema from Google Sheets by SCHEMA_ID.
    
    Expects a worksheet with headers: SCHEMA_ID, SCHEMA
    The SCHEMA column contains the schema text (markdown).
    
    Args:
        schema_id: The ID of the schema to load (e.g., "KGS_001")
        google_util: GoogleUtil instance with credentials
        spreadsheet_id: Google Spreadsheet ID
        worksheet_name: Name of the worksheet containing schemas
        
    Returns:
        Schema text as string.
        
    Raises:
        ValueError: If schema_id is not found.
    """
    rows = google_util.retrieve_worksheet(spreadsheet_id, worksheet_name)
    
    if not rows:
        raise ValueError(f"Worksheet '{worksheet_name}' is empty")
    
    # Find column indices from header
    headers = [h.strip().upper() for h in rows[0]]
    try:
        id_col = headers.index("SCHEMA_ID")
        schema_col = headers.index("SCHEMA")
    except ValueError:
        raise ValueError(f"Worksheet must have SCHEMA_ID and SCHEMA columns. Found: {headers}")
    
    # Find matching row
    for row in rows[1:]:
        if len(row) > max(id_col, schema_col) and row[id_col].strip() == schema_id:
            schema_text = row[schema_col].strip()
            if not schema_text:
                raise ValueError(f"Schema '{schema_id}' found but SCHEMA column is empty")
            logger.info(f"Loaded schema '{schema_id}' ({len(schema_text)} chars)")
            return schema_text
    
    available = [row[id_col] for row in rows[1:] if len(row) > id_col and row[id_col].strip()]
    raise ValueError(f"Schema '{schema_id}' not found. Available: {available}")


# ============================================================
# Prompt builder
# ============================================================

def build_prompt(
    template: str = None,
    kg_schema: str = None,
    schema_id: str = None,
    google_util=None,
    spreadsheet_id: str = None,
    schema_worksheet: str = "KG_SCHEMA",
    **extra_vars,
) -> str:
    """Build a system prompt by injecting {kg_schema} into the template.
    
    Schema source priority:
      1. kg_schema (direct string)
      2. schema_id + google_util (fetch from GSheets)
      3. Local file (config/kg_schema.json)
    
    Args:
        template: Prompt template with {kg_schema} placeholder.
                  If None, uses DEFAULT_PROMPT_TEMPLATE.
        kg_schema: Schema text to inject directly.
        schema_id: Schema ID to fetch from Google Sheets.
        google_util: GoogleUtil instance (required if schema_id is used).
        spreadsheet_id: Google Spreadsheet ID (required if schema_id is used).
        schema_worksheet: Worksheet name for schemas.
        **extra_vars: Additional template variables to substitute.
    
    Returns:
        Complete system prompt string.
    """
    if template is None:
        template = DEFAULT_PROMPT_TEMPLATE
    
    # Resolve schema
    if kg_schema is None:
        if schema_id and google_util and spreadsheet_id:
            kg_schema = load_schema_from_gsheets(
                schema_id, google_util, spreadsheet_id, schema_worksheet
            )
        else:
            kg_schema = load_schema_from_file()
    
    # Inject into template
    prompt = template.replace("{kg_schema}", kg_schema)
    
    # Substitute any extra variables
    for key, value in extra_vars.items():
        prompt = prompt.replace(f"{{{key}}}", str(value))
    
    return prompt


# ============================================================
# Helpers
# ============================================================

def _schema_dict_to_markdown(schema: dict) -> str:
    """Convert a schema JSON dict to markdown tables."""
    lines = []

    # Node Types
    lines.append("## Valid Node Types\n")
    lines.append("| Type | Description | Example Labels |")
    lines.append("|------|-------------|----------------|")
    for ntype, ndef in schema["node_types"].items():
        examples = ", ".join(f'"{e}"' for e in ndef.get("example_labels", []))
        desc = ndef["description"]
        props = ndef.get("properties", {})
        if "jenis" in props:
            jenis_enum = ", ".join(props["jenis"].get("enum", []))
            desc += f' Include a "jenis" property: one of [{jenis_enum}].'
        lines.append(f"| {ntype} | {desc} | {examples} |")

    lines.append("")

    # Edge Types
    lines.append("## Valid Relation Types\n")
    lines.append("| Type | Direction | Description |")
    lines.append("|------|-----------|-------------|")
    for etype, edef in schema["edge_types"].items():
        sources = "/".join(edef.get("allowed_sources", []))
        targets = "/".join(edef.get("allowed_targets", []))
        lines.append(f"| {etype} | {sources} → {targets} | {edef['description']} |")

    return "\n".join(lines)
