import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import TypedDict, Dict, List, Union, Any

global_config = None

class FieldSchema(TypedDict):
    title: str
    type: str
    length: Union[int, str]
    introduction: str

class TableSchema(TypedDict):
    introduction: str
    fields: List[FieldSchema]

class TableMapping(TypedDict):
    name: str
    introduction: str

class KeywordMapping(TypedDict):
    key: str
    tables: List[TableMapping]

class SystemConfig(TypedDict):
    keyword: str
    url: str
    schemas: Dict[str, TableSchema]
    keyword_maps: List[KeywordMapping]

@dataclass
class Config:
    rag_url: str
    mcp_db_dict: Dict[str, SystemConfig]
    PROMPT_TRIGGER_KEYWORDS: list[str]
    rag_classification: list[str]

class ConfigObject:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObject(value))
            else:
                setattr(self, key, value)

# Load config from JSON file
def load_config():
    global global_config
    try:
        with (Path(__file__).resolve().parents[1] / 'config.jsonc').open(encoding='utf-8') as f:
            """Load JSON file with comments (// and /* */)"""
            content = f.read()
            
            # Remove // comments
            content = re.sub(r'//\s+.*\n', '', content)
            # Load configuration at the beginning
            global_config = Config(**json.loads(content))
            return
    except FileNotFoundError:
        print("Error: config.jsonc file not found")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config.jsonc: {e}")
        print(f"Error at line {e.lineno}, column {e.colno}")
        exit(1)

