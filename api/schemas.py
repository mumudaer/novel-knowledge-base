"""
FastAPI 数据模型定义
使用 Pydantic 定义请求和响应的数据结构
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


# ===================== 通用响应 =====================

class BaseResponse(BaseModel):
    success: bool = True
    message: str = "OK"
    data: Any = None


# ===================== 世界观查询 =====================

class WorldSettingsQuery(BaseModel):
    book_name: Optional[str] = None
    module: Optional[str] = None
    tags: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=100)


class WorldSettingsItem(BaseModel):
    book_name: str
    author: str
    category: str
    module: str
    entity: str
    content: str
    tags: str
    daily_life: str = ""
    taboos: str = ""
    conflict_roots: str = ""
    geography: str = ""
    economy: str = ""
    culture: str = ""
    causal_chain: str = ""


class WorldTimelineItem(BaseModel):
    book_name: str
    era_or_year: str
    event_name: str
    event_description: str
    impact: str


# ===================== 人物查询 =====================

class CharacterProfileQuery(BaseModel):
    book_name: Optional[str] = None
    character_name: Optional[str] = None
    role_type: Optional[str] = None
    limit: int = Field(default=20, ge=1, le=100)


class CharacterProfileItem(BaseModel):
    book_name: str
    author: str
    category: str
    name: str
    role_type: str
    appearance: str = ""
    quirks: str = ""
    identity: str = ""
    motivation: str = ""
    internal_conflict: str = ""
    fatal_flaw: str = ""
    symbolism: str = ""
    personality: str = ""
    relation_to_mc: str = ""
    relations_to_others: str = ""
    climax_or_fate: str = ""
    background: str = ""
    desire_vs_need: str = ""
    secrets: str = ""
    fears: str = ""
    social_masks: str = ""
    growth_cost: str = ""
    speech_samples: str = ""
    behavior_samples: str = ""
    relationship_evolution: str = ""
    abilities: str = ""
    arc_trajectory: str = ""
    internal_dilemma: str = ""


class CharacterSpeechStyleItem(BaseModel):
    book_name: str
    character_name: str
    catchphrases: str = ""
    vocabulary_preference: str = ""
    sentence_pattern: str = ""
    tone_contexts_json: str = ""
    dialogue_samples_json: str = ""


class CharacterRelationshipItem(BaseModel):
    book_name: str
    character_a: str
    character_b: str
    timeline_json: str = ""


# ===================== 大纲/细纲查询 =====================

class PlotStructureQuery(BaseModel):
    book_name: str


class BookStructureItem(BaseModel):
    book_name: str
    structure_type: str
    act_breakdown_json: str
    surface_theme: str = ""
    deep_theme: str = ""


class PlotLineItem(BaseModel):
    book_name: str
    line_type: str
    theme: str
    chapter_distribution: str
    milestones_json: str


class EmotionalArcItem(BaseModel):
    book_name: str
    arc_data_json: str


class CoolPointItem(BaseModel):
    book_name: str
    distribution_json: str
    rhythm_pattern: str


class ForeshadowingItem(BaseModel):
    book_name: str
    hook_name: str
    planted_chapter: str
    planned_payoff: str
    status: str
    resolved_chapter: str
    resolution_excerpt: str = ""


class ChapterFunctionItem(BaseModel):
    book_name: str
    chapter_id: str
    function_type: str
    structure_pattern_json: str
    hook_type: str
    hook_content: str
    information_gap_json: str
    active_plotlines: str = ""


# ===================== 写作风格查询 =====================

class StyleFingerprintQuery(BaseModel):
    book_name: Optional[str] = None
    author: Optional[str] = None


class AuthorFingerprintItem(BaseModel):
    book_name: str
    category: str
    verbs: str
    adjectives: str
    imagery: str
    transitions: str
    negative_prompts: str
    narrative_perspective: str
    sentence_rhythm: str


# ===================== 样本库查询 =====================

class ExcerptSearchQuery(BaseModel):
    query: str
    book_name: Optional[str] = None
    category: Optional[str] = None
    scene_type: Optional[str] = None
    limit: int = Field(default=5, ge=1, le=20)


class DialogueSampleItem(BaseModel):
    book_name: str
    chapter_id: str
    scene_type: str
    original_text: str
    emotional_tension: str
    subtext: str
    plot_function: str


class DescriptionSampleItem(BaseModel):
    book_name: str
    chapter_id: str
    description_type: str
    original_text: str
    technique_analysis: str
    sensory_details: str


class ExcerptSearchResult(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]
    distance: float = 0.0


# ===================== 新增接口模型 =====================

class RevelationPacingItem(BaseModel):
    book_name: str
    revelation_name: str
    reveal_chapter: str
    reveal_method: str
    impact: str


class TransitionSampleItem(BaseModel):
    book_name: str
    chapter_id: str
    transition_type: str
    original_text: str
    technique_analysis: str


class StyleSummaryItem(BaseModel):
    book_name: str
    category: str
    summary_type: str
    scene_or_desc_type: str
    style_description: str
    key_features: str


class BookStatisticsItem(BaseModel):
    book_name: str
    total_words: int
    avg_chapter_words: int
    min_chapter_words: int
    max_chapter_words: int
    median_chapter_words: int
    dialogue_ratio: float
    description_ratio: float
    avg_paragraph_length: float
    short_para_ratio: float
    medium_para_ratio: float
    long_para_ratio: float
    rhythm_pattern: str
