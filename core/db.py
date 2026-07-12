"""
数据库管理模块
封装 SQLite 数据库的初始化、表结构管理、数据操作
"""
import sqlite3
import logging
import threading
from typing import Optional, List, Dict, Any
from config.settings import SQLITE_PATH

logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite 数据库管理器"""

    # 表结构定义
    TABLE_SCHEMAS = {
        # Stage A: 剧情脉络
        "plot_arcs": "(chapter_id TEXT PRIMARY KEY, book_name TEXT, category TEXT, summary TEXT, character_state_json TEXT)",
        
        # Stage B: 写作技法
        "skills": "(id TEXT PRIMARY KEY, book_name TEXT, chapter_id TEXT, category TEXT, scene_type TEXT, skill_name TEXT, analysis TEXT, original_example TEXT, tags TEXT)",
        
        # Stage C: 文风指纹
        "author_fingerprints": "(id TEXT PRIMARY KEY, book_name TEXT, category TEXT, verbs TEXT, adjectives TEXT, imagery TEXT, transitions TEXT, negative_prompts TEXT, narrative_perspective TEXT, sentence_rhythm TEXT)",
        "sensory_mappings": "(id TEXT PRIMARY KEY, book_name TEXT, chapter_id TEXT, category TEXT, emotion TEXT, show_not_tell TEXT, analysis TEXT)",
        
        # Stage D: 世界观与人物
        "world_settings": "(id TEXT PRIMARY KEY, book_name TEXT, author TEXT, category TEXT, module TEXT, entity TEXT, content TEXT, tags TEXT, daily_life TEXT, taboos TEXT, conflict_roots TEXT, geography TEXT, economy TEXT, culture TEXT, causal_chain TEXT, rules_exceptions TEXT)",
        "golden_finger": "(id TEXT PRIMARY KEY, book_name TEXT, name TEXT, type TEXT, abilities TEXT, upgrade_path TEXT, limitations TEXT, cost_layers TEXT, interaction_with_plot TEXT, source_chapter TEXT)",
        "character_profiles": "(id TEXT PRIMARY KEY, book_name TEXT, author TEXT, category TEXT, name TEXT, role_type TEXT, appearance TEXT, quirks TEXT, identity TEXT, motivation TEXT, internal_conflict TEXT, fatal_flaw TEXT, personality TEXT, relation_to_mc TEXT, relations_to_others TEXT, climax_or_fate TEXT, background TEXT, desire_vs_need TEXT, secrets TEXT, fears TEXT, social_masks TEXT, growth_cost TEXT, speech_samples TEXT, behavior_samples TEXT, relationship_evolution TEXT, abilities TEXT, arc_trajectory TEXT, internal_dilemma TEXT, decision_pattern TEXT, cognitive_bias TEXT, transformation_trigger TEXT, contrast_design TEXT)",
        "world_timeline": "(id TEXT PRIMARY KEY, book_name TEXT, era_or_year TEXT, event_name TEXT, event_description TEXT, impact TEXT)",
        
        # Stage D 扩展: 势力关系网络与设定演变
        "faction_networks": "(id TEXT PRIMARY KEY, book_name TEXT, faction_a TEXT, faction_b TEXT, relation_type TEXT, relation_detail TEXT, stability TEXT, key_events TEXT)",
        "setting_evolutions": "(id TEXT PRIMARY KEY, book_name TEXT, setting_module TEXT, setting_entity TEXT, chapter_range TEXT, evolution_type TEXT, before_state TEXT, after_state TEXT, trigger_event TEXT)",
        
        # Stage E: 宏观大纲
        "macro_outlines": "(id TEXT PRIMARY KEY, book_name TEXT, category TEXT, volume_index INTEGER, chapter_range TEXT, theme TEXT, conflict TEXT, beats_json TEXT, arc TEXT)",
        "plot_foreshadowing": "(id TEXT PRIMARY KEY, book_name TEXT, hook_name TEXT, planted_chapter TEXT, planned_payoff TEXT, status TEXT, payoff_timing TEXT, scope_type TEXT, resolved_chapter TEXT, resolution_excerpt TEXT, last_advanced_chapter TEXT)",
        "entity_state_tracker": "(id TEXT PRIMARY KEY, book_name TEXT, entity_name TEXT, chapter_range TEXT, current_state_json TEXT)",
        "chapter_functions": "(id TEXT PRIMARY KEY, book_name TEXT, chapter_id TEXT, function_type TEXT, pacing_type TEXT, structure_pattern_json TEXT, hook_type TEXT, hook_intensity TEXT, hook_content TEXT, cool_point_type TEXT, arc_length TEXT, information_gap_json TEXT, active_plotlines TEXT)",
        
        # Stage F: 样本库
        "dialogue_samples": "(id TEXT PRIMARY KEY, book_name TEXT, chapter_id TEXT, scene_type TEXT, original_text TEXT, emotional_tension TEXT, subtext TEXT, plot_function TEXT, writing_quality INTEGER DEFAULT 5)",
        "description_samples": "(id TEXT PRIMARY KEY, book_name TEXT, chapter_id TEXT, description_type TEXT, original_text TEXT, technique_analysis TEXT, sensory_details TEXT, writing_quality INTEGER DEFAULT 5)",
        
        # Stage G: 人物深度特征
        "character_speech_style": "(id TEXT PRIMARY KEY, book_name TEXT, character_name TEXT, catchphrases TEXT, vocabulary_preference TEXT, sentence_pattern TEXT, tone_contexts_json TEXT, dialogue_samples_json TEXT)",
        "character_behavior_marks": "(id TEXT PRIMARY KEY, book_name TEXT, character_name TEXT, habitual_actions TEXT, micro_expressions TEXT, defense_mechanisms TEXT, behavior_samples_json TEXT)",
        "character_relationship_dynamics": "(id TEXT PRIMARY KEY, book_name TEXT, character_a TEXT, character_b TEXT, timeline_json TEXT)",
        
        # Stage H: 全书宏观分析
        "book_structure": "(id TEXT PRIMARY KEY, book_name TEXT, act_breakdown_json TEXT, surface_theme TEXT, deep_theme TEXT)",
        "plot_lines": "(id TEXT PRIMARY KEY, book_name TEXT, line_type TEXT, theme TEXT, chapter_distribution TEXT, milestones_json TEXT)",
        "emotional_arc": "(id TEXT PRIMARY KEY, book_name TEXT, arc_data_json TEXT)",
        "climax_point_distribution": "(id TEXT PRIMARY KEY, book_name TEXT, distribution_json TEXT)",
        "symbol_system": "(id TEXT PRIMARY KEY, book_name TEXT, symbols_json TEXT)",
        
        # Stage F 扩展: 转场样本与风格总结
        "transition_samples": "(id TEXT PRIMARY KEY, book_name TEXT, chapter_id TEXT, transition_type TEXT, original_text TEXT, technique_analysis TEXT, writing_quality INTEGER DEFAULT 5)",
        "style_summaries": "(id TEXT PRIMARY KEY, book_name TEXT, category TEXT, summary_type TEXT, scene_or_desc_type TEXT, style_description TEXT, key_features TEXT)",
        
        # Stage H 扩展: 信息揭露节奏、章节模式、情感转变铺垫
        "revelation_pacing": "(id TEXT PRIMARY KEY, book_name TEXT, revelation_name TEXT, reveal_chapter TEXT, reveal_method TEXT, impact TEXT)",
        "chapter_patterns": "(id TEXT PRIMARY KEY, book_name TEXT, opening_patterns TEXT, ending_patterns TEXT, common_transitions TEXT)",
        "emotion_transition_patterns": "(id TEXT PRIMARY KEY, book_name TEXT, transition_type TEXT, foreshadowing_method TEXT, original_example TEXT)",
        
        # Stage H 扩展: 信息管理策略、高潮构建链、冲突升级阶梯
        "information_management": "(id TEXT PRIMARY KEY, book_name TEXT, strategy_type TEXT, target_info TEXT, conceal_method TEXT, reveal_timing TEXT, dramatic_purpose TEXT)",
        "climax_buildup_chains": "(id TEXT PRIMARY KEY, book_name TEXT, climax_name TEXT, climax_chapter TEXT, buildup_steps_json TEXT, tension_escalation TEXT)",
        "conflict_escalation": "(id TEXT PRIMARY KEY, book_name TEXT, conflict_line TEXT, escalation_steps_json TEXT, escalation_pattern TEXT)",
        
        # Stage F 扩展: 叙事距离控制与 Show vs Tell 策略
        "narrative_distance": "(id TEXT PRIMARY KEY, book_name TEXT, chapter_id TEXT, distance_type TEXT, trigger_reason TEXT, original_example TEXT, writing_quality INTEGER DEFAULT 5)",
        "show_tell_patterns": "(id TEXT PRIMARY KEY, book_name TEXT, chapter_id TEXT, pattern_type TEXT, ratio_estimate TEXT, switching_triggers TEXT, original_example TEXT, writing_quality INTEGER DEFAULT 5)",
        
        # Stage I: 纯统计模块
        "book_statistics": "(id TEXT PRIMARY KEY, book_name TEXT, total_words INTEGER, avg_chapter_words INTEGER, min_chapter_words INTEGER, max_chapter_words INTEGER, median_chapter_words INTEGER, dialogue_ratio REAL, description_ratio REAL, avg_paragraph_length REAL, short_para_ratio REAL, medium_para_ratio REAL, long_para_ratio REAL, rhythm_pattern TEXT)",
        
        # ===== 通用类型补强 =====
        # 感情线追踪（言情/爱情/所有有CP的作品）
        "romance_lines": "(id TEXT PRIMARY KEY, book_name TEXT, couple_a TEXT, couple_b TEXT, line_type TEXT, development_stages_json TEXT, sweet_points_json TEXT, angst_points_json TEXT, interaction_patterns_json TEXT, resolution TEXT)",
        
        # 线索与推理链（悬疑/推理）
        "mystery_clues": "(id TEXT PRIMARY KEY, book_name TEXT, clue_name TEXT, clue_type TEXT, planted_chapter TEXT, payoff_chapter TEXT, red_herring INTEGER, misdirection_method TEXT, reasoning_chain_json TEXT, twist_design TEXT)",
        
        # 恐惧/氛围构建链（克苏鲁/恐怖/悬疑）
        "fear_building": "(id TEXT PRIMARY KEY, book_name TEXT, fear_type TEXT, building_steps_json TEXT, atmosphere_techniques_json TEXT, climax_moment TEXT, original_example TEXT)",
        
        # 升级/成长体系（玄幻/仙侠/游戏竞技/职场）
        "progression_systems": "(id TEXT PRIMARY KEY, book_name TEXT, system_type TEXT, levels_json TEXT, upgrade_conditions_json TEXT, power_comparison_json TEXT, milestones_json TEXT, growth_pattern TEXT)",
        
        # 类型特定技法（通用灵活表，通过 genre_tag 区分类型）
        "genre_specific_techniques": "(id TEXT PRIMARY KEY, book_name TEXT, genre_tag TEXT, technique_name TEXT, technique_category TEXT, analysis TEXT, original_example TEXT, applicable_scenarios TEXT)",
        
        # 动作/战斗场景范文（武侠/玄幻/竞技）
        "action_scene_samples": "(id TEXT PRIMARY KEY, book_name TEXT, chapter_id TEXT, action_type TEXT, original_text TEXT, technique_analysis TEXT, pacing_analysis TEXT, sensory_details TEXT, writing_quality INTEGER DEFAULT 5)",
        
        # 书籍元数据（从文件信息自动生成）
        "book_metadata": "(id TEXT PRIMARY KEY, book_name TEXT, author TEXT, category TEXT, genre_tags TEXT, total_chapters INTEGER, total_words INTEGER, description TEXT, added_at TEXT)",
        
        # ===== 覆盖率补强 =====
        # 高潮段落/名场面原文提取
        "climax_excerpts": "(id TEXT PRIMARY KEY, book_name TEXT, chapter_id TEXT, excerpt_type TEXT, original_text TEXT, technique_analysis TEXT, emotional_impact TEXT, writing_quality INTEGER DEFAULT 5)",
        
        # 章节开头/结尾范文
        "chapter_opening_ending_samples": "(id TEXT PRIMARY KEY, book_name TEXT, chapter_id TEXT, sample_position TEXT, original_text TEXT, technique_analysis TEXT, hook_type TEXT)",
        
        # 金句/名句
        "memorable_quotes": "(id TEXT PRIMARY KEY, book_name TEXT, chapter_id TEXT, quote_text TEXT, context TEXT, technique_analysis TEXT, quote_type TEXT, writing_quality INTEGER DEFAULT 5)",
        
        # 多视角切换模式
        "pov_switching_patterns": "(id TEXT PRIMARY KEY, book_name TEXT, pattern_type TEXT, pov_characters_json TEXT, switching_triggers TEXT, frequency TEXT, original_example TEXT)",
        
        # ===== 知识库搜索支撑层 =====
        # 正文质量评审（评审结果存储在知识库侧，方便历史对比）
        "chapter_reviews": "(id TEXT PRIMARY KEY, project_name TEXT, chapter_index INTEGER, overall_score REAL, dimension_scores_json TEXT, issues_json TEXT, suggestions_json TEXT, rewrite_samples_json TEXT, benchmark_books TEXT, reviewed_at TEXT)",
        
        # 知识库引用追踪（记录创作过程中引用了哪些知识库条目）
        "kb_references": "(id TEXT PRIMARY KEY, project_name TEXT, target_type TEXT, target_id TEXT, ref_book TEXT, ref_table TEXT, ref_id TEXT, ref_content TEXT, usage_purpose TEXT)",
        
        # 知识库搜索历史（记录 Reasonix 创作 skill 的搜索请求）
        "search_logs": "(id TEXT PRIMARY KEY, project_name TEXT, search_type TEXT, query_text TEXT, result_summary TEXT, result_count INTEGER, timestamp TEXT)",
        
        # ===== 高级功能层 =====
        # 跨书对比分析（Stage L）
        "cross_book_comparisons": "(id TEXT PRIMARY KEY, comparison_dimension TEXT, books_analyzed TEXT, common_patterns_json TEXT, unique_features_json TEXT, best_practices TEXT, created_at TEXT)",
        
        # 常见错误模式库（Stage M）
        "common_mistakes": "(id TEXT PRIMARY KEY, dimension TEXT, mistake_name TEXT, typical_manifestation TEXT, frequency INTEGER, correction_direction TEXT, benchmark_example TEXT, benchmark_book TEXT, created_at TEXT)",
        
        # 技法组合模板（Stage N）
        "technique_combinations": "(id TEXT PRIMARY KEY, scene_type TEXT, combo_name TEXT, technique_sequence_json TEXT, technique_roles_json TEXT, applicable_scenarios TEXT, variations TEXT, benchmark_book TEXT, original_example TEXT, created_at TEXT)",
        
        # Stage O: 事件因果图谱
        "story_events": "(id TEXT PRIMARY KEY, book_name TEXT, chapter_id TEXT, event_name TEXT, event_summary TEXT, event_type TEXT, characters_involved TEXT, significance TEXT)",
        "event_causal_edges": "(id TEXT PRIMARY KEY, book_name TEXT, source_event_id TEXT, target_event_id TEXT, relation_type TEXT, relation_detail TEXT)",
        
        # 题材裁决规则（后处理聚合）
        "genre_rules": "(id TEXT PRIMARY KEY, genre TEXT, technique_name TEXT, frequency INTEGER, priority_rank INTEGER, applicable_scenarios TEXT, benchmark_books TEXT, updated_at TEXT)",
        
        # 质量自检记录（Stage Q）
        "quality_checks": "(id TEXT PRIMARY KEY, book_name TEXT, stage TEXT, chapter_id TEXT, severity TEXT, description TEXT, suggestion TEXT, detail_json TEXT, checked_at TEXT)",
    }

    CHAPTER_ID_TABLES = ["skills", "plot_arcs", "sensory_mappings", "dialogue_samples", "description_samples", "chapter_functions"]

    def __init__(self, db_path: Optional[str] = None):
        """初始化数据库连接"""
        self.db_path = db_path or SQLITE_PATH
        # 使用线程本地存储，每个线程独立的 SQLite 连接，避免多线程并发写入冲突
        self._local = threading.local()

    def connect(self) -> sqlite3.Connection:
        """获取当前线程的数据库连接（线程本地）"""
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                check_same_thread=False,
            )
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            self._local.conn = conn
        return conn

    def close(self):
        """关闭当前线程的数据库连接"""
        conn = getattr(self._local, 'conn', None)
        if conn:
            conn.close()
            self._local.conn = None

    def init_tables(self):
        """初始化所有表结构"""
        conn = self.connect()
        cursor = conn.cursor()

        print("🔍 正在执行数据库结构强制校验...")
        for table_name, schema in self.TABLE_SCHEMAS.items():
            try:
                cols = [
                    row[1]
                    for row in cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
                ]
                need_rebuild = len(cols) == 0
                
                # 检查是否需要 chapter_id 字段
                if (
                    not need_rebuild
                    and table_name in self.CHAPTER_ID_TABLES
                    and "chapter_id" not in cols
                ):
                    need_rebuild = True

                if need_rebuild:
                    print(f"🔧 表 [{table_name}] 结构缺失或不兼容，正在强制重建...")
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                    cursor.execute(f"CREATE TABLE {table_name} {schema}")
                else:
                    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} {schema}")
            except Exception as e:
                print(f"⚠️ 表 [{table_name}] 校验异常: {e}，强制重建...")
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                cursor.execute(f"CREATE TABLE {table_name} {schema}")

        conn.commit()
        
        # 迁移: 添加 writing_quality 列 (v2.0)
        _WQ_TABLES = ["dialogue_samples", "description_samples", "transition_samples",
                       "narrative_distance", "show_tell_patterns", "action_scene_samples",
                       "climax_excerpts", "memorable_quotes"]
        for t in _WQ_TABLES:
            try:
                cols = [row[1] for row in cursor.execute(f"PRAGMA table_info({t})").fetchall()]
                if cols and "writing_quality" not in cols:
                    cursor.execute(f"ALTER TABLE {t} ADD COLUMN writing_quality INTEGER DEFAULT 5")
                    logger.info(f"✅ 迁移 {t}: 添加 writing_quality 列")
            except Exception:
                pass
        
        print("✅ 数据库结构校验完毕。")

    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """执行 SQL 查询"""
        conn = self.connect()
        return conn.execute(query, params)

    def executemany(self, query: str, params_list: List[tuple]):
        """批量执行 SQL"""
        conn = self.connect()
        conn.executemany(query, params_list)

    def commit(self):
        """提交当前线程的事务（3 次指数退避重试）"""
        conn = getattr(self._local, 'conn', None)
        if conn:
            import time
            for attempt in range(3):
                try:
                    conn.commit()
                    return
                except Exception as e:
                    if attempt == 2:
                        logger.error(f"SQLite commit 3次重试均失败: {e}")
                        raise
                    logger.warning(f"SQLite commit 失败 (尝试 {attempt+1}/3): {e}")
                    time.sleep(0.5 * (2 ** attempt))

    from contextlib import contextmanager

    @contextmanager
    def transaction(self):
        """事务上下文管理器 — 自动 BEGIN/COMMIT/ROLLBACK"""
        conn = self.connect()
        conn.execute("BEGIN")
        try:
            yield conn.cursor()
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def get_existing_ids(self, table: str, book_name: str, id_column: str = "chapter_id") -> set:
        """获取已存在的 ID 集合"""
        try:
            cursor = self.execute(
                f"SELECT DISTINCT {id_column} FROM {table} WHERE book_name = ?",
                (book_name,),
            )
            return {row[0] for row in cursor.fetchall()}
        except Exception:
            return set()

    def count_records(self, table: str, book_name: str) -> int:
        """统计指定书籍的记录数"""
        try:
            cursor = self.execute(
                f"SELECT COUNT(*) FROM {table} WHERE book_name = ?",
                (book_name,),
            )
            return cursor.fetchone()[0]
        except Exception:
            return 0


# 全局数据库管理器实例
_global_db_manager: Optional[DatabaseManager] = None
_db_manager_lock = threading.Lock()


def get_db_manager() -> DatabaseManager:
    """获取全局数据库管理器实例（线程安全）"""
    global _global_db_manager
    if _global_db_manager is None:
        with _db_manager_lock:
            # 双重检查锁定
            if _global_db_manager is None:
                _global_db_manager = DatabaseManager()
    return _global_db_manager
