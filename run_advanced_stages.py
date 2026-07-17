"""
高级功能 Stage 独立执行脚本
在所有书籍的基础 Stage（A-I）处理完成后，手动运行此脚本
执行 Stage L（跨书对比）、Stage M（错误模式）、Stage N（技法组合）

使用方式：
    python run_advanced_stages.py              # 执行全部高级功能
    python run_advanced_stages.py --only L     # 只执行 Stage L
    python run_advanced_stages.py --only M     # 只执行 Stage M
    python run_advanced_stages.py --only N     # 只执行 Stage N
    python run_advanced_stages.py --skip M     # 执行 L 和 N，跳过 M
    python run_advanced_stages.py --incremental  # 增量模式：只处理新增书籍

增量处理说明：
    新增小说后，先运行 novel_analyzer.py 完成基础构建，
    然后运行本脚本 --incremental 模式，只会对新增书籍进行跨书对比更新，
    不会重跑全部已处理书籍。
"""

import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import BASE_DIR
from core.db import get_db_manager
from core.chroma_client import get_chroma_manager
from core.ollama_client import get_ollama_client
from core.utils import load_manifest, save_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(BASE_DIR, "advanced_stages.log"),
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)


def run_stage_l():
    """Stage L: 跨书对比分析"""
    from stages.stage_l import StageL

    print("\n" + "=" * 50)
    print("🔍 Stage L: 跨书对比分析")
    print("=" * 50)

    stage = StageL()
    all_dimensions = [
        "感情线设计",
        "高潮铺垫方式",
        "冲突升级模式",
        "人物塑造",
        "世界观设计",
        "对话风格",
        "描写技法",
        "结构编排",
        "信息管理",
        "伏笔设计",
    ]

    # 动态过滤：只处理知识库中有足够数据的维度
    comparison_dimensions = []
    for dimension in all_dimensions:
        # 检查该维度是否有足够的对比数据
        data_count = stage.check_dimension_data(dimension)
        if data_count >= 2:  # 至少需要 2 本书才能对比
            comparison_dimensions.append(dimension)
            print(f"   ✅ 维度 '{dimension}': {data_count} 条数据")
        else:
            print(f"   ⏭️  跳过维度 '{dimension}': 数据不足 ({data_count} 条)")

    if not comparison_dimensions:
        print("⚠️ 没有足够数据进行跨书对比，跳过 Stage L")
        return 0, 0

    total_success = 0
    total_fail = 0

    for dimension in comparison_dimensions:
        try:
            print(f"\n   📊 对比维度: {dimension}")
            result = stage.run(comparison_dimension=dimension)
            stats = stage.insert(result)
            books_count = len(result.get("books_analyzed", []))
            patterns_count = len(result.get("analysis", {}).get("common_patterns", []))
            print(f"   ✅ 分析书籍: {books_count} | 共同模式: {patterns_count}")
            total_success += 1
        except Exception as exc:
            logger.error(f"❌ Stage L 对比分析失败 ({dimension}): {exc}")
            total_fail += 1

    print(
        f"\n📊 Stage L 完成: 成功 {total_success}/{len(comparison_dimensions)}, 失败 {total_fail}"
    )
    return total_success, total_fail


def run_stage_m():
    """Stage M: 常见错误模式提取"""
    from stages.stage_m import StageM

    print("\n" + "=" * 50)
    print("🔍 Stage M: 常见错误模式提取")
    print("=" * 50)

    stage = StageM()

    try:
        result = stage.run(min_frequency=2)
        stats = stage.insert(result)
        mistakes_count = len(result.get("mistakes", []))
        reviews_count = result.get("total_reviews_analyzed", 0)
        print(f"   ✅ 分析评审数: {reviews_count} | 错误模式数: {mistakes_count}")

        if mistakes_count == 0:
            print("   ⚠️ 未提取到错误模式，可能是评审历史数据不足")
            print(
                "   💡 建议：先通过 /api/kb/review 接口评审几章正文，积累评审数据后再运行"
            )

        return mistakes_count, 0
    except Exception as exc:
        logger.error(f"❌ Stage M 错误模式提取失败: {exc}")
        return 0, 1


def run_stage_n():
    """Stage N: 技法组合模板提取"""
    from stages.stage_n import StageN

    print("\n" + "=" * 50)
    print("🔍 Stage N: 技法组合模板提取")
    print("=" * 50)

    stage = StageN()

    try:
        result = stage.run()
        stats = stage.insert(result)
        combos_count = len(result.get("combinations", []))
        scenes_count = len(result.get("scene_types_analyzed", []))
        print(f"   ✅ 分析场景类型: {scenes_count} | 技法组合数: {combos_count}")
        return combos_count, 0
    except Exception as exc:
        logger.error(f"❌ Stage N 技法组合提取失败: {exc}")
        return 0, 1


def main():
    parser = argparse.ArgumentParser(description="高级功能 Stage 独立执行脚本")
    parser.add_argument(
        "--only",
        choices=["L", "M", "N"],
        help="只执行指定的 Stage（GR=题材裁决规则聚合）",
    )
    parser.add_argument(
        "--skip",
        choices=["L", "M", "N"],
        help="跳过指定的 Stage",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="增量模式：只对新增书籍执行高级分析，不重跑已有书籍",
    )
    args = parser.parse_args()

    # 健康检查
    client = get_ollama_client()
    if not client.check_health():
        print("❌ Ollama 服务检查失败，请确保 Ollama 已启动并安装了所需模型")
        return

    # 初始化数据库
    db = get_db_manager()
    db.init_tables()

    # 初始化 ChromaDB
    chroma = get_chroma_manager()
    chroma.init_collections()

    # 检查知识库中是否有足够的数据
    cursor = db.connect().cursor()
    cursor.execute("SELECT COUNT(DISTINCT book_name) FROM book_metadata")
    book_count = cursor.fetchone()[0]

    if book_count < 2:
        print(
            f"⚠️ 知识库中只有 {book_count} 本书，跨书对比分析（Stage L）和技法组合（Stage N）需要至少 2 本书的数据"
        )
        print("💡 建议：先用 novel_analyzer.py 处理多本书后再运行此脚本")
        if args.only in ["L", "N"]:
            return

    print(f"\n📚 知识库中共有 {book_count} 本书的数据")

    # 增量模式：检查新增书籍数量
    manifest = load_manifest()
    last_advanced_run = manifest.get("last_advanced_run_book_count", 0)
    new_books_since_last = book_count - last_advanced_run

    if args.incremental:
        if new_books_since_last <= 0:
            print(
                f"✅ 没有新增书籍（上次运行后知识库已有 {last_advanced_run} 本，当前 {book_count} 本），无需增量更新"
            )
            return
        print(f"📈 增量模式：自上次运行后新增 {new_books_since_last} 本书")
    elif new_books_since_last > 0:
        print(
            f"📈 提示：自上次高级分析后新增 {new_books_since_last} 本书，建议使用 --incremental 模式"
        )

    print("🚀 开始执行高级功能 Stage...\n")

    stages_to_run = ["L", "M", "N"]
    if args.only:
        stages_to_run = [args.only]
    if args.skip:
        stages_to_run = [s for s in stages_to_run if s != args.skip]

    results = {}

    if "L" in stages_to_run:
        results["L"] = run_stage_l()

    if "M" in stages_to_run:
        results["M"] = run_stage_m()

    if "N" in stages_to_run:
        results["N"] = run_stage_n()

    if "GR" in stages_to_run:
        print("   ⚠️ GR stage removed (genre_specific_techniques deleted)")

    # 汇总报告
    print("\n" + "=" * 50)
    print("📊 高级功能执行报告")
    print("=" * 50)
    for stage_name, (success, fail) in results.items():
        status = "✅" if fail == 0 else "⚠️"
        print(f"   {status} Stage {stage_name}: 成功 {success}, 失败 {fail}")

    print("\n🏆 高级功能执行完成！")

    # 更新 manifest 中的上次运行书籍数
    manifest["last_advanced_run_book_count"] = book_count
    save_manifest(manifest)


if __name__ == "__main__":
    main()
