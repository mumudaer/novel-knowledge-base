import os
import re

# ================= 配置区 =================
TARGET_FOLDER = r"./novels"
DRY_RUN = False  # 预览模式
# ==========================================

# 🚫 文件夹黑名单：这些文件夹名绝对不能被当成作者！
IGNORE_FOLDER_NAMES = {
    "热门小说",
    "男频热文",
    "晋江长佩top合集带番外（精校版全本）",
    "晋江长佩top合集带番外",
    "晋江长佩top带番外",
    "晋江高收藏必存50本",
    "作者合集小说",
    "novels",
    "男频文62本",
    "《盗墓笔记》全系列",
    "盗墓笔记全系列",
    "凡人修仙传",
    "十日终焉",
    "三体·书籍",
    "三体书籍",
    "三体",
    "全职高手",
    "诛仙",
    "轮回乐园",
    "辰东7本合集",
    "辰东7本",
    "辰东",
    "龙族",
    "诡秘地海",
    "诡舍",
    "大奉打更人",
    "天才俱乐部",
    "学霸的黑科技系统",
    "宿命之环",
    "异兽迷城",
    "我师兄实在太稳健了",
    "我的属性修行人生",
    "我的模拟长生路",
    "我真没想重生啊",
    "择日走红",
    "深海余烬",
    "牧神记",
    "穷鬼的上下两千年",
    "逆天邪神",
    "遮天",
    "雪中悍刀行",
    "高武纪元",
    "天倾之后",
    "大乘期才有逆袭系统",
    "庆余年",
    "全球高武",
    "剑来",
    "剑出大唐",
    "一世之尊",
    "万古神帝",
    "万族之劫",
    "从红月开始",
    "光阴之外",
    "全职法师",
    "大王饶命",
    "将夜",
    "小阁老",
    "惊悚乐园",
    "斗破苍穹",
    "明日之劫",
    "明朝那些事儿",
    "晚明",
    "死亡万花筒",
    "死人经",
    "求魔",
    "灵境行者",
    "烂柯棋缘",
    "黎明之剑",
    "黜龙",
    "FOG［电竞］",
    "FOG[电竞]",
    "《嘘别说话》请及时转存，失效不补！",
    "《子夜十》+《子夜鸮》",
    "《宫廷生存纪事》请及时转存，购买三天后失效不补！！",
    "《江医生他怀了死对头的崽》by葫芦酱",
    "《荒唐》作者：臣年",
    "《陈年烈苟》",
    "《默读》作者：priest",
    "值此新婚",
    "剑来（1-49册）(烽火戏诸侯)",
    "囚于永夜（及时保存，过后失效不补）",
    "天官赐福",
    "将进酒",
    "提灯映桃花",
    "沧澜道",
    "漂亮的玫瑰",
    "老婆孩子热炕头",
    "臣服四部曲+笼中鸟",
    "神秘复苏",
    "秦吏",
    "第一序列",
    "覆汉",
    "超神机械师",
    "道诡异仙",
    "太平令",
    "热门",
    "盗墓笔记",
    "男频热文",
    "晋江长佩top",
    "我不是戏神",
}

# 🧹 杂质词库：
JUNK_WORDS = [
    "番外完结",
    "精校版",
    "未删减",
    "未删",
    "全本",
    "完整版",
    "完结",
    "番外",
    "补番",
    "补车",
    "补车番",
    "补C",
    "补F",
    "连载",
    "坑",
    "作者",
    "补r",
    "补R",
    "实体",
    "福利",
    "校对",
    "精校",
    "全订",
    "-txt",
    "txt",
    "TXT",
    "NP",
    "np",
    "全文",
]

# 🚨 独立的单字/短词杂质（使用正则边界匹配，防止误伤“七月新番”）
INDEPENDENT_JUNK = [r"番全", r"番", r"补"]

# 🚫 “假作者”拦截正则库
FAKE_AUTHOR_PATTERNS = [
    r"热门",
    r"^未知$",
    r"^全$",
    r"^\+全$",
    r"^\+$",
    r"感言",
    r"贺岁篇",
    r"补最新福",
    r"东南亚篇",
    r"南京篇",
    r"共七册",
    r"个故事",
    r"篇$",
    r"^车$",
    r"微博小段",
    r"补全",
    r"未删减完整版",
    r"精校版全本",
    r"校对版全本",
    r"共\d+章",
    r"补\d+",
    r"\+番外",
    r"^\[.*\]$",
    r"^［.*］$",
    r"^不含.*?$",
    r"^番番$",
    r"^完&$",
    r"^\d+$",
]

# ✅ 单字作者白名单
SINGLE_CHAR_AUTHOR_WHITELIST = {"乱"}

# ✅ 英文/拼音作者白名单（保护 priest, 阎ZK 等）
ENGLISH_AUTHOR_WHITELIST = {"priest", "阎ZK", "larivegauche"}


def extract_author_from_folder(folder_name):
    name = folder_name
    name = re.sub(r"[\(（].*?[\)）]", "", name)
    name = re.sub(r"\d+本.*$", "", name)
    name = (
        name.replace("合集", "")
        .replace("小说", "")
        .replace("书籍", "")
        .replace("·", "")
        .replace(".", "")
        .strip()
    )
    name = re.sub(r"［电竞］|\[电竞\]", "", name)
    if "／" in name or "/" in name:
        name = re.split(r"[／/]", name)[0]
    name = re.sub(r"^[A-Za-z\s]+", "", name).strip()
    if name in ["三体", "盗墓笔记"]:
        return ""
    return name if name else ""


def safe_clean_brackets(text):
    """安全剔除书名/作者名里的各种括号及内容，并清理残留的孤儿括号"""
    text = re.sub(r"[\(（\[【『].*?[\)）\]】』]", "", text)
    text = re.sub(r"[\(（\[【『］\]]+", "", text)
    text = re.sub(r"[\)）\]】』]+$", "", text)
    return text


def is_fake_author(author):
    """校验是否为假作者"""
    if not author:
        return True
    for pattern in FAKE_AUTHOR_PATTERNS:
        if re.search(pattern, author):
            return True
    if author.isdigit() or "电竞" in author:
        return True

    if re.search(r"[a-zA-Z]", author):
        if author.lower() in {x.lower() for x in ENGLISH_AUTHOR_WHITELIST}:
            return False
        if re.search(r"[\u4e00-\u9fa5]", author):
            return False
        return True

    return False


def process_filename(filename, folder_name, parent_folder_name):
    if not filename.lower().endswith(".txt"):
        return None

    name = filename[:-4]

    # 1. 提取类型后缀标记
    extra_type = ""
    type_keywords = [
        "补番",
        "番全",
        "补车",
        "补车番",
        "补C",
        "补F",
        "fw番外",
        "fw完结",
        "fw连载",
        "番外",
        "含番外",
        "坑",
        "未删减",
        "未删",
        "精校",
    ]
    for kw in type_keywords:
        if kw in name:
            if kw == "坑":
                extra_type = ""
            elif any(x in kw for x in ["番外", "番全", "补番", "补车", "补C", "补F"]):
                extra_type = "番外"
            break

    # 2. 优先剥离前缀/后缀干扰词
    clean_name = name
    clean_name = re.sub(r"^[\[【（\(]+.*?[\]】）\)]+\s*", "", clean_name)
    clean_name = re.sub(
        r"^(完整版|实体番全|番外全|福利番全|附带番外|精校版|含实体番|新版|旧版|回归重修版|【.*?】)\s*",
        "",
        clean_name,
    )
    clean_name = re.sub(r"^\d+\s*", "", clean_name)

    # 强制抹除书名内部的“毒瘤括号”和“电竞标签”
    clean_name = re.sub(r"（加作者感言）", "", clean_name)
    clean_name = re.sub(r"\[电竞\]", "", clean_name)
    clean_name = re.sub(r"［电竞］", "", clean_name)
    clean_name = re.sub(r"（精校版全本.*?）", "", clean_name)
    clean_name = re.sub(r"（校对版全本.*?）", "", clean_name)
    clean_name = re.sub(r"（未删.*?）", "", clean_name)
    clean_name = re.sub(r"（完结.*?）", "", clean_name)

    clean_name = clean_name.strip()

    # 3. 核心：精准切割书名和作者
    book_name = ""
    author = "未知"

    start_idx = clean_name.find("《")
    end_idx = clean_name.rfind("》")

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        book_name = clean_name[start_idx + 1 : end_idx]
        remaining_text = clean_name[end_idx + 1 :].strip()

        bracket_author_match = re.search(r"[\(（]([^\)）]+)[\)）]", remaining_text)
        underline_match = re.search(
            r"[_\-]{1,2}([a-zA-Z0-9\u4e00-\u9fa5]+)$", remaining_text
        )
        author_match = re.search(
            r"(?:作者[：:\s_]*|by\s*)(.*)", remaining_text, re.IGNORECASE
        )

        if author_match:
            author = author_match.group(1).strip()
        elif bracket_author_match:
            author = bracket_author_match.group(1).strip()
        elif underline_match:
            author = underline_match.group(1).strip()
        else:
            author = re.split(r"[\s\[（\(【\-_\.]", remaining_text)[0].strip()
    else:
        # 🚨 核心修复：恢复对 - 的切割逻辑，但增加更严格的校验！
        dash_match = re.match(r"^(.+?)[-_]([^\d\-_]{2,10})$", clean_name)
        if dash_match:
            book_name = dash_match.group(1).strip()
            author = dash_match.group(2).strip()
        else:
            parts = re.split(r"\s+|_|作者[：:\s_]*|by\s*", clean_name, maxsplit=1)
            book_name = parts[0].strip()
            remaining_text = clean_name[len(book_name) :]

            bracket_author_match = re.search(r"[\(（]([^\)）]+)[\)）]", remaining_text)
            author_match = re.search(
                r"(?:作者[：:\s_]*|by\s*)(.*)", remaining_text, re.IGNORECASE
            )

            if author_match:
                author = author_match.group(1).strip()
            elif bracket_author_match:
                author = bracket_author_match.group(1).strip()
            else:
                end_author_match = re.search(
                    r"[\u4e00-\u9fa5A-Za-z]{2,}$", remaining_text
                )
                if end_author_match:
                    author = end_author_match.group(0).strip()
                else:
                    author = remaining_text.strip()

    # 4. 深度清理书名和作者的杂质
    book_name = safe_clean_brackets(book_name)
    book_name = re.sub(
        r"(番外完结|番外全|完结\+番外|正文\+微博番外全|精校版|未删减|全本|\+番外全|\+番外)",
        "",
        book_name,
    )
    book_name = (
        book_name.replace("》", "")
        .replace("《", "")
        .replace("作者", "")
        .replace("txt", "")
        .replace("TXT", "")
        .strip(" -_·.！!？?")
    )
    book_name = re.sub(r"\s+", "", book_name)

    if book_name == "子-鸮":
        book_name = "子夜鸮"
    book_name = re.sub(r"(?<!\d)-(?!\d)", "", book_name)

    author = safe_clean_brackets(author)
    author = (
        author.replace("-", " ")
        .replace("_", " ")
        .replace("作者", "")
        .replace("txt", "")
        .replace("TXT", "")
        .strip(" -_·.！!？?()（）+")
    )

    for junk in JUNK_WORDS:
        author = author.replace(junk, "").strip()

    for junk in INDEPENDENT_JUNK:
        author = re.sub(rf"(?:^|[\s\-_]){junk}(?:$|[\s\-_])", "", author).strip()

    author = re.sub(r"[番补]\d+$", "", author).strip()
    author = re.sub(r"\d{4}$", "", author).strip()

    # 🚨 核心修复：强制剔除作者名中的所有空格！确保“水千丞 番全”变成“水千丞”！
    author = re.sub(r"\s+", "", author).strip()

    # 5. 🚫 “假作者”拦截校验
    is_single_char = len(author) == 1
    if (
        is_fake_author(author)
        or author.startswith("+")
        or ("-" in author and len(author) > 5)
        or (is_single_char and author not in SINGLE_CHAR_AUTHOR_WHITELIST)
    ):
        author = ""

    # 6. 终极继承逻辑
    if not author or author == "未知":
        clean_folder = extract_author_from_folder(folder_name)
        clean_parent = extract_author_from_folder(parent_folder_name)

        if (
            clean_folder
            and clean_folder not in IGNORE_FOLDER_NAMES
            and clean_folder != book_name
            and not is_fake_author(clean_folder)
        ):
            author = clean_folder
        elif (
            clean_parent
            and clean_parent not in IGNORE_FOLDER_NAMES
            and clean_parent != book_name
            and not is_fake_author(clean_parent)
        ):
            author = clean_parent
        else:
            author = "未知"

    # 7. 组装标准格式
    illegal_chars = r'[\\/:*?"<>|&]'
    book_name = re.sub(illegal_chars, "", book_name)
    author = re.sub(illegal_chars, "", author)

    author = author.strip("·. ")

    if not author or author.strip() == "":
        author = "未知"

    type_str = f"[{extra_type}]" if extra_type else ""
    new_name = f"《{book_name}》作者：{author}{type_str}.txt"

    if not book_name:
        return None

    return new_name


def run():
    if not os.path.exists(TARGET_FOLDER):
        print(f"❌ 错误：找不到文件夹 {TARGET_FOLDER}")
        return

    print(f"🔍 正在扫描: {TARGET_FOLDER}\n")
    print("-" * 90)

    rename_count = 0
    error_count = 0

    for root, dirs, files in os.walk(TARGET_FOLDER):
        current_folder_name = os.path.basename(root)
        parent_folder_name = os.path.basename(os.path.dirname(root))

        for filename in files:
            if not filename.lower().endswith(".txt"):
                continue

            new_filename = process_filename(
                filename, current_folder_name, parent_folder_name
            )

            if not new_filename or filename == new_filename:
                continue

            old_path = os.path.join(root, filename)
            new_path = os.path.join(root, new_filename)

            if os.path.exists(new_path) and old_path != new_path:
                base, ext = os.path.splitext(new_filename)
                i = 1
                while os.path.exists(new_path):
                    new_filename = f"{base}_{i}{ext}"
                    new_path = os.path.join(root, new_filename)
                    i += 1

            print(f"📂 目录: {current_folder_name}")
            print(f"📄 原名: {filename}")
            print(f"✨ 新名: {new_filename}")

            if not DRY_RUN:
                try:
                    os.rename(old_path, new_path)
                    print("✅ 状态: 重命名成功")
                except Exception as e:
                    print(f"❌ 状态: 失败 ({e})")
                    error_count += 1
            else:
                print("👀 状态: [预览模式]")

            print("-" * 90)
            rename_count += 1

    print(
        f"\n🎉 扫描完成！共发现 {rename_count} 个需要规范化的文件。失败 {error_count} 个。"
    )
    if DRY_RUN:
        print(
            "💡 提示：当前为【预览模式】。确认无误后，请将代码中的 DRY_RUN = True 改为 False，再次运行即可！"
        )


if __name__ == "__main__":
    run()
