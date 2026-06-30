import os


def show_directory_tree(folder_path, prefix=""):
    """递归生成并打印目录树"""
    if not os.path.exists(folder_path):
        print(f"❌ 错误：找不到路径 '{folder_path}'")
        return

    try:
        # 获取目录下所有内容，按名称排序（文件夹和文件混排）
        items = sorted(os.listdir(folder_path))
    except PermissionError:
        print(f"{prefix}└── [权限不足，无法访问]")
        return

    # 过滤掉常见的系统/隐藏文件，保持输出干净
    ignore_list = {".DS_Store", "Thumbs.db", "desktop.ini", ".gitkeep"}
    items = [i for i in items if i not in ignore_list and not i.startswith(".")]

    for index, item in enumerate(items):
        # 判断是否是最后一项，以决定使用 └── 还是 ├──
        is_last = index == len(items) - 1
        connector = "└── " if is_last else "├── "
        item_path = os.path.join(folder_path, item)

        # 打印当前项
        print(f"{prefix}{connector}{item}")

        # 如果是目录，则递归向下遍历
        if os.path.isdir(item_path):
            # 下一层的缩进前缀：如果是最后一项用空格，否则用竖线 │
            next_prefix = prefix + ("    " if is_last else "│   ")
            show_directory_tree(item_path, next_prefix)


if __name__ == "__main__":
    # 👇👇👇 请把这里改成你的 novels 文件夹的实际绝对路径 👇👇👇
    # Windows 示例: r"D:\MyBooks\novels"  (注意前面的 r)
    # Mac/Linux 示例: "/Users/yourname/Documents/novels"
    MY_NOVELS_PATH = r"./novels"

    print(f"📂 正在生成 '{MY_NOVELS_PATH}' 的目录树...\n")
    show_directory_tree(MY_NOVELS_PATH)
    print("\n✅ 输出完毕！请将以上结果复制发给我。")
