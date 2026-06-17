import re
import os

def clean_text(text):
    """清理文本中的非法字符，避免干扰正则表达式匹配"""
    text = re.sub(r'[^\w\s\-\.]', '', text)  # 保留字母、数字、空格、连字符和点号
    return text.strip()

def read_titles(file_path):
    """读取论文题目文件，返回清理后的标题列表"""
    with open(file_path, "r", encoding="utf-8") as file:
        titles = [clean_text(line.strip()).lower() for line in file.readlines()]  # 清理并转换为小写
    return titles

def extract_paragraphs(file_path, titles):
    """从文献文件中提取包含指定标题的段落，并显示进度"""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    paragraphs = re.split(r"\n\n", content)  # 段落由两个换行符分隔
    total_paragraphs = len(paragraphs)
    matched_paragraphs = []
    progress = 0

    for paragraph in paragraphs:
        progress += 1
        print(f"Processing paragraph {progress}/{total_paragraphs}", end="\r")  # 显示进度
        cleaned_paragraph = clean_text(paragraph)
        for title in titles:
            if re.search(re.escape(title), cleaned_paragraph, re.IGNORECASE):
                matched_paragraphs.append(paragraph)  # 保存原始段落
                break  # 匹配到一个标题后不再检查其他标题

    print("\nProcessing complete.")
    return matched_paragraphs

def save_results(matched_paragraphs, output_file):
    """将匹配到的段落保存到同一个 .txt 文件中"""
    with open(output_file, "w", encoding="utf-8") as file:
        for paragraph in matched_paragraphs:
            file.write(paragraph + "\n\n")  # 每个段落之间用两个换行符分隔
    print(f"All matched paragraphs have been saved to {output_file}")

def main():
    titles_file = "论文题目.txt"
    literature_file = "6001-15111.txt"
    output_file = "匹配结果.txt"

    # 读取论文标题
    titles = read_titles(titles_file)

    # 提取包含指定标题的段落
    matched_paragraphs = extract_paragraphs(literature_file, titles)

    # 保存结果到同一个文件
    save_results(matched_paragraphs, output_file)

if __name__ == "__main__":
    main()


