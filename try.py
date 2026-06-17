import requests
import pandas as pd
import json
import re
from tqdm import tqdm
from openai import OpenAI
import PyPDF2
import time
import os
import multiprocessing
import logging

# 配置日志记录
logging.basicConfig(filename='data_extraction.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义单位映射，用于处理数量级差
UNIT_MAPPING = {
    "KPa": "kPa",
    "MPa": "MPa",
    "W m−1 K−1": "W m⁻¹ K⁻¹",
    # 可以根据需要添加更多单位映射
}

# 读取 PDF 文件内容
def read_pdf_file(file_path):
    print(f"开始读取 PDF 文件: {file_path}...")
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    print(f"PDF 文件 {file_path} 读取完成！")
    return text

# 调用 API
def call_api(pdf_text, instructions):
    API_KEY = 'sk-vPaZ8buhFUbUaenjs8FucCkVpI28SYvoeszmDVPW1xYT9r5N'
    BASE_URL = 'https://api.hunyuan.cloud.tencent.com/v1'
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = [
        {"role": "system", "content": "You are a highly skilled data extraction expert. Carefully read the provided PDF content and extract all relevant data according to the instructions. Pay special attention to the correspondence between different data items."},
        {"role": "user", "content": f"{instructions}\nPDF 内容: {pdf_text}"}
    ]
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="hunyuan-turbos-latest",
                messages=messages,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"调用 API 时出现错误: {e}，正在重试 ({attempt + 1}/{max_retries})...")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                logging.error(f"调用 API 时出现错误: {e}，已达到最大重试次数。")
                return None

# 提取纯数值
def extract_numeric_value(s):
    # 处理 "数值± 数值 W m−1 K−1" 格式，只保留前面的数值
    s = re.split(r'±', str(s))[0]
    match = re.search(r'[-+]?\d*\.\d+|[-+]?\d+', s)
    return float(match.group()) if match else 0

# 进程 1：提取文章题目
def extract_title(pdf_text):
    title_instructions = """请仔细阅读提供的 PDF 论文内容，提取论文的文章题目，并以 JSON 格式返回，如 {"文章题目": "具体题目"}。题目用英文回答，若未找到则返回 {"文章题目": "无"}。"""
    title_result = call_api(pdf_text, title_instructions)
    try:
        title_data = json.loads(title_result)
        title = title_data.get("文章题目", "无")
    except (json.JSONDecodeError, TypeError):
        logging.error("提取文章题目时 API 返回结果格式错误，无法解析为 JSON。")
        title = "无"
    return title

# 进程 2：提取导热系数、密度、玻璃化转变温度、气孔率
def extract_thermal_density(pdf_text):
    thermal_density_instructions = """请仔细阅读提供的 PDF 论文内容，提取论文中涉及的基体材料导热系数、增强体材料导热系数、基体材料密度、增强体材料密度、基体的玻璃化转变温度、增强体的玻璃化转变温度、气孔率。
    对于基体材料导热系数和增强体材料导热系数，只提取数值，若未提及则对应的值返回 0。
    对于基体材料密度和增强体材料密度，单位统一为克每立方厘米，只提取数值，若未提及则对应的值返回 0。
    对于基体的玻璃化转变温度和增强体的玻璃化转变温度，单位统一为摄氏度，只提取数值，若未提及则对应的值返回 -300。
    对于气孔率，只提取数值，若未提及则返回 200。
    以 JSON 格式返回，如 {"基体材料导热系数": "数值", "增强体材料导热系数": "数值", "基体材料密度": "数值", "增强体材料密度": "数值", "基体的玻璃化转变温度": "数值", "增强体的玻璃化转变温度": "数值", "气孔率": "数值"}。只返回对应数值信息，不包含单位和其他多余文字描述。"""
    thermal_density_result = call_api(pdf_text, thermal_density_instructions)
    try:
        thermal_density_data = json.loads(thermal_density_result)
        matrix_thermal = extract_numeric_value(thermal_density_data.get("基体材料导热系数", 0))
        filler_thermal = extract_numeric_value(thermal_density_data.get("增强体材料导热系数", 0))
        matrix_density = extract_numeric_value(thermal_density_data.get("基体材料密度", 0))
        filler_density = extract_numeric_value(thermal_density_data.get("增强体材料密度", 0))
        matrix_tg = extract_numeric_value(thermal_density_data.get("基体的玻璃化转变温度", -300))
        filler_tg = extract_numeric_value(thermal_density_data.get("增强体的玻璃化转变温度", -300))
        porosity = extract_numeric_value(thermal_density_data.get("气孔率", 200))
    except (json.JSONDecodeError, TypeError):
        logging.error("提取导热系数、密度、玻璃化转变温度、气孔率时 API 返回结果格式错误，无法解析为 JSON。")
        matrix_thermal = 0
        filler_thermal = 0
        matrix_density = 0
        filler_density = 0
        matrix_tg = -300
        filler_tg = -300
        porosity = 200
    return matrix_thermal, filler_thermal, matrix_density, filler_density, matrix_tg, filler_tg, porosity

# 进程 3：提取其他信息
def extract_other_info(pdf_text):
    other_instructions = """请仔细阅读提供的 PDF 论文内容，按照以下要求整理信息并以 JSON 格式返回：
    1. 提取论文中涉及的基体材料名称和增强体材料名称。对于可能存在多种基体材料或增强体材料的情况，尽可能全部列出。若未提及则填‘无’。
    2. 提取基体比热容和增强体比热容，若未提及则对应的值返回 0。
    3. 提取填料的形状，若未提及则填‘无’。若存在多种填料形状，用逗号分隔以字符串形式输出，例如 '片状,管状'。
    4. 对于填料含量，若文中使用的是体积分数，提取对应数值填入，质量分数栏统一反馈数值 500；反之，若文中使用的是质量分数，提取对应数值填入，体积分数栏统一反馈数值 500；若两个数据都没有，则同时反馈数值 500。
    5. 对于复合导热系数、界面热阻、热扩散率，需全面查找论文中的数据，包括不同实验条件下的数据。提取所有不同数据对应的数值。若未提及则对应的值返回 0。
    注意数据之间的对应关系，将相关联的数据组合在一起。最终返回的结果应该是一个列表，列表中的每个元素是一个包含上述所有信息的字典。例如：
    [
        {
            "基体材料": "具体名称",
            "增强体材料": "具体名称",
            "基体比热容": "数值",
            "增强体比热容": "数值",
            "填料形状": "具体形状或无",
            "填料含量（体积分数）": "数值",
            "填料含量（质量分数）": "数值",
            "复合导热系数": "数值",
            "界面热阻": "数值",
            "热扩散率": "数值"
        }
        // 可能有更多数据组对应的字典
    ]
    只返回对应数值信息，不包含单位和其他多余文字描述。"""
    other_result = call_api(pdf_text, other_instructions)
    try:
        other_data = json.loads(other_result)
        for item in other_data:
            item["基体比热容"] = extract_numeric_value(item.get("基体比热容", 0))
            item["增强体比热容"] = extract_numeric_value(item.get("增强体比热容", 0))
            item["填料含量（体积分数）"] = extract_numeric_value(item.get("填料含量（体积分数）", 500))
            item["填料含量（质量分数）"] = extract_numeric_value(item.get("填料含量（质量分数）", 500))
            item["复合导热系数"] = extract_numeric_value(item.get("复合导热系数", 0))
            item["界面热阻"] = extract_numeric_value(item.get("界面热阻", 0))
            item["热扩散率"] = extract_numeric_value(item.get("热扩散率", 0))
            # 处理填料形状字段
            if isinstance(item.get("填料形状"), list):
                item["填料形状"] = ', '.join(item["填料形状"])
    except (json.JSONDecodeError, TypeError):
        logging.error("提取其他信息时 API 返回结果格式错误，无法解析为 JSON。")
        other_data = []
    return other_data

# 进程 4：提取基体相关参数
def extract_matrix_params(pdf_text):
    matrix_params_instructions = """请仔细阅读提供的 PDF 论文内容，提取论文中涉及的基体熔点（℃）、基体体积模量（Pa）、基体剪切模量（Pa）、基体相对分子量、基体晶格常数a（Å）、基体晶格常数b（Å）、基体晶格常数c（Å）。
    对于基体熔点，若未提及则返回 -300。
    对于基体体积模量和剪切模量，若未提及则返回 0。
    对于基体相对分子量，若未提及则返回 0。
    对于基体晶格常数a、b、c，若未提及则返回 0。
    以 JSON 格式返回，如 {"基体熔点（℃）": "数值", "基体体积模量（Pa）": "数值", "基体剪切模量（Pa）": "数值", "基体相对分子量": "数值", "基体晶格常数a（Å）": "数值", "基体晶格常数b（Å）": "数值", "基体晶格常数c（Å）": "数值"}。只返回对应数值信息，不包含单位和其他多余文字描述。"""
    matrix_params_result = call_api(pdf_text, matrix_params_instructions)
    try:
        matrix_params_data = json.loads(matrix_params_result)
        matrix_melting_point = extract_numeric_value(matrix_params_data.get("基体熔点（℃）", -300))
        matrix_bulk_modulus = extract_numeric_value(matrix_params_data.get("基体体积模量（Pa）", 0))
        matrix_shear_modulus = extract_numeric_value(matrix_params_data.get("基体剪切模量（Pa）", 0))
        matrix_molecular_weight = extract_numeric_value(matrix_params_data.get("基体相对分子量", 0))
        matrix_lattice_a = extract_numeric_value(matrix_params_data.get("基体晶格常数a（Å）", 0))
        matrix_lattice_b = extract_numeric_value(matrix_params_data.get("基体晶格常数b（Å）", 0))
        matrix_lattice_c = extract_numeric_value(matrix_params_data.get("基体晶格常数c（Å）", 0))
    except (json.JSONDecodeError, TypeError):
        logging.error("提取基体相关参数时 API 返回结果格式错误，无法解析为 JSON。")
        matrix_melting_point = -300
        matrix_bulk_modulus = 0
        matrix_shear_modulus = 0
        matrix_molecular_weight = 0
        matrix_lattice_a = 0
        matrix_lattice_b = 0
        matrix_lattice_c = 0
    return matrix_melting_point, matrix_bulk_modulus, matrix_shear_modulus, matrix_molecular_weight, matrix_lattice_a, matrix_lattice_b, matrix_lattice_c

# 进程 5：提取增强体相关参数
def extract_filler_params(pdf_text):
    filler_params_instructions = """请仔细阅读提供的 PDF 论文内容，提取论文中涉及的增强体熔点（℃）、增强体体积模量（Pa）、增强体剪切模量（Pa）、增强体相对分子量、增强体晶格常数a（Å）、增强体晶格常数b（Å）、增强体晶格常数c（Å）。
    对于增强体熔点，若未提及则返回 -300。
    对于增强体体积模量和剪切模量，若未提及则返回 0。
    对于增强体相对分子量，若未提及则返回 0。
    对于增强体晶格常数a、b、c，若未提及则返回 0。
    以 JSON 格式返回，如 {"增强体熔点（℃）": "数值", "增强体体积模量（Pa）": "数值", "增强体剪切模量（Pa）": "数值", "增强体相对分子量": "数值", "增强体晶格常数a（Å）": "数值", "增强体晶格常数b（Å）": "数值", "增强体晶格常数c（Å）": "数值"}。只返回对应数值信息，不包含单位和其他多余文字描述。"""
    filler_params_result = call_api(pdf_text, filler_params_instructions)
    try:
        filler_params_data = json.loads(filler_params_result)
        filler_melting_point = extract_numeric_value(filler_params_data.get("增强体熔点（℃）", -300))
        filler_bulk_modulus = extract_numeric_value(filler_params_data.get("增强体体积模量（Pa）", 0))
        filler_shear_modulus = extract_numeric_value(filler_params_data.get("增强体剪切模量（Pa）", 0))
        filler_molecular_weight = extract_numeric_value(filler_params_data.get("增强体相对分子量", 0))
        filler_lattice_a = extract_numeric_value(filler_params_data.get("增强体晶格常数a（Å）", 0))
        filler_lattice_b = extract_numeric_value(filler_params_data.get("增强体晶格常数b（Å）", 0))
        filler_lattice_c = extract_numeric_value(filler_params_data.get("增强体晶格常数c（Å）", 0))
    except (json.JSONDecodeError, TypeError):
        logging.error("提取增强体相关参数时 API 返回结果格式错误，无法解析为 JSON。")
        filler_melting_point = -300
        filler_bulk_modulus = 0
        filler_shear_modulus = 0
        filler_molecular_weight = 0
        filler_lattice_a = 0
        filler_lattice_b = 0
        filler_lattice_c = 0
    return filler_melting_point, filler_bulk_modulus, filler_shear_modulus, filler_molecular_weight, filler_lattice_a, filler_lattice_b, filler_lattice_c

# 处理单个 PDF 文件
def process_single_pdf(pdf_file):
    pdf_text = read_pdf_file(pdf_file)

    # 使用多进程提取信息
    pool = multiprocessing.Pool(processes=5)
    title = pool.apply_async(extract_title, args=(pdf_text,))
    thermal_density = pool.apply_async(extract_thermal_density, args=(pdf_text,))
    other_info = pool.apply_async(extract_other_info, args=(pdf_text,))
    matrix_params = pool.apply_async(extract_matrix_params, args=(pdf_text,))
    filler_params = pool.apply_async(extract_filler_params, args=(pdf_text,))

    title = title.get()
    matrix_thermal, filler_thermal, matrix_density, filler_density, matrix_tg, filler_tg, porosity = thermal_density.get()
    other_data = other_info.get()
    matrix_melting_point, matrix_bulk_modulus, matrix_shear_modulus, matrix_molecular_weight, matrix_lattice_a, matrix_lattice_b, matrix_lattice_c = matrix_params.get()
    filler_melting_point, filler_bulk_modulus, filler_shear_modulus, filler_molecular_weight, filler_lattice_a, filler_lattice_b, filler_lattice_c = filler_params.get()

    pool.close()
    pool.join()

    final_data = []
    for item in other_data:
        new_item = {
            "文章题目": title,
            "基体材料导热系数": matrix_thermal,
            "增强体材料导热系数": filler_thermal,
            "基体材料": item.get("基体材料", "无"),
            "增强体材料": item.get("增强体材料", "无"),
            "基体比热容": item.get("基体比热容", 0),
            "增强体比热容": item.get("增强体比热容", 0),
            "填料形状": item.get("填料形状", "无"),
            "填料含量（体积分数）": item.get("填料含量（体积分数）", 500),
            "填料含量（质量分数）": item.get("填料含量（质量分数）", 500),
            "复合导热系数": item.get("复合导热系数", 0),
            "界面热阻": item.get("界面热阻", 0),
            "热扩散率": item.get("热扩散率", 0),
            "基体材料密度": matrix_density,
            "增强体材料密度": filler_density,
            "基体的玻璃化转变温度": matrix_tg,
            "增强体的玻璃化转变温度": filler_tg,
            "气孔率": porosity,
            "基体熔点（℃）": matrix_melting_point,
            "基体体积模量（Pa）": matrix_bulk_modulus,
            "基体剪切模量（Pa）": matrix_shear_modulus,
            "基体相对分子量": matrix_molecular_weight,
            "基体晶格常数a（Å）": matrix_lattice_a,
            "基体晶格常数b（Å）": matrix_lattice_b,
            "基体晶格常数c（Å）": matrix_lattice_c,
            "增强体熔点（℃）": filler_melting_point,
            "增强体体积模量（Pa）": filler_bulk_modulus,
            "增强体剪切模量（Pa）": filler_shear_modulus,
            "增强体相对分子量": filler_molecular_weight,
            "增强体晶格常数a（Å）": filler_lattice_a,
            "增强体晶格常数b（Å）": filler_lattice_b,
            "增强体晶格常数c（Å）": filler_lattice_c
        }
        final_data.append(new_item)

    return final_data

# 主函数
def main():
    pdf_folder_path = '千早爱音'  # 替换为你的 PDF 文件夹路径

    try:
        # 获取 PDF 文件夹中的所有 PDF 文件
        pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

        all_data = []
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            data = process_single_pdf(pdf_file)
            all_data.extend(data)

        # 创建 DataFrame 并将文章题目放在第一列
        df = pd.DataFrame(all_data)
        if "文章题目" in df.columns:
            cols = df.columns.tolist()
            cols = [cols.pop(cols.index("文章题目"))] + cols
            df = df[cols]

        # 导出为 Excel 文件
        df.to_excel('filled_量纲表（全）.xlsx', index=False)
        print("填充后的 Excel 文件已导出为 filled_量纲表9.0.xlsx。")

    except Exception as e:
        logging.error(f"程序运行时出现错误: {e}")

if __name__ == '__main__':
    main()