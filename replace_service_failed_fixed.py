import glob
import os
import json
import subprocess
from typing import List, Tuple
from tqdm import tqdm
import re
from fix_syntax import check_syntax, fix_with_context, fix_test_case
# 在全局添加统计变量
initial_success = 0
final_success = 0
total_split_methods = 0
def load_split_result(file_path: str) -> List[Tuple[str, str, dict]]:
    """从JSON文件加载拆分结果"""
    json_path = os.path.join(
        os.path.dirname(file_path),
        f"{os.path.splitext(os.path.basename(file_path))[0]}_split_raw.json"
    )
    
    if not os.path.exists(json_path):
        print(f"No split result found for {file_path}")
        return []
        
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        results = []
        for method in data["split_methods"]:
            results.append((
                method["original_name"],
                method["original_body"],
                method["split_methods"]
            ))
        return results
    except Exception as e:
        print(f"Error loading split result for {file_path}: {str(e)}")
        return []


def verify_test(directory: str) -> Tuple[bool, str]:
    """运行mvn test并返回验证结果及错误信息"""
    try:
        result = subprocess.run(
            ['mvn', 'test'],
            cwd=directory,
            capture_output=True,
            text=True,
            # encoding='latin1',
            # errors='replace',
            shell=True,
        )
        # 手动解码标准输出和错误输出
        stdout = result.stdout
        # stdout = result.stdout.decode('utf-8', errors='replace')

        # 合并标准输出和错误输出
        # print(f"标准输出: {stdout}\n")
        # print('=========================================================\n')

        # 判断构建是否成功
        if result.returncode == 0 and "BUILD SUCCESS" in stdout:
            return True, ""

        # 错误信息解析逻辑
        error_info = extract_test_errors(stdout)
        # print(f"error_info: {error_info}")
        return False, error_info

    except Exception as e:
        return False, f"执行mvn test命令失败: {str(e)}"


def extract_test_errors(output: str) -> str:
    """从Maven输出中提取关键错误信息"""
    error_patterns = [
        (r"Failed tests:.*?\n((?:.*?$.*?$).*?\n)*", "FAILED TESTS"),
        (r"Tests in error:.*?\n((?:.*?->.*?\n)*)", "TESTS IN ERROR"),
        (r"Test.*?FAILED.*?\n", "GENERAL FAILURES"),
        (r"Caused by:.*?\n", "ROOT CAUSE"),
        (r"ERROR $$.*?$$ (.*?)\n", "ERROR MESSAGES")
    ]

    extracted = []
    seen_errors = set()

    # 使用多级模式匹配提取关键信息
    for pattern, category in error_patterns:
        matches = re.finditer(pattern, output, re.DOTALL)
        for match in matches:
            error_lines = match.group(0).strip().split('\n')
            for line in error_lines:
                clean_line = re.sub(r'\x1b\[.*?m', '', line).strip()  # 去除ANSI颜色代码
                if clean_line and clean_line not in seen_errors:
                    extracted.append(f"[{category}] {clean_line}")
                    seen_errors.add(clean_line)

    # 提取最后10行作为补充信息
    if not extracted:
        last_lines = '\n'.join(output.splitlines()[-20:])
        extracted.append(f"[LAST LINES]\n{last_lines}")

    # 合并错误信息并截断长度
    error_info = '\n'.join(extracted[:10])  # 最多保留前5个错误
    return error_info[:2000]  # 限制最大长度
def replace_file(file_path: str, split_results: List[Tuple[str, str, dict]], project_dir: str) -> None:
    """应用拆分结果到文件"""
    global initial_success, final_success

    if not split_results:
        return

    # 读取原文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    original_content = content

    # 首先尝试替换所有方法
    for orig_name, orig_body, split_methods in split_results:
        # 合并所有拆分后的方法
        new_body = "\n\n".join(split_methods.values())
        content = content.replace(orig_body, new_body)

    # 写入更改并验证编译
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

    is_pass, err_msg = verify_test(project_dir)
    if is_pass:
        print(f"All changes in {file_path} replace successfully")
        # 记录初步成功和最终成功
        initial_success += len(split_results)
        final_success += len(split_results)
        return

    # 如果整体替换失败，尝试逐个方法替换
    print(f"Execution failed with all changes, trying individual methods...")
    # 回滚代码
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(original_content)

    content = original_content

    # 逐个替换
    for orig_name, orig_body, split_methods in split_results:
        # 检查原方法体是否存在
        if orig_body not in original_content:
            continue

        current_body = "\n\n".join(split_methods.values())
        original_test_case = orig_body  # 原始完整测试用例

        # 阶段1: 初始替换验证（修复回滚方式）
        original_content_before_replace = content  # 保存当前状态
        new_content = original_content_before_replace.replace(orig_body, current_body)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        is_pass, err_msg = verify_test(project_dir)

        if is_pass:
            initial_success += 1
            final_success += 1
            print(f"初步方案替换成功: {orig_name}")
            content = new_content
            continue
        else:
            content = original_content_before_replace
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)


        # 语法修复阶段（最多2次尝试）
        syntax_valid = False
        for syntax_attempt in range(2):
            # 实际语法验证（不依赖模型返回的success）
            temp_content = content.replace(orig_body, current_body)
            is_valid, error_msg = check_syntax(temp_content)
            
            if is_valid:
                syntax_valid = True
                break
                
            # 执行修复并重新验证
            fixed_code, _ = fix_with_context(current_body, error_msg, len(split_methods))
            current_body = fixed_code  # 无论success如何都更新代码

        if not syntax_valid:
            print(f"语法验证失败: {orig_name}")
            continue

        # 测试修复阶段（最多3次尝试）
        test_passed = False
        for test_attempt in range(3):
            # 写入文件进行实际测试验证
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content.replace(orig_body, current_body))
                
            is_pass, err_msg = verify_test(project_dir)
            
            if is_pass:
                test_passed = True
                break
                
            # 执行测试修复并重新验证
            fixed_code, _ = fix_test_case(
                error_code=current_body,
                error_msg=err_msg,
                original_test_case=original_test_case,
                expected_method_count=len(split_methods)
            )
            # print(f"fixed_code: {fixed_code}")
            current_body = fixed_code  # 无论success如何都更新代码
            temp_content = content.replace(orig_body, current_body)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(temp_content)
                
            is_pass, _ = verify_test(project_dir)
            if is_pass:
                test_passed = True
                break

        # 最终结果处理
        if test_passed:
            content = temp_content
            print(f"方法验证通过: {orig_name}")
            final_success += 1
        else:
            print(f"测试验证失败: {orig_name}")
            content = content.replace(current_body, orig_body)  # 回滚代码

    # 写入最终内容
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def process_single_file(file_path: str, project_dir: str) -> None:
    """处理单个文件"""
    global total_split_methods
    split_results = load_split_result(file_path)
    if split_results:
        total_split_methods += len(split_results)
        replace_file(file_path, split_results, project_dir)
    else:
        raise Exception("No valid split results found")



def process_directory(directory: str) -> None:
    """处理目录下的所有测试文件"""
    global initial_success, final_success, total_split_methods
    test_files = glob.glob(os.path.join(directory, '**/src/test/**/*.java'), recursive=True)
    print(directory)
    total_files = len(test_files)
    print(f"Found {total_files} test files")
    
    # 统计有多少文件有对应的JSON
    files_with_json = [f for f in test_files if os.path.exists(
        os.path.join(
            os.path.dirname(f),
            f"{os.path.splitext(os.path.basename(f))[0]}_split_raw.json"
        )
    )]
    print(f"Found {len(files_with_json)} files with split results")
    
    if not files_with_json:
        print("No files to process. Exiting...")
        return
    
    success_count = 0
    fail_count = 0
    
    # 使用tqdm创建进度条
    for file_path in tqdm(files_with_json, desc="Processing files"):
        try:
            process_single_file(file_path, directory)
            success_count += 1
        except Exception as e:
            print(f"\nError processing file {file_path}: {str(e)}")
            fail_count += 1


    # 在打印最终统计时添加成功率信息
    print("\nProcessing completed!")
    print(f"Total files processed: {len(files_with_json)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed to process: {fail_count}")
    print(f"Total split methods: {total_split_methods}")

    if total_split_methods > 0:
        initial_rate = (initial_success / total_split_methods) * 100
        final_rate = (final_success / total_split_methods) * 100
        print(f"Initial success rate: {initial_rate:.2f}% ({initial_success}/{total_split_methods})")
        print(f"Final success rate after fixes: {final_rate:.2f}% ({final_success}/{total_split_methods})")
    else:
        print("No split methods to calculate success rates.")

if __name__ == '__main__':
    # import argparse
    #
    # parser = argparse.ArgumentParser(description='Process test files in a directory')
    # parser.add_argument('directory', help='Directory to scan for test files')
    #
    # args = parser.parse_args()
    process_directory(r'D:\learn\junit-test\EightQueens\EightQueens_tmp')