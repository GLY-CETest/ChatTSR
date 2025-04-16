import glob
import re
import sys
import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# ... existing imports ...
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
import datetime
# 添加到文件顶部的导入
from threading import Lock
import time
# 新增的导入
import javalang
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import OutputParserException
# 创建全局锁
compilation_lock = Lock()


# 设置环境变量
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["DEEPSEEK_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxx"
from langchain_deepseek import ChatDeepSeek

model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # 其他参数...
)
# 初始化模型
# model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)



from pydantic import BaseModel, Field
from typing import List

class SplitMethod(BaseModel):
    method_name: str = Field(..., description="The name of the split method")
    method_body: str = Field(..., description="The body of the split method")

class SplitMethodsResponse(BaseModel):
    split_methods: List[SplitMethod] = Field(..., description="List of split methods")

# 创建 PydanticOutputParser 实例
output_parser = JsonOutputParser(pydantic_object=SplitMethodsResponse)


def get_test_methods(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    method_declarations = re.findall(r'(@Test\s*(?:\(.*?\))?\s*public void (\w+)\(\))', content)
    # method_declarations = re.findall(r'(public void (\w+)\(\))', content)

    test_methods = []
    for method_declaration in method_declarations:
        method_body = method_declaration[0]
        origin_index = content.index(method_body)
        start_index = origin_index + len(method_body)
        # 继续向前查找，直到找到第一个 {
        while content[start_index] != '{':
            start_index += 1
        brace_count = 1
        in_string = False
        in_char = False
        in_single_line_comment = False
        in_multi_line_comment = False

        for i in range(start_index + 1, len(content)):
            c = content[i]

            # Handle string literals
            if not in_char and not in_single_line_comment and not in_multi_line_comment:
                if c == '"' and (i == 0 or content[i - 1] != '\\'):
                    in_string = not in_string

            # Handle character literals
            if not in_string and not in_single_line_comment and not in_multi_line_comment:
                if c == '\'' and (i == 0 or content[i - 1] != '\\'):
                    in_char = not in_char

            # Handle single-line comments
            if not in_string and not in_char and not in_multi_line_comment and c == '/' and i + 1 < len(content) and content[i + 1] == '/':
                in_single_line_comment = True

            # End of single-line comments
            if in_single_line_comment and c == '\n':
                in_single_line_comment = False

            # Handle multi-line comments
            if not in_string and not in_char and not in_single_line_comment and c == '/' and i + 1 < len(content) and content[i + 1] == '*':
                in_multi_line_comment = True

            # End of multi-line comments
            if in_multi_line_comment and c == '*' and i + 1 < len(content) and content[i + 1] == '/':
                in_multi_line_comment = False
                i += 1  # Skip the '/' character

            # Skip processing inside strings, characters, and comments
            if in_string or in_char or in_single_line_comment or in_multi_line_comment:
                continue

            # Count braces
            if c == '{':
                brace_count += 1
            elif c == '}':
                brace_count -= 1

            if brace_count == 0:
                method_end_index = i + 1  # 方法的结束位置
                # Found the complete method body
                method_body = content[origin_index:i+1]
                # print("Found method:", method_body)
                test_methods.append((method_body, method_declaration[1], origin_index, method_end_index))
                break


    filtered_test_methods = []
    for method_body, method_name, start_index, end_index in test_methods:
        if re.search(r'\b(?:assert\w*|assertTrue|assertFalse)\s*\(', method_body):
            # 查找所有断言
            assertions = re.findall(r'\b(?:assert\w*|assertTrue|assertFalse)\s*\(', method_body)
            if len(assertions) >= 2:
                filtered_test_methods.append((method_name, method_body, start_index, end_index))
    
    return filtered_test_methods


def save_split_result(file_path: str, split_results: List[Tuple[str, str, dict]]) -> None:
    """保存拆分结果到JSON文件"""
    results = {
        "original_file": file_path,
        "timestamp": str(datetime.datetime.now()),
        "split_methods": []
    }
    
    # 记录每个方法的拆分结果
    for orig_name, orig_body, split_methods in split_results:
        method_info = {
            "original_name": orig_name,
            "original_body": orig_body,
            "split_methods": split_methods
        }
        results["split_methods"].append(method_info)
    
    # 生成输出文件路径（与源文件同目录）
    output_path = os.path.join(
        os.path.dirname(file_path),
        f"{os.path.splitext(os.path.basename(file_path))[0]}_split_raw.json"
    )
    
    # 使用文件锁保存结果
    with Lock():
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Raw split results saved to: {output_path}")

def split_assertions(file_path, test_methods):
    new_test_methods = []
    split_result = []

    for method_name, method_body, method_line, method_dependency in test_methods:
        # 使用OpenAI对方法体进行拆分
        # 创建提示模板
        # 提取断言的数量
        assertions = re.findall(r'\b(?:assert\w*|assertTrue|assertFalse)\s*\(', method_body)
        assertions_count = len(assertions)
        # ... existing code ...
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a split-code AI assistant tasked with splitting a junit code method into several independently runnable junit test unit methods. Request:
            1.You must keep the business logic of the testing method unchanged
            2. There can only be one assertion per test unit.
            3. Do not add new assertions; only use the existing ones.
            4. Name each split method using the original method name plus "_i" suffix (where i starts from 1).
            5. Your input is a complete method with the number of assertions specified.
            6. Output the exact number of methods as there are assertions in the input method.
            7. Return your response in a JSON format that matches the following Pydantic model:
            {format_instructions}

            Here's an example:
            **Input method:**
            @Test
            public void testCalculator() {{
                Calculator calc = new Calculator();
                int result1 = calc.add(2, 3);
                int result2 = calc.multiply(4, 5);
                assertEquals(5, result1);
                assertEquals(20, result2);
            }}

            Expected JSON output:
            ```json
            {{
                "split_methods": [
                    {{
                        "method_name": "testCalculator_1",
                        "method_body": "@Test\\npublic void testCalculator_1() {{\\n    Calculator calc = new Calculator();\\n    int result1 = calc.add(2, 3);\\n    assertEquals(5, result1);\\n}}"
                    }},
                    {{
                        "method_name": "testCalculator_2",
                        "method_body": "@Test\\npublic void testCalculator_2() {{\\n    Calculator calc = new Calculator();\\n    int result2 = calc.multiply(4, 5);\\n    assertEquals(20, result2);\\n}}"
                    }}
                ]
            }}
            """
                ),
                ("user", """Input:\\n{code}
                        \\n
                        You need to split the method into {assertions_count} test units.
                 Return your response in a JSON format that matches the Pydantic model above. Do not add any comments or other text to the response.
                            """),
            ]
        ).partial(format_instructions=output_parser.get_format_instructions())

        chain = prompt | model | output_parser
        try:
            response = chain.invoke({"assertions_count":assertions_count, "code": method_body})
            split_methods_response = response
            split_dict = {
                method['method_name']: method['method_body']
                for method in split_methods_response['split_methods']
            }
            print(f"Method '{method_name}' was successfully split into {len(split_methods_response['split_methods'])} test units.")
            split_result.append((method_name, method_body, split_dict))
            new_body = "\n\n".join(method['method_body'] for method in split_methods_response['split_methods'])
            new_test_methods.append((method_name, new_body))
            # print(f"Method '{method_name}' was successfully split into {len(split_methods_response['split_methods'])} test units.")
        except OutputParserException as e:
            print(f"Failed to parse model output: {e}")
            print(f"Raw model output: {e.llm_output}")
        except Exception as e:
            print(f"Failed to split method '{method_name}'. Error: {str(e)}")
    save_split_result(file_path, split_result)
    return new_test_methods

def check_split_exists(file_path: str) -> bool:
    """检查是否已存在拆分结果文件"""
    expected_json_path = os.path.join(
        os.path.dirname(file_path),
        f"{os.path.splitext(os.path.basename(file_path))[0]}_split_raw.json"
    )
    # 检查文件是否存在
    if not os.path.exists(expected_json_path):
        return False
    
    # 读取文件内容并检查 split_methods 字段
    try:
        with open(expected_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查 split_methods 是否存在且不为空
        if "split_methods" not in data:
            return False
        
        split_methods = data["split_methods"]
        if not split_methods or len(split_methods) <= 1:
            return False
        
        return True
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading or parsing JSON file: {e}")
        return False

def process_single_file(file_path: str) -> None:
    """Process a single test file"""
    print(f"Processing file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    # 检查是否已存在拆分结果
    if check_split_exists(file_path):
        print(f"Split result already exists for {file_path}, skipping...")
        return
        
    test_methods = get_test_methods(file_path)
    if test_methods:
        split_assertions(file_path, test_methods, original_content)

def process_directory(directory: str) -> None:
    test_files = glob.glob(os.path.join(directory, '**/src/test/**/*Test.java'), recursive=True)
    print(f"Found {len(test_files)} test files")
    
    # 首先过滤掉已经处理过的文件
    unprocessed_files = [f for f in test_files if not check_split_exists(f)]
    print(f"Found {len(unprocessed_files)} unprocessed test files")
    
    if not unprocessed_files:
        print("No files need to be processed. Exiting...")
        return
    
    # 只处理未处理的文件
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_single_file, file_path) 
                  for file_path in unprocessed_files]
        
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file: {str(e)}")


# 指定要扫描的目录
if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     print("Usage: python script.py <directory>")
    #     sys.exit(1)
    # directory_to_scan = sys.argv[1]r
    # process_directory(directory_to_scan)
    # delay_seconds = 3 * 60 * 60
    
    # # 延时三小时
    # print(f"Waiting for {delay_seconds} seconds (3 hours) before starting...")
    # time.sleep(delay_seconds)
    directory_to_scan = r'D:\learn\junit-test\BPlusTree\BPlusTree'
    process_directory(directory_to_scan)
