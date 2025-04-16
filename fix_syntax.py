import json
import os
from typing import List, Tuple
import javalang
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from javalang.parser import JavaSyntaxError
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field, validator, field_validator
from langchain_core.output_parsers import PydanticOutputParser
# 初始化模型
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["DEEPSEEK_API_KEY"] = "XXXXXXXXXXXXX"
from langchain_deepseek import ChatDeepSeek

model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # 其他参数...
)


def check_syntax(full_class_content: str) -> Tuple[bool, str]:
    """基于完整类内容的语法检查"""
    try:
        javalang.parse.parse(full_class_content)
        return True, ""
    except JavaSyntaxError as e:
        return False, f"Syntax error : {e.at} {e.description}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


# 新增：定义结构化输出模型
class CodeFixResult(BaseModel):
    fixed_code: List[str] = Field(description="修复后的完整方法列表，每个元素必须是一个完整的方法体，包含方法头、测试逻辑和断言")
    success: bool = Field(description="是否成功修复")

def fix_with_context(error_code: str, error_msg: str, expected_method_count: int) -> Tuple[
    str, bool]:
    parser = PydanticOutputParser(pydantic_object=CodeFixResult)
    one_shot_example = """
    [示例 - 需要修复的代码片段]
    public void testExample_1() {
        int x = 10
        System.out.println(x);
    }
    
    public void testExample_2() {
        int x = 0
        System.out.println(x));
    }

    [示例 - 编译错误信息]
    Syntax error: expected ';' at line 2

    [示例 - 修复后的代码]
    {
      "fixed_code": [
        "public void testExample() {\n    int x = 10;\n    System.out.println(x);\n}",
        "public void testExample() {\n    int x = 0;\n    System.out.println(x);\n}"
      ],
      "success": true
    }
        """
    # 构建提示模板
    prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个Java代码修复专家。请修复测试用例中的语法错误，要求：
1. 保持测试方法的业务逻辑不变
2. 仅修复语法错误不改变代码结构
3. 输出必须与原方法数量不变（当前应有{expected_method_count}个方法）
4. 参照需要修复的代码片段输出所有完整的方法
5.严格按照以下格式返回
\n{format_instructions}
"""),
    ("human", """以下是一个示例：
{one_shot_example}

[需要修复的代码片段]
{error_code}

[编译错误信息]
{error_msg}

请生成{expected_method_count}个独立方法，严格按示例格式修复：""")
]).partial(format_instructions=parser.get_format_instructions())

    # 构建链
    chain = prompt | model | parser

    try:
        # 调用模型并解析输出
        response = chain.invoke({ "error_code": error_code,
                                    "error_msg": error_msg,
                                    "expected_method_count": expected_method_count,
                                    "one_shot_example": one_shot_example})
        if expected_method_count != len(response.fixed_code):
            print(f"与实际预期方法数量不符: 预期{expected_method_count}, 实际{len(response.fixed_code)}")
            return error_code, False
        print(f"修复成功: {response.fixed_code}")
        return "\n\n".join(response.fixed_code), response.success
    except Exception as e:
        print(f"修复失败: {str(e)}")
        return error_code, False


# 新增的结构化输出模型
# class TestCaseFixResult(BaseModel):
#     fixed_code: List[str] = Field(description="修复后的完整测试方法列表")
#     success: bool = Field(description="是否成功修复")
#     modification_notes: List[str] = Field(description="修复说明列表")


# 增强的修复函数
def fix_test_case(
        error_code: str,
        error_msg: str,
        original_test_case: str,
        expected_method_count: int
) -> Tuple[str, bool]:
    """修复执行失败的测试用例"""
    parser = PydanticOutputParser(pydantic_object=CodeFixResult)

    example_template = """
    [原测试用例参考]
    public void testUserLogin_1() {
        setupUser();
        Response response = login("user", "pass123");
        assertEquals(200, response.getStatus());
        assertNotNull(response.getSession());
    }

    [拆分后的错误用例]
    public void testUserLogin_1() {
        setupUser();
        Response response = login("user", "pass123");
        assertEquals(200, response.getStatus());
    }
    
    public void testUserLogin_2() {
        setupUser();
        assertNotNull(response.getSession());
    }

    [测试失败信息]
    缺失符号response

    [修复后的用例]
    {{
      "fixed_code": [
        "public void testUserLogin_1() {{\n    Response response = login(\"user\", \"pass123\");\n    assertEquals(200, response.getStatus());\n}}",
        "public void testUserLogin_2() {{\n    Response response = login(\"user\", \"pass123\");\n    assertNotNull(response.getSession());\n}}"
      ],
      "success": true
    }}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", """作为高级Java测试修复专家，请基于原测试用例修复拆分后的测试问题，要求：
1. 保持测试意图不变，仅修正执行错误，对于没有错误的方法，保持原方法。
2. 确保断言与原始用例逻辑一致
3. 方法数量保持{expected_method_count}个不变
4. 参考错误信息：{error_msg}
5. 严格按照以下格式返回
\n{format_instructions}"""),
        ("human", """参考示例：
{example_template}

[原始完整测试用例]
{original_test_case}

[需要修复的拆分用例]
{error_code}

[测试失败信息]
{error_msg}

请严格按格式输出修复方案：""")
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | model | parser

    try:
        response = chain.invoke({
            "error_code": error_code,
            "error_msg": error_msg,
            "original_test_case": original_test_case,
            "expected_method_count": expected_method_count,
            "example_template": example_template
        })

        # 验证方法数量一致性
        if len(response.fixed_code) != expected_method_count:
            print(f"方法数量不匹配！预期：{expected_method_count}，实际：{len(response.fixed_code)}\n 输出：{response.fixed_code}")
            return error_code, False

        return "\n\n".join(response.fixed_code), response.success
    except Exception as e:
        print(f"修复失败：{str(e)}")
        return error_code, False


if __name__ == "__main__":
    code = """
package com.google.gson;

import java.io.CharArrayReader;
import java.io.CharArrayWriter;
import java.io.StringReader;

import junit.framework.TestCase;

import com.google.gson.common.TestTypes.BagOfPrimitives;
import com.google.gson.internal.Streams;
import com.google.gson.stream.JsonReader;

/**
 * Unit test for {@link JsonParser}
 *
 * @author Inderjeet Singh
 */
public class JsonParserTest extends TestCase {
  private JsonParser parser;

  @Override
  protected void setUp() throws Exception {
    super.setUp();
    parser = new JsonParser();
  }

  public void testParseInvalidJson() {
    try {
      parser.parse("[[]");
      fail();
    } catch (JsonSyntaxException expected) { }
  }

  public void testParseUnquotedStringArrayFails() {
    JsonElement element = parser.parse("[a,b,c]");
    assertEquals("a", element.getAsJsonArray().get(0).getAsString());
    assertEquals("b", element.getAsJsonArray().get(1).getAsString());
    assertEquals("c", element.getAsJsonArray().get(2).getAsString());
    assertEquals(3, element.getAsJsonArray().size());
  }

  public void testParseString() {
    String json = "{a:10,b:'c'}";
    JsonElement e = parser.parse(json);
    assertTrue(e.isJsonObject());
    assertEquals(10, e.getAsJsonObject().get("a").getAsInt());
    assertEquals("c", e.getAsJsonObject().get("b").getAsString());
  }

  public void testParseEmptyString() {
    JsonElement e = parser.parse("\"   \"");
    assertTrue(e.isJsonPrimitive());
    assertEquals("   ", e.getAsString());
  }

  public void testParseEmptyWhitespaceInput() {
    JsonElement e = parser.parse("     ");
    assertTrue(e.isJsonNull());
  }

  public void testParseUnquotedSingleWordStringFails() {
    assertEquals("Test", parser.parse("Test").getAsString());
  }

  public void testParseUnquotedMultiWordStringFails() {
    String unquotedSentence = "Test is a test..blah blah";
    try {
      parser.parse(unquotedSentence);
      fail();
    } catch (JsonSyntaxException expected) { }
  }

  public void testParseMixedArray() {
    String json = "[{},13,\"stringValue\"]";
    JsonElement e = parser.parse(json);
    assertTrue(e.isJsonArray());

    JsonArray  array = e.getAsJsonArray();
    assertEquals("{}", array.get(0).toString());
    assertEquals(13, array.get(1).getAsInt());
    assertEquals("stringValue", array.get(2).getAsString());
  }

  public void testParseReader() {
    StringReader reader = new StringReader("{a:10,b:'c'}");
    JsonElement e = parser.parse(reader);
    assertTrue(e.isJsonObject());
    assertEquals(10, e.getAsJsonObject().get("a").getAsInt());
    assertEquals("c", e.getAsJsonObject().get("b").getAsString());
  }

  public void testReadWriteTwoObjects() throws Exception {
    Gson gson = new Gson();
    CharArrayWriter writer = new CharArrayWriter();
    BagOfPrimitives expectedOne = new BagOfPrimitives(1, 1, true, "one");
    writer.write(gson.toJson(expectedOne).toCharArray());
    BagOfPrimitives expectedTwo = new BagOfPrimitives(2, 2, false, "two");
    writer.write(gson.toJson(expectedTwo).toCharArray());
    CharArrayReader reader = new CharArrayReader(writer.toCharArray());

    JsonReader parser = new JsonReader(reader);
    parser.setLenient(true);
    JsonElement element1 = Streams.parse(parser);
    JsonElement element2 = Streams.parse(parser);
    BagOfPrimitives actualOne = gson.fromJson(element1, BagOfPrimitives.class);
    assertEquals("one", actualOne.stringValue);
    BagOfPrimitives actualTwo = gson.fromJson(element2, BagOfPrimitives.class);
    assertEquals("two", actualTwo.stringValue);
  }
}
    """


    err_code = """
    public void testParseEmptyString_1() {
    JsonElement e = parser.parse(\"\"   \"\");\n    assertTrue(e.isJsonPrimitive());\n}
    """

    # 测试
    success, info = check_syntax(code)
    c, r = fix_with_context(err_code, info, 1)

    print(c)
