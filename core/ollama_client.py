"""
Ollama API 客户端模块
封装与本地 Ollama 服务的通信
"""

import json
import re
import time
import requests
import logging
import threading
import json_repair
from typing import Optional, Dict, Any
from requests.adapters import HTTPAdapter
from config.settings import (
    OLLAMA_API_URL,
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
    MODEL_CONFIG,
    HTTP_POOL_CONNECTIONS,
    HTTP_POOL_MAXSIZE,
    HTTP_MAX_RETRIES,
    get_model_config,
)

logger = logging.getLogger(__name__)

# 全局截断计数器（用于验证是否有正文损失）
_truncation_count = 0
_truncation_lock = threading.Lock()


def get_truncation_count() -> int:
    """获取截断发生次数（0 = 零正文损失）"""
    return _truncation_count


class VRAMManager:
    """
    显存分时复用管理器
    在 16GB 显存限制下，协调 LLM 推理模型和 Embedding 模型的显存分配。
    策略：
    - 构建期：LLM 推理为主，embedding 只在批量写入时短暂加载
    - 服务期：embedding 常驻（约 2GB），LLM 按需加载
    - 切换模型时，通过 keep_alive=0 立即释放旧模型显存
    """

    def __init__(self, total_vram_gb: int = 16):
        self.total_vram = total_vram_gb
        self._lock = threading.Lock()
        self._current_model: Optional[str] = None
        self._model_load_time: Optional[float] = None

    def ensure_model_loaded(self, model_name: str):
        """
        确保指定模型已加载。如果当前加载的是其他模型，先卸载旧模型。
        通过 Ollama 的 keep_alive 机制控制模型生命周期。
        """
        with self._lock:
            if self._current_model == model_name:
                return  # 已是当前模型，无需切换

            if self._current_model and self._current_model != model_name:
                self._unload_model(self._current_model)

            self._current_model = model_name
            self._model_load_time = time.time()
            logger.debug(f"VRAM: 模型已切换为 {model_name}")

    def _unload_model(self, model_name: str):
        """
        通过发送 keep_alive=0 请求，让 Ollama 立即卸载模型释放显存
        """
        try:
            requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": model_name, "keep_alive": 0},
                timeout=10,
            )
            logger.debug(f"VRAM: 模型 {model_name} 已卸载，显存已释放")
        except Exception as e:
            logger.warning(f"VRAM: 卸载模型 {model_name} 失败: {e}")

    def unload_all(self):
        """卸载所有已加载模型，释放全部显存"""
        with self._lock:
            if self._current_model:
                self._unload_model(self._current_model)
                self._current_model = None

    def get_current_model(self) -> Optional[str]:
        """获取当前加载的模型名"""
        return self._current_model


# 全局 VRAM 管理器实例
_global_vram_manager: Optional[VRAMManager] = None
_vram_manager_lock = threading.Lock()


def get_vram_manager() -> VRAMManager:
    """获取全局 VRAM 管理器实例"""
    global _global_vram_manager
    if _global_vram_manager is None:
        with _vram_manager_lock:
            if _global_vram_manager is None:
                _global_vram_manager = VRAMManager()
    return _global_vram_manager


class OllamaClient:
    """Ollama API 客户端"""

    def __init__(self):
        """初始化 HTTP Session"""
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=HTTP_POOL_CONNECTIONS,
            pool_maxsize=HTTP_POOL_MAXSIZE,
            max_retries=HTTP_MAX_RETRIES,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def check_health(self) -> bool:
        """检查 Ollama 服务状态"""
        print("🔍 正在检查 Ollama 服务状态...")
        try:
            resp = self.session.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            if not models:
                print("⚠️ Ollama 服务正常，但未发现任何模型。请先下载所需模型。")
                return False

            installed_models = {m["name"] for m in models}
            print(f"✅ Ollama 服务正常，已安装模型: {', '.join(installed_models)}")

            # 检查必需模型
            required_models = {
                "qwen3.5:9b",
                "qwen2.5:7b",
                "qwen14b:latest",
            }
            missing = required_models - installed_models
            if missing:
                print(f"⚠️ 缺少以下模型: {', '.join(missing)}")
                print("   请使用 'ollama pull <model_name>' 下载")
                return False

            # 可选模型（不强制）
            optional_models = {"qwen2.5:3b"}  # 3b 已不再用于核心流程，仅作为可选
            for opt in optional_models:
                if opt not in installed_models:
                    print(f"ℹ️ 可选模型未安装: {opt}（非必需，不影响核心流程）")

            return True
        except requests.exceptions.ConnectionError:
            print("❌ 无法连接到 Ollama 服务，请确保 Ollama 已启动")
            return False
        except Exception as e:
            print(f"❌ Ollama 健康检查失败: {e}")
            return False

    def chat(
        self,
        prompt: str,
        temperature: float = 0.2,
        stage: str = "A",
        max_retries: int = 3,
        system: Optional[str] = None,
    ) -> str:
        """
        调用 Ollama API 进行对话

        Args:
            prompt: 提示词
            temperature: 温度参数
            stage: Stage 标识（用于选择模型）
            max_retries: 最大重试次数

        Returns:
            模型响应文本
        """
        config = get_model_config(stage)
        model = config["model"]
        num_ctx = config["num_ctx"]
        num_predict = config.get("num_predict", 2048)  # 从 Stage 配置读取，默认 2048

        # Prompt 长度检查：防止 Ollama 静默截断导致指令丢失
        # 粗略估算 token 数：中文字符 ≈ 1.5 token，ASCII ≈ 0.25 token
        safe_token_limit = num_ctx - num_predict - 500  # 预留 500 token 缓冲
        estimated_tokens = sum(
            1.5 if ord(ch) > 127 else 0.25 for ch in prompt
        )
        if estimated_tokens > safe_token_limit:
            # 保留头部（指令）和尾部（JSON Schema），切除中间正文
            # 找到 JSON schema 起始位置（最后一个 { 块）
            json_start = prompt.rfind('\n{')
            if json_start == -1:
                json_start = prompt.rfind('{')
            if json_start > 0:
                head = prompt[:json_start]
                tail = prompt[json_start:]
                tail_tokens = sum(1.5 if ord(ch) > 127 else 0.25 for ch in tail)
                head_budget_tokens = safe_token_limit - tail_tokens
                head_budget_chars = int(head_budget_tokens / 1.5)  # 与估算一致，按 1.5 token/字
                if head_budget_chars < 200:
                    head_budget_chars = 200  # 至少保留 200 字指令
                # 按语义边界截断：找到截断点最近的段落/句子结尾
                trunc_pos = head_budget_chars
                for boundary in ('\n\n', '\n', '。', '！', '？', '.', '!', '?'):
                    pos = head.rfind(boundary, max(0, head_budget_chars - 500), head_budget_chars)
                    if pos > head_budget_chars // 2:
                        trunc_pos = pos + len(boundary)
                        break
                prompt = head[:trunc_pos] + "\n...(正文已截断)..." + tail
            else:
                # 无 JSON schema，从尾部截断
                max_chars = int(safe_token_limit / 1.5)
                prompt = prompt[:max_chars] + f"\n...(内容已截断，原始长度: {len(prompt)}字)"
            logger.error(
                f"⚠️ [Stage {stage}] Prompt 过长（估算 {int(estimated_tokens)} token，"
                f"上限 {safe_token_limit} token），已截断正文，保留指令和输出格式"
            )
            with _truncation_lock:
                global _truncation_count
                _truncation_count += 1

        # 显存分时复用：确保当前模型已加载，必要时卸载旧模型
        vram = get_vram_manager()
        vram.ensure_model_loaded(model)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,  # 流式模式：对长响应更稳定，不会因响应太大导致一次性返回超时
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "num_gpu": 99,  # 强制模型留在 GPU，防止 Ollama 偷偷卸载到 CPU 导致卡死
            },
        }

        for attempt in range(max_retries):
            try:
                with self.session.post(
                    OLLAMA_API_URL,
                    json=payload,
                    stream=True,
                    timeout=(30, OLLAMA_TIMEOUT),  # (连接超时, 读取超时)
                ) as resp:
                    resp.raise_for_status()
                    full_content = []
                    for line in resp.iter_lines():
                        if line:
                            try:
                                json_resp = json.loads(line.decode("utf-8"))
                                if (
                                    "message" in json_resp
                                    and "content" in json_resp["message"]
                                ):
                                    full_content.append(json_resp["message"]["content"])
                            except json.JSONDecodeError:
                                continue
                    return "".join(full_content)

            except requests.exceptions.Timeout:
                logger.warning(f"\u26a0\ufe0f 请求超时 (尝试 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                continue

            except requests.exceptions.RequestException as e:
                logger.error(f"\u2764 请求失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                continue

            except Exception as e:
                logger.error(f"\u2764 未知错误: {e}")
                return ""

        logger.error(f"❌ 达到最大重试次数，返回空响应")
        return ""

    def close(self):
        """关闭 HTTP Session"""
        self.session.close()


# JSON 解析工具函数
JSON_BLOCK_RE = re.compile(r"`{3}json\s*(\{.*\})\s*`{3}", re.DOTALL)


def extract_raw_json(text: str) -> str:
    """
    从文本中提取 JSON 字符串。
    使用括号计数状态机：跟踪花括号嵌套深度 + 字符串上下文 + 转义字符，
    能正确处理 LLM 输出中的嵌套 JSON 和字符串内的花括号。
    """
    if not text:
        return ""

    # 尝试提取 ```json ... ``` 块
    match = JSON_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()

    # 括号计数状态机
    start = text.find("{")
    if start == -1:
        return text

    brace_count = 0
    in_string = False
    escape = False
    last_open_brace_idx = -1  # 只跟踪 { 的位置，用于未闭合时截断

    for idx, char in enumerate(text[start:]):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
        if not in_string:
            if char == "{":
                brace_count += 1
                last_open_brace_idx = idx
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    return text[start : start + idx + 1]

    # 未闭合括号，尝试修复：从最后一个 { 处截断并补齐 }
    if last_open_brace_idx != -1:
        return text[start : start + last_open_brace_idx + 1] + "}" * brace_count
    return text[start:]


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """安全解析 JSON，支持修复常见格式错误"""
    if not text:
        return None

    # 检测是否被截断修复：原始文本括号数 vs 修复后括号数
    orig_open = text.count('{') + text.count('}')
    raw = extract_raw_json(text)
    if not raw:
        return None
    was_truncated = (raw.count('{') + raw.count('}')) != orig_open

    try:
        result = json.loads(raw)
        if was_truncated and isinstance(result, dict):
            result["_truncated_by_safe_parse"] = True
        return result
    except json.JSONDecodeError:
        # 尝试使用 json_repair 修复
        try:
            return json_repair.repair_json(raw, return_objects=True)
        except Exception:
            pass

        # 尝试手动修复常见问题
        try:
            # 移除尾部逗号
            fixed = re.sub(r",\s*([}\]])", r"\1", raw)
            return json.loads(fixed)
        except Exception:
            pass

        return None


# 全局客户端实例
_global_client: Optional[OllamaClient] = None
_client_lock = threading.Lock()


def get_ollama_client() -> OllamaClient:
    """获取全局 Ollama 客户端实例（线程安全）"""
    global _global_client
    if _global_client is None:
        with _client_lock:
            # 双重检查锁定
            if _global_client is None:
                _global_client = OllamaClient()
    return _global_client


def ollama_chat(prompt: str, temperature: float = 0.2, stage: str = "A") -> str:
    """便捷函数：调用 Ollama API"""
    client = get_ollama_client()
    return client.chat(prompt, temperature, stage)
