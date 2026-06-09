import os
import json
import time
import uuid
import hashlib
import requests


def get_md5(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest().upper()


class TranslationClient:
    """奥云翻译客户端，带连接复用、指数退避和失败熔断"""
    def __init__(self, pid, appkey, max_retries=3, base_timeout=(10, 20)):
        self.pid = pid
        self.appkey = appkey
        self.max_retries = max_retries
        self.base_timeout = base_timeout  # (conn_timeout, read_timeout)
        self.session = requests.Session()
        self.failure_count = 0
        self.last_failure_time = 0
        self.cooldown_seconds = 60

        # 读取系统代理（如有）
        self.proxies = {
            "http": os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY"),
            "https": os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
        }
        self.session.proxies.update({k: v for k, v in self.proxies.items() if v})

    def _should_enter_cooldown(self):
        """连续失败超过阈值时进入冷却"""
        if self.failure_count >= 3:
            if time.time() - self.last_failure_time > self.cooldown_seconds:
                self.failure_count = 0  # 冷却结束
                return False
            return True
        return False

    def translate(self, text):
        if not text.strip():
            return text

        # 熔断检查
        if self._should_enter_cooldown():
            print(f"    ⚠ 连续失败过多，冷却 {self.cooldown_seconds} 秒，保留原文")
            return text

        url = "http://oy.nmgoyun.com/api/fy/v1"  # 也可用 https
        timestamp = str(int(time.time() * 1000))
        nonce = uuid.uuid4().hex

        # 签名参数（类型统一为整数）
        sign_params = {
            "appKey": self.appkey,
            "inputStr": text,
            "nonce": nonce,
            "pid": self.pid,
            "timestamp": timestamp,
            "type": 5
        }
        sorted_keys = sorted(sign_params.keys())
        sign_str = "&".join(f"{k}={sign_params[k]}" for k in sorted_keys)
        sign = get_md5(sign_str)

        payload = {
            "inputStr": text,
            "nonce": nonce,
            "pid": self.pid,
            "sign": sign,
            "timestamp": timestamp,
            "type": 5
        }

        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(url, json=payload, timeout=self.base_timeout)
                res_json = resp.json()
                if res_json.get("code") == "0000":
                    # 成功，重置失败计数
                    self.failure_count = 0
                    return res_json.get("data", text)
                else:
                    print(f"    ⚠ [接口错误] 第 {attempt+1} 次, code={res_json.get('code')}, msg={res_json.get('message')}")
                    # 参数错误或签名错误不再重试
                    if res_json.get("code") in ("0002", "0003", "0004"):
                        return text

            except requests.exceptions.Timeout:
                print(f"    ⚠ [超时] 第 {attempt+1} 次请求超时", end="")
            except Exception as e:
                print(f"    ⚠ [异常] 第 {attempt+1} 次请求失败: {e}", end="")

            # 指数退避
            if attempt < self.max_retries - 1:
                wait = 2 ** attempt  # 1, 2, 4 秒
                print(f"，等待 {wait} 秒后重试...")
                time.sleep(wait)

        # 所有重试失败，记录连续失败
        self.failure_count += 1
        self.last_failure_time = time.time()
        print(f"    ❌ [彻底失败] 保留原文")
        return text


def translate_srt_file(srt_input_path, srt_output_path, pid, appkey):
    """翻译整个 SRT 文件，返回是否全部成功（粗略）"""
    client = TranslationClient(pid, appkey)

    with open(srt_input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = content.strip().split('\n\n')
    translated_blocks = []
    total = len(blocks)

    for i, block in enumerate(blocks):
        lines = block.split('\n')
        if len(lines) >= 3:
            idx = lines[0]
            timestamp = lines[1]
            original_text = " ".join(lines[2:])
            print(f"[{i+1}/{total}] 原文: {original_text}")

            translated = client.translate(original_text)
            print(f"          -> 译文: {translated}\n")

            translated_blocks.append(f"{idx}\n{timestamp}\n{translated}")
        else:
            translated_blocks.append(block)

        # 避免请求过快（限流保护）
        time.sleep(0.5)

    with open(srt_output_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(translated_blocks) + "\n")

    print(f"✅ 翻译完成，结果保存至: {srt_output_path}")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "api_configure", "api_configuration.json")
    srt_input = os.path.join(current_dir, "test.srt")
    srt_output = os.path.join(current_dir, "test_mn.srt")

    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    pid = cfg.get("pid", "")
    appkey = cfg.get("appKey", "")
    if not pid or not appkey:
        print("❌ 配置文件中 pid 或 appKey 为空")
        return

    if not os.path.exists(srt_input):
        print(f"❌ 输入 SRT 文件不存在: {srt_input}")
        return

    print(f"🚀 开始翻译 {srt_input}")
    translate_srt_file(srt_input, srt_output, pid, appkey)


if __name__ == "__main__":
    main()