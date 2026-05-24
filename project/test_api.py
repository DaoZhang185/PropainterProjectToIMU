import os
import json
import time
import uuid
import hashlib
import requests


def get_md5(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest().upper()


def translate_to_mongolian(text, pid, appKey):
    if not text.strip():
        return text

    url = "http://oy.nmgoyun.com/api/fy/v1"
    timestamp = str(int(time.time() * 1000))
    nonce = uuid.uuid4().hex

    # 严格遵循主产线的签名机制：统一 type 为整数 5
    sign_params = {
        "appKey": appKey,
        "inputStr": text,
        "nonce": nonce,
        "pid": pid,
        "timestamp": timestamp,
        "type": 5
    }

    # 严格签名机制：按字典序排序，直接拼接原始字符串，弃用 URL 编码
    sorted_keys = sorted(sign_params.keys())
    sign_str = "&".join([f"{k}={sign_params[k]}" for k in sorted_keys])
    sign = get_md5(sign_str)

    payload = {
        "inputStr": text,
        "nonce": nonce,
        "pid": pid,
        "sign": sign,
        "timestamp": timestamp,
        "type": 5
    }

    # 网络代理穿透
    proxies = {
        "http": os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY"),
        "https": os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # (3秒连接, 5秒读取) 快连快断策略
            resp = requests.post(url, json=payload, timeout=(3, 5), proxies=proxies)
            res_json = resp.json()

            if res_json.get("code") == "0000":
                return res_json.get("data", text)
            else:
                print(
                    f"    ⚠ [接口报错] 第 {attempt + 1} 次请求，错误码: {res_json.get('code')}, 信息: {res_json.get('message')}")

        except requests.exceptions.Timeout:
            print(f"    ⚠ [超时] 第 {attempt + 1} 次请求超时，正在重试...")
        except Exception as e:
            print(f"    ⚠ [异常] 第 {attempt + 1} 次请求失败: {e}，正在重试...")

        time.sleep(1)  # 失败缓冲 1 秒

    print(f"    ❌ [彻底失败] 超过 {max_retries} 次仍无法连接，保留原文。")
    return text


def test_srt_translation():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "api_configure", "api_configuration.json")
    srt_input_path = os.path.join(current_dir, "test.srt")
    srt_output_path = os.path.join(current_dir, "test_mn.srt")

    if not os.path.exists(config_path):
        print(f"❌ 找不到配置文件: {config_path}")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        api_config = json.load(f)

    pid = api_config.get("pid", "")
    appKey = api_config.get("appKey", "")

    if not pid or not appKey:
        print("❌ PID或APPKEY为空，请检查配置文件！")
        return

    if not os.path.exists(srt_input_path):
        print(f"❌ 找不到测试文件: {srt_input_path}")
        return

    print(f"🚀 启动翻译引擎，开始处理: {srt_input_path}\n" + "=" * 40)

    with open(srt_input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 解析 SRT，以双换行符切割块
    blocks = content.strip().split('\n\n')
    translated_blocks = []

    total_blocks = len(blocks)
    for i, block in enumerate(blocks):
        lines = block.split('\n')
        # 标准的 SRT 块至少有 3 行：序号、时间轴、文本内容
        if len(lines) >= 3:
            idx = lines[0]
            timestamp = lines[1]
            # 应对单条字幕内存在换行的情况，将其合并为一行发送给 API
            text = " ".join(lines[2:])

            print(f"[{i + 1}/{total_blocks}] 提取原文: {text}")
            mn_text = translate_to_mongolian(text, pid, appKey)
            print(f"          -> 翻译结果: {mn_text}\n")

            # 重新组装 SRT 块
            translated_blocks.append(f"{idx}\n{timestamp}\n{mn_text}")
        else:
            translated_blocks.append(block)

    # 写入带有 _mn 后缀的新文件
    with open(srt_output_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(translated_blocks) + "\n")

    print("=" * 40 + f"\n✅ 全部测试完成！双语分离结果已输出至: {srt_output_path}")


if __name__ == "__main__":
    test_srt_translation()