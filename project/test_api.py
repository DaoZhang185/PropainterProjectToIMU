import os
import json
import time
import uuid
import hashlib
import urllib.parse
import requests


def get_md5(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest().upper()


def test_translation():
    # 1. 读取配置文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "api_configure", "api_configuration.json")

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

    # 2. 准备测试数据
    text = "九十三年的风雨历程，留下了不可磨灭的印记。"  # 换成你想要的中文测试句子
    url = "http://oy.nmgoyun.com/api/fy/v1"
    timestamp = str(int(time.time() * 1000))
    nonce = uuid.uuid4().hex

    params = {
        "inputStr": text,
        "nonce": nonce,
        "pid": pid,
        "timestamp": timestamp,
        "type": "5",  # 5: 汉 -> 传统蒙文
        "appKey": appKey
    }

    print(f"正在测试翻译: '{text}' ...\n")

    # 3. 严格 MD5 签名生成
    sorted_keys = sorted(params.keys())
    temp_list = []
    for k in sorted_keys:
        val = str(params[k])
        encoded_val = urllib.parse.quote_plus(val)
        temp_list.append(f"{k}={encoded_val}")

    query_string = "&".join(temp_list)
    sign = get_md5(query_string)

    # 4. 发起请求
    payload = {
        "inputStr": text,
        "nonce": nonce,
        "pid": pid,
        "sign": sign,
        "timestamp": timestamp,
        "type": 5
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        res_json = resp.json()
        print("=== 接口返回原始数据 ===")
        print(json.dumps(res_json, indent=4, ensure_ascii=False))

        if res_json.get("code") == "0000":
            print(f"\n✅ 翻译成功！\n蒙文结果: {res_json.get('data')}")
        else:
            print(f"\n❌ 接口报错，错误码: {res_json.get('code')}, 信息: {res_json.get('message')}")

    except Exception as e:
        print(f"❌ 网络请求异常: {e}")


if __name__ == "__main__":
    test_translation()