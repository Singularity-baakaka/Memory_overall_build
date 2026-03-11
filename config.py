#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
config.py - 全局配置

提供 LLM API 的默认配置。
用户应根据自己的实际部署修改此处的 API 地址、密钥和模型名称。

注意：请勿将真实的 API 密钥提交到版本控制系统中。
建议通过环境变量或独立的不受版本控制的配置文件管理敏感信息。
"""

LLM_CONFIG = {
    "api_url": "http://localhost:11434/v1/chat/completions",
    "api_key": "your-api-key-here",
    "model": "deepseek-chat",
    "temperature": 1.3,
    "max_tokens": 4000,
    "top_p": 0.9,
    "frequency_penalty": 0.3,
    "presence_penalty": 0.2,
    "timeout": 120,
}


def get_llm_config() -> dict:
    """
    返回 LLM API 配置字典。

    :return: 包含 api_url, api_key, model 等字段的配置字典
    """
    return LLM_CONFIG.copy()
