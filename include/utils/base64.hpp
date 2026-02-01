#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace yolo_edge {

/**
 * Base64 编解码工具
 * 简洁高效的实现
 */
namespace base64 {

namespace detail {

// Base64字符表
constexpr char ENCODE_TABLE[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// 解码表 (256个元素，无效字符用-1表示)
constexpr int8_t
    DECODE_TABLE
        [] =
            {
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, // 0-15
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, // 16-31
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1, -1,
                63, // 32-47  (+,/)
                52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -1, -1,
                -1, // 48-63  (0-9)
                -1, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                14, // 64-79  (A-O)
                15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1, -1,
                -1, // 80-95  (P-Z)
                -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                40, // 96-111 (a-o)
                41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, -1, -1, -1, -1,
                -1, // 112-127 (p-z)
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1,
};

} // namespace detail

/**
 * Base64编码
 * @param data 原始数据
 * @return Base64字符串
 */
inline std::string encode(const std::vector<uint8_t> &data) {
  if (data.empty())
    return "";

  size_t out_len = ((data.size() + 2) / 3) * 4;
  std::string result;
  result.reserve(out_len);

  size_t i = 0;
  while (i + 2 < data.size()) {
    uint32_t triple = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
    result += detail::ENCODE_TABLE[(triple >> 18) & 0x3F];
    result += detail::ENCODE_TABLE[(triple >> 12) & 0x3F];
    result += detail::ENCODE_TABLE[(triple >> 6) & 0x3F];
    result += detail::ENCODE_TABLE[triple & 0x3F];
    i += 3;
  }

  // 处理剩余字节
  if (i < data.size()) {
    uint32_t triple = data[i] << 16;
    if (i + 1 < data.size()) {
      triple |= data[i + 1] << 8;
    }

    result += detail::ENCODE_TABLE[(triple >> 18) & 0x3F];
    result += detail::ENCODE_TABLE[(triple >> 12) & 0x3F];

    if (i + 1 < data.size()) {
      result += detail::ENCODE_TABLE[(triple >> 6) & 0x3F];
    } else {
      result += '=';
    }
    result += '=';
  }

  return result;
}

/**
 * Base64编码 (从字符串)
 */
inline std::string encode(const std::string &data) {
  return encode(std::vector<uint8_t>(data.begin(), data.end()));
}

/**
 * Base64解码
 * @param encoded Base64字符串
 * @return 原始数据
 */
inline std::vector<uint8_t> decode(const std::string &encoded) {
  if (encoded.empty())
    return {};

  // 计算输出长度
  size_t len = encoded.size();
  size_t padding = 0;
  if (len >= 1 && encoded[len - 1] == '=')
    padding++;
  if (len >= 2 && encoded[len - 2] == '=')
    padding++;

  size_t out_len = (len / 4) * 3 - padding;
  std::vector<uint8_t> result;
  result.reserve(out_len);

  uint32_t buffer = 0;
  int bits = 0;

  for (char c : encoded) {
    if (c == '=' || c == '\n' || c == '\r' || c == ' ') {
      continue;
    }

    int8_t val = detail::DECODE_TABLE[static_cast<uint8_t>(c)];
    if (val < 0) {
      throw std::runtime_error("Invalid Base64 character");
    }

    buffer = (buffer << 6) | val;
    bits += 6;

    if (bits >= 8) {
      bits -= 8;
      result.push_back(static_cast<uint8_t>((buffer >> bits) & 0xFF));
    }
  }

  return result;
}

/**
 * 解码为字符串
 */
inline std::string decode_string(const std::string &encoded) {
  auto data = decode(encoded);
  return std::string(data.begin(), data.end());
}

/**
 * 移除data URL前缀
 * 例如: "data:image/jpeg;base64,/9j/..." -> "/9j/..."
 */
inline std::string strip_data_url(const std::string &data_url) {
  size_t comma_pos = data_url.find(',');
  if (comma_pos != std::string::npos) {
    return data_url.substr(comma_pos + 1);
  }
  return data_url;
}

} // namespace base64
} // namespace yolo_edge
