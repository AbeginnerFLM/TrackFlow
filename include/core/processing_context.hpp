#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <optional>
#include <any>
#include <unordered_map>

namespace yolo_edge {

/**
 * 检测结果结构体
 * 包含目标检测和跟踪的所有信息
 */
struct Detection {
    int track_id = -1;              // 跟踪ID (-1表示未跟踪)
    int class_id = 0;               // 类别ID
    std::string class_name;         // 类别名称
    float confidence = 0.0f;        // 置信度
    cv::RotatedRect obb;            // OBB: center, size, angle(度)
    
    // 地理坐标 (可选, 由GeoTransformer填充)
    std::optional<double> lon;      // 经度
    std::optional<double> lat;      // 纬度
    std::optional<double> ground_x; // 相对原点的米 (东向)
    std::optional<double> ground_y; // 相对原点的米 (北向)
};

/**
 * 处理上下文
 * 在Pipeline中各处理器之间传递数据
 */
class ProcessingContext {
public:
    // === 核心数据 ===
    cv::Mat frame;                          // 当前帧图像
    int frame_id = 0;                       // 帧序号
    std::string session_id;                 // 会话ID
    std::string request_id;                 // 请求ID
    std::vector<Detection> detections;      // 检测/跟踪结果
    
    // === 计时信息 (毫秒) ===
    double decode_time_ms = 0.0;
    double infer_time_ms = 0.0;
    double track_time_ms = 0.0;
    double geo_time_ms = 0.0;
    double total_time_ms = 0.0;
    
    // === 扩展数据存储 ===
    
    /**
     * 设置扩展数据
     */
    template<typename T>
    void set(const std::string& key, T value) {
        extras_[key] = std::move(value);
    }
    
    /**
     * 获取扩展数据 (不存在则抛异常)
     */
    template<typename T>
    T get(const std::string& key) const {
        return std::any_cast<T>(extras_.at(key));
    }
    
    /**
     * 获取扩展数据 (不存在则返回默认值)
     */
    template<typename T>
    T get_or(const std::string& key, T default_val) const {
        auto it = extras_.find(key);
        if (it == extras_.end()) {
            return default_val;
        }
        try {
            return std::any_cast<T>(it->second);
        } catch (const std::bad_any_cast&) {
            return default_val;
        }
    }
    
    /**
     * 检查是否存在某个键
     */
    bool has(const std::string& key) const {
        return extras_.find(key) != extras_.end();
    }
    
    /**
     * 移除扩展数据
     */
    void remove(const std::string& key) {
        extras_.erase(key);
    }
    
    /**
     * 清空扩展数据
     */
    void clear_extras() {
        extras_.clear();
    }

private:
    std::unordered_map<std::string, std::any> extras_;
};

} // namespace yolo_edge
