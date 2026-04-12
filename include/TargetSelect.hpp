#pragma once

#include "Config.hpp"
#include "Param.hpp"
#include <vector>

class TargetSelector {
public:
    TargetSelector(const Config& cfg);
    ~TargetSelector() = default;

    // 主接口：输入当前帧所有装甲板，返回最佳目标（指针，若无可选则返回nullptr）
    const Armor* select(const std::vector<Armor>& armors, double timestamp_ms);

    // 重置目标选择状态
    void reset();

private:
    double computeScore(const Armor& armor, int tracked_count) const;
    double centerDistanceScore(const cv::Point2f& center) const;
    double distanceScore(double distance_mm) const;
    double stabilityScore(int tracked_count) const;

    // 配置参数
    float m_w_center;
    float m_w_distance;
    float m_w_stability;
    float m_hysteresis;
    int   m_max_tracked;
    float m_max_distance_mm;
    int   m_lost_timeout_ms;

    // 状态变量
    int   m_current_target_id;       // 当前选中的目标 ID（使用装甲板中心 hash）
    int   m_tracked_count;           // 连续跟踪帧数
    int   m_lost_count;              // 连续丢失帧数
    double m_last_timestamp_ms;      // 上一帧时间戳（用于失锁超时）
};