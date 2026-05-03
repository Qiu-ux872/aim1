#include "TargetSelect.hpp"
#include <cmath>
#include <limits>
#include <algorithm>

using namespace std;
using namespace cv;

// 辅助函数：生成装甲板的唯一ID
static int generateArmorId(const Point2f& center) {
    return static_cast<int>(center.x * 10) + static_cast<int>(center.y * 100);
}

// 构造函数：初始化目标选择器
TargetSelector::TargetSelector(const Config& cfg)
    : m_w_center(cfg.target_select.w_center)       // 中心距离权重
    , m_w_distance(cfg.target_select.w_distance)   // 实际距离权重
    , m_w_stability(cfg.target_select.w_stability) // 稳定性权重
    , m_hysteresis(cfg.target_select.hysteresis)    // 切换迟滞阈值
    , m_max_tracked(cfg.target_select.max_tracked)  // 最大追踪帧数
    , m_max_distance_mm(cfg.target_select.max_distance_mm) // 最大有效距离
    , m_lost_timeout_ms(cfg.target_select.lost_timeout_ms)  // 目标丢失超时
    , m_current_target_id(-1)    // 当前追踪目标ID，-1表示无目标
    , m_tracked_count(0)         // 连续追踪帧数
    , m_lost_count(0)            // 连续丢失帧数
    , m_last_timestamp_ms(0.0)   // 上次更新的时间戳
{
}

// 重置函数：清空所有追踪状态
void TargetSelector::reset() {
    m_current_target_id = -1;
    m_tracked_count = 0;
    m_lost_count = 0;
    m_last_timestamp_ms = 0.0;
}

// 计算中心距离得分
double TargetSelector::centerDistanceScore(const Point2f& center) const {
    const Point2f image_center(320.0f, 240.0f);  // 图像中心点
    float max_dist = sqrt(320.0f*320.0f + 240.0f*240.0f);  // 最大可能距离（对角线）
    float dist = norm(center - image_center);    // 欧几里得距离
    return 1.0 - (dist / max_dist);  // 归一化：中心为1，边缘为0
}

// 计算实际距离得分
double TargetSelector::distanceScore(double distance_mm) const {
    if (distance_mm <= 0) return 0.0;  // 无效距离返回0
    // 归一化到[0,1]，并取反使距离越近得分越高
    double norm_dist = min(distance_mm, (double)m_max_distance_mm) / m_max_distance_mm;
    return 1.0 - norm_dist;
}

// 计算稳定性得分
double TargetSelector::stabilityScore(int tracked_count) const {
    int stable = min(tracked_count, m_max_tracked);  // 封顶到最大追踪帧数
    return static_cast<double>(stable) / m_max_tracked;
}

// 综合评分函数：计算装甲板的综合选择得分
double TargetSelector::computeScore(const Armor& armor, int tracked_count) const {
    double center_score = centerDistanceScore(armor.armor_center);  // 中心分
    double dist_score = distanceScore(armor.distance_mm);           // 距离分
    double stab_score = stabilityScore(tracked_count);             // 稳定分
    // 加权求和
    return m_w_center * center_score + m_w_distance * dist_score + m_w_stability * stab_score;
}

// 核心函数：选择最优攻击目标
const Armor* TargetSelector::select(const vector<Armor>& armors, double timestamp_ms) {
    
    // 情况1：无装甲板检测到
    if (armors.empty()) {
        m_lost_count++;  // 丢失帧数+1
        // 超时判断：超过设定时间没有检测到目标，则放弃当前目标
        if (m_current_target_id != -1 && (timestamp_ms - m_last_timestamp_ms) > m_lost_timeout_ms) {
            m_current_target_id = -1;
            m_tracked_count = 0;
        }
        m_last_timestamp_ms = timestamp_ms;
        return nullptr;
    }

    // 初始化最佳分数为负无穷，最佳索引为无效值
    double best_score = -numeric_limits<double>::max();
    int best_index = -1;

    // 遍历所有装甲板，计算得分并选择最优
    for (size_t i = 0; i < armors.size(); ++i) {
        const auto& armor = armors[i];
        int id = generateArmorId(armor.armor_center);  // 生成装甲板ID
        
        // 如果是当前追踪目标，使用累积的追踪帧数；否则视为新目标
        int tracked = (id == m_current_target_id) ? m_tracked_count : 0;
        double score = computeScore(armor, tracked);

        bool is_current = (id == m_current_target_id);
        if (is_current) {
            // 当前目标：直接比较得分
            if (score > best_score) {
                best_score = score;
                best_index = i;
            }
        } else {
            // 非当前目标：需要克服迟滞阈值才能切换
            // 迟滞阈值作用：防止在两个得分相近的目标间频繁切换
            float threshold = (m_current_target_id != -1) ? m_hysteresis : 0.0f;
            if (score > best_score + threshold) {
                best_score = score;
                best_index = i;
            }
        }
    }

    // 情况2：没有找到合适的候选目标
    if (best_index == -1) {
        if (m_current_target_id != -1) {
            m_lost_count++;
            // 超时判断：超过设定时间没有候选目标，则放弃当前目标
            if ((timestamp_ms - m_last_timestamp_ms) > m_lost_timeout_ms) {
                m_current_target_id = -1;
                m_tracked_count = 0;
            }
        }
        m_last_timestamp_ms = timestamp_ms;
        return nullptr;
    }

    // 情况3：确定最终选择的目标
    const Armor& selected = armors[best_index];
    int selected_id = generateArmorId(selected.armor_center);
    
    if (selected_id == m_current_target_id) {
        // 仍是当前目标：累加追踪计数，重置丢失计数
        m_tracked_count++;
        m_lost_count = 0;
    } else {
        // 切换到新目标：更新目标ID，重新开始计数
        m_current_target_id = selected_id;
        m_tracked_count = 1;
        m_lost_count = 0;
    }
    
    m_last_timestamp_ms = timestamp_ms;
    return &selected;
}
