#include "TargetSelect.hpp"
#include <cmath>
#include <limits>
#include <algorithm>

using namespace std;
using namespace cv;

// 辅助函数：生成装甲板的唯一ID（基于中心坐标）
static int generateArmorId(const Point2f& center) {
    return static_cast<int>(center.x * 10) + static_cast<int>(center.y * 100);
}

TargetSelector::TargetSelector(const Config& cfg)
    : m_w_center(cfg.target_select.w_center)
    , m_w_distance(cfg.target_select.w_distance)
    , m_w_stability(cfg.target_select.w_stability)
    , m_hysteresis(cfg.target_select.hysteresis)
    , m_max_tracked(cfg.target_select.max_tracked)
    , m_max_distance_mm(cfg.target_select.max_distance_mm)
    , m_lost_timeout_ms(cfg.target_select.lost_timeout_ms)
    , m_current_target_id(-1)
    , m_tracked_count(0)
    , m_lost_count(0)
    , m_last_timestamp_ms(0.0)
{
}

void TargetSelector::reset() {
    m_current_target_id = -1;
    m_tracked_count = 0;
    m_lost_count = 0;
    m_last_timestamp_ms = 0.0;
}

double TargetSelector::centerDistanceScore(const Point2f& center) const {
    const Point2f image_center(320.0f, 240.0f);
    float max_dist = sqrt(320.0f*320.0f + 240.0f*240.0f);
    float dist = norm(center - image_center);
    return 1.0 - (dist / max_dist);
}

double TargetSelector::distanceScore(double distance_mm) const {
    if (distance_mm <= 0) return 0.0;
    double norm_dist = min(distance_mm, (double)m_max_distance_mm) / m_max_distance_mm;
    return 1.0 - norm_dist;  // 距离越近得分越高
}

double TargetSelector::stabilityScore(int tracked_count) const {
    int stable = min(tracked_count, m_max_tracked);
    return static_cast<double>(stable) / m_max_tracked;
}

double TargetSelector::computeScore(const Armor& armor, int tracked_count) const {
    double center_score = centerDistanceScore(armor.armor_center);
    double dist_score = distanceScore(armor.distance_mm);
    double stab_score = stabilityScore(tracked_count);
    return m_w_center * center_score + m_w_distance * dist_score + m_w_stability * stab_score;
}

const Armor* TargetSelector::select(const vector<Armor>& armors, double timestamp_ms) {
    if (armors.empty()) {
        m_lost_count++;
        if (m_current_target_id != -1 && (timestamp_ms - m_last_timestamp_ms) > m_lost_timeout_ms) {
            m_current_target_id = -1;
            m_tracked_count = 0;
        }
        m_last_timestamp_ms = timestamp_ms;
        return nullptr;
    }

    double best_score = -numeric_limits<double>::max();
    int best_index = -1;

    for (size_t i = 0; i < armors.size(); ++i) {
        const auto& armor = armors[i];
        int id = generateArmorId(armor.armor_center);
        int tracked = (id == m_current_target_id) ? m_tracked_count : 0;
        double score = computeScore(armor, tracked);

        bool is_current = (id == m_current_target_id);
        if (is_current) {
            if (score > best_score) {
                best_score = score;
                best_index = i;
            }
        } else {
            float threshold = (m_current_target_id != -1) ? m_hysteresis : 0.0f;
            if (score > best_score + threshold) {
                best_score = score;
                best_index = i;
            }
        }
    }

    if (best_index == -1) {
        if (m_current_target_id != -1) {
            m_lost_count++;
            if ((timestamp_ms - m_last_timestamp_ms) > m_lost_timeout_ms) {
                m_current_target_id = -1;
                m_tracked_count = 0;
            }
        }
        m_last_timestamp_ms = timestamp_ms;
        return nullptr;
    }

    const Armor& selected = armors[best_index];
    int selected_id = generateArmorId(selected.armor_center);
    if (selected_id == m_current_target_id) {
        m_tracked_count++;
        m_lost_count = 0;
    } else {
        m_current_target_id = selected_id;
        m_tracked_count = 1;
        m_lost_count = 0;
    }
    m_last_timestamp_ms = timestamp_ms;
    return &selected;
}