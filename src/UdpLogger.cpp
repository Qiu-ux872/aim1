#include "UdpLogger.hpp"

UdpLogger::UdpLogger(const std::string& host, int port) {
    m_sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (m_sock < 0) {
        std::cerr << "[UDP] 创建socket失败" << std::endl;
        return;
    }

    memset(&m_addr, 0, sizeof(m_addr));
    m_addr.sin_family = AF_INET;
    m_addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host.c_str(), &m_addr.sin_addr) <= 0) {
        std::cerr << "[UDP] 无效的IP地址: " << host << std::endl;
        close(m_sock);
        m_sock = -1;
        return;
    }
    m_addr_len = sizeof(m_addr);
    std::cout << "[UDP] 目标 " << host << ":" << port << " 已设置" << std::endl;
}

UdpLogger::~UdpLogger() {
    if (m_sock != -1) close(m_sock);
}

double UdpLogger::safeDouble(double v) {
    // 处理 NaN
    if (v != v) return 0.0;
    // 处理无穷大
    if (v == std::numeric_limits<double>::infinity() ||
        v == -std::numeric_limits<double>::infinity()) return 0.0;
    return v;
}

void UdpLogger::send(double timestamp,
                     double pnp_distance,
                     double pnp_yaw, double pnp_pitch, double pnp_roll,
                     double aim_yaw, double aim_pitch,
                     double est_x, double est_y, double est_z,
                     double pred_x, double pred_y, double pred_z) {
    if (m_sock == -1) return;

    std::stringstream ss;
    ss << "{"
       << "\"timestamp\":" << safeDouble(timestamp) << ","
       << "\"pnp_distance\":" << safeDouble(pnp_distance) << ","
       << "\"pnp_yaw\":" << safeDouble(pnp_yaw) << ","
       << "\"pnp_pitch\":" << safeDouble(pnp_pitch) << ","
       << "\"pnp_roll\":" << safeDouble(pnp_roll) << ","
       << "\"aim_yaw\":" << safeDouble(aim_yaw) << ","
       << "\"aim_pitch\":" << safeDouble(aim_pitch) << ","
       << "\"est_x\":" << safeDouble(est_x) << ","
       << "\"est_y\":" << safeDouble(est_y) << ","
       << "\"est_z\":" << safeDouble(est_z) << ","
       << "\"pred_x\":" << safeDouble(pred_x) << ","
       << "\"pred_y\":" << safeDouble(pred_y) << ","
       << "\"pred_z\":" << safeDouble(pred_z)
       << "}";

    // ===========Debug===========
    static int count = 0;
    if (++count % 30 == 0) {
        std::cout << "[UDP] " << ss.str() << std::endl;
    }

    sendJson(ss.str());
}

void UdpLogger::sendJson(const std::string& json) {
    sendto(m_sock, json.c_str(), json.length(), 0,
           (struct sockaddr*)&m_addr, m_addr_len);
}