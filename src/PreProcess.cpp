#include "PreProcess.hpp"
#include "Config.hpp"
#include <algorithm>

Mat PreProcess::camera_matrix;
Mat PreProcess::dist_coeffs;
extern Point2f projectPoint(const Point3f& pt, const Mat& cameraMatrix, const Mat& distCoeffs);

// 排列角点 - 基于坐标对比（按 y 排序再按 x 排序）
vector<Point2f> sortCorners(const RotatedRect &rect)
{
    Point2f corners[4];
    rect.points(corners);

    // 将四个点放入 vector
    vector<Point2f> pts(corners, corners + 4);

    // 按 y 坐标升序排序，如果 y 相同则按 x 升序
    sort(pts.begin(), pts.end(), [](const Point2f &a, const Point2f &b)
         {
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x; });

    // 前两个点为上排（y 较小），后两个点为下排（y 较大）
    vector<Point2f> top = {pts[0], pts[1]};
    vector<Point2f> bottom = {pts[2], pts[3]};

    // 在上排中按 x 排序（左到右）
    sort(top.begin(), top.end(), [](const Point2f &a, const Point2f &b)
         { return a.x < b.x; });
    // 在下排中按 x 排序
    sort(bottom.begin(), bottom.end(), [](const Point2f &a, const Point2f &b)
         { return a.x < b.x; });

    // 组装结果：左上、右上、右下、左下
    vector<Point2f> sorted(4);
    sorted[0] = top[0];    // 左上
    sorted[1] = top[1];    // 右上
    sorted[2] = bottom[1]; // 右下
    sorted[3] = bottom[0]; // 左下

    return sorted;
}

Mat PreProcess::process(const Mat &frame)
{
    Config &c = Config::get();

    // 检查输入图像
    if (frame.empty())
    {
        cerr << "[预处理] 输入图像为空！" << endl;
        return Mat();
    }

    // 转灰度
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    // 降曝光
    Mat contrast;
    gray.convertTo(contrast, CV_8U, c.preprocess.rdc_exposure_x, c.preprocess.rdc_exposure_y);

    // OTSU二值化
    Mat binary;
    double otsu_thresh = threshold(contrast, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // 输出OTSU阈值（每30帧输出一次）
    static int frame_count = 0;
    if (++frame_count % 30 == 0)
    {
        cout << "[预处理] OTSU阈值: " << otsu_thresh << endl;
    }

    // 高斯模糊
    Mat blur;
    GaussianBlur(binary, blur, Size(c.preprocess.gaussian_k_size, c.preprocess.gaussian_k_size), c.preprocess.gaussian_sigma);

    // 形态学操作
    Mat kernel = getStructuringElement(MORPH_RECT, Size(c.preprocess.morph_k_size, c.preprocess.morph_k_size));
    morphologyEx(blur, blur, MORPH_OPEN, kernel);
    morphologyEx(blur, blur, MORPH_CLOSE, kernel);

    imshow("binary", blur);
    return blur;
}

vector<LightBar> PreProcess::detectLightBars(const Mat &blur)
{
    Config &c = Config::get();
    // 提取轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(blur, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 筛选灯条
    vector<LightBar> detected_bars;
    int filtered_area = 0, filtered_ratio = 0, filtered_angle = 0, total = 0;

    for (auto &contour : contours)
    {
        total++;

        // 面积筛选
        double area = contourArea(contour);
        if (area < c.preprocess.min_area || area > c.preprocess.max_area)
        {
            filtered_area++;
            continue;
        }

        // 最小外接矩形
        RotatedRect rect = minAreaRect(contour);
        // 规范化宽高
        float width = rect.size.width;
        float length = rect.size.height;
        if (width > length)
            swap(width, length);

        // 对比宽高比
        const float eps = 1e-6f;
        if (width <= eps)
        {
            filtered_ratio++;
            continue;
        }

        float ratio = length / width;
        float angle = rect.angle;

        // 宽高比筛选
        if (ratio < c.preprocess.min_ratio)
            continue;
        if (ratio > c.preprocess.max_ratio)
            continue;

        // 偏转角
        const float VERTICAL_ANGLE = 90.0f;
        if (abs(angle) < VERTICAL_ANGLE - c.preprocess.max_angle)
        {
            filtered_angle++;
            continue;
        }

        // 排列角点
        LightBar light;
        light.bar_center = rect.center;
        light.bar_width = width;
        light.bar_length = length;
        light.bar_angle = angle;

        // 获取并排列角点
        light.bar_pts = sortCorners(rect);

        detected_bars.push_back(light);
    }

    // 输出筛选统计信息
    static int frame_count = 0;
    if (++frame_count % 10 == 0)
    {
        cout << "[灯条筛选] 总轮廓:" << total
             << " 面积过滤:" << filtered_area
             << " 宽高比过滤:" << filtered_ratio
             << " 角度过滤:" << filtered_angle
             << " 保留:" << detected_bars.size() << endl;
    }

    return detected_bars;
}

vector<Point2f> PreProcess::calculateArmorCorners(const Armor &armor)
{
    vector<Point2f> corners(4);

    // 收集装甲板的四个关键角点：左灯条的左上、左下 和 右灯条的右上、右下
    // 假设左右灯条的 bar_pts 已通过 sortCorners 排序为 [左上, 右上, 右下, 左下]
    vector<Point2f> allPts;
    allPts.push_back(armor.left.bar_pts[0]);  // 左灯条左上
    allPts.push_back(armor.left.bar_pts[3]);  // 左灯条左下
    allPts.push_back(armor.right.bar_pts[1]); // 右灯条右上
    allPts.push_back(armor.right.bar_pts[2]); // 右灯条右下

    // 按 y 坐标升序排序，y 小的为上排
    sort(allPts.begin(), allPts.end(), [](const Point2f &a, const Point2f &b)
         { return a.y < b.y; });

    // 前两个点为上排，后两个点为下排
    vector<Point2f> top(allPts.begin(), allPts.begin() + 2);
    vector<Point2f> bottom(allPts.begin() + 2, allPts.end());

    // 上排按 x 升序（左到右）
    sort(top.begin(), top.end(), [](const Point2f &a, const Point2f &b)
         { return a.x < b.x; });
    // 下排按 x 升序
    sort(bottom.begin(), bottom.end(), [](const Point2f &a, const Point2f &b)
         { return a.x < b.x; });

    // 组装顺序：左上、右上、右下、左下
    corners[0] = top[0];
    corners[1] = top[1];
    corners[2] = bottom[1];
    corners[3] = bottom[0];

    return corners;
}

vector<Armor> PreProcess::detectArmors(const vector<LightBar>& detected_bars, const Point3f* predictedPos)   // 注意：这里不要写默认参数
{
    Config &c = Config::get();
    vector<Armor> armors;
    if (detected_bars.size() < 2)
    {
        cerr << "[预处理] 识别到的灯条不足2,共识别到：" << detected_bars.size() << "根" << endl;
        return armors;
    }

    // 遍历所有可能的灯条
    for (size_t i = 0; i < detected_bars.size(); i++)
    {
        for (size_t j = i + 1; j < detected_bars.size(); j++)
        {
            const LightBar &left = detected_bars[i];
            const LightBar &right = detected_bars[j];

            // 高度差过滤
            float height_diff = abs(left.bar_length - right.bar_length);
            float height_avg = (left.bar_length + right.bar_length) / 2.0f;
            if (height_diff > c.armor.max_height_diff)
                continue;

            // 计算中心点距离
            float center_dist = norm(left.bar_center - right.bar_center);
            if (center_dist < c.armor.min_center_dist || center_dist > c.armor.max_center_dist)
                continue;

            // 计算两灯条中心连线角度
            float connect_angle = atan2(
                                      right.bar_center.y - left.bar_center.y,
                                      right.bar_center.x - left.bar_center.x) *
                                  180 / CV_PI;

            // 连线角度与灯条角度过滤
            float left_angle_diff = abs(connect_angle - (left.bar_angle + 90));
            float right_angle_diff = abs(connect_angle - (right.bar_angle + 90));

            left_angle_diff = min(left_angle_diff, 180 - left_angle_diff);
            right_angle_diff = min(right_angle_diff, 180 - right_angle_diff);

            if (left_angle_diff > c.armor.max_angle_diff || right_angle_diff > c.armor.max_angle_diff)
                continue;

            // 装甲板宽高比过滤
            float armor_w_h_ratio = center_dist / height_avg;
            if (armor_w_h_ratio < c.armor.min_w_h_ratio || armor_w_h_ratio > c.armor.max_w_h_ratio)
                continue;

            // 装甲板中心距与灯条比
            float half_center_dist = center_dist / 2.0f;
            float left_ratio = half_center_dist / left.bar_length;
            float right_ratio = half_center_dist / right.bar_length;
            float avg_ratio = (left_ratio + right_ratio) / 2.0f;
            if (avg_ratio < c.armor.min_center_bar_ratio || avg_ratio > c.armor.max_center_bar_ratio)
                continue;

            // 构建装甲板
            Armor armor;
            armor.left = left;
            armor.right = right;
            armor.armor_center.x = (left.bar_center.x + right.bar_center.x) / 2.0f;
            armor.armor_center.y = (left.bar_center.y + right.bar_center.y) / 2.0f;
            armor.armor_width = center_dist;
            armor.armor_height = height_avg;
            armor.armor_angle = (left.bar_angle + right.bar_angle) / 2.0f;
            armor.armor_pts = calculateArmorCorners(armor);

            armors.push_back(armor);
        }
    }

    // 卡尔曼稳定识别：如果提供了预测位置，选择最接近的装甲板
    if (predictedPos != nullptr && !armors.empty() && !PreProcess::camera_matrix.empty())
    {
        Point2f predImg = projectPoint(*predictedPos, PreProcess::camera_matrix, PreProcess::dist_coeffs);
        int bestIdx = 0;
        float minDist = norm(armors[0].armor_center - predImg);
        for (size_t i = 1; i < armors.size(); i++)
        {
            float dist = norm(armors[i].armor_center - predImg);
            if (dist < minDist)
            {
                minDist = dist;
                bestIdx = i;
            }
        }
        vector<Armor> selected;
        selected.push_back(armors[bestIdx]);
        return selected;
    }

    return armors;
}