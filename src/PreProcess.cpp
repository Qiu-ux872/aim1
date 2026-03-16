#include "PreProcess.hpp"
#include "Config.hpp"

// 排列角点
vector<Point2f> sortCorners(const RotatedRect& rect) {
    Point2f corners[4];
    rect.points(corners);
    
    // 获取矩形的尺寸和角度
    Size2f size = rect.size;
    float width = min(size.width, size.height);   // 短边
    float length = max(size.width, size.height);  // 长边
    float angle = rect.angle;  // 长轴与水平线夹角
    
    // 根据角度确定矩形的方向向量
    float cosA = cos(angle * CV_PI / 180);
    float sinA = sin(angle * CV_PI / 180);
    
    // 计算四个角点相对于中心的偏移
    Point2f center = rect.center;
    Point2f halfLength(0, length/2);
    Point2f halfWidth(width/2, 0);
    
    // 旋转偏移量
    Point2f offset1 = Point2f(
        halfLength.x * cosA - halfLength.y * sinA,
        halfLength.x * sinA + halfLength.y * cosA
    );
    Point2f offset2 = Point2f(
        halfWidth.x * cosA - halfWidth.y * sinA,
        halfWidth.x * sinA + halfWidth.y * cosA
    );
    
    // 计算四个角点（左上、右上、右下、左下）
    vector<Point2f> sorted(4);
    sorted[0] = center - offset1 - offset2;  // 左上
    sorted[1] = center - offset1 + offset2;  // 右上
    sorted[2] = center + offset1 + offset2;  // 右下
    sorted[3] = center + offset1 - offset2;  // 左下
    
    return sorted;
}

Mat PreProcess::process(const Mat& frame){
    Config& c = Config::get();
    // 转灰度
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    // 高斯模糊
    Mat blur;
    GaussianBlur(gray, blur, Size(c.preprocess.gaussian_k_size, c.preprocess.gaussian_k_size), c.preprocess.gaussian_sigma);
    // 降曝光
    Mat binary;
    gray.convertTo(binary, CV_8U, c.preprocess.rdc_exposure_x, c.preprocess.rdc_exposure_y);
    // OTSU二值化
    threshold(binary, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
    // 形态学操作
    Mat kernel = getStructuringElement(MORPH_RECT, Size(c.preprocess.morph_k_size, c.preprocess.morph_k_size));
    morphologyEx(binary, binary, MORPH_CLOSE, kernel);
    morphologyEx(binary, binary, MORPH_OPEN, kernel);

    imshow("binary", binary);
    return binary;
}

vector<LightBar> PreProcess::detectLightBars(const Mat& binary){
    Config& c = Config::get();
    // 提取轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarcy;
    findContours(binary, contours, hierarcy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 筛选灯条
    vector<LightBar> detected_bars;
    for(auto& contour : contours){
        // 面积筛选
        double area = contourArea(contour);
        if(area < c.preprocess.min_area || area > c.preprocess.max_area) continue;

        // 最小外接矩形
        RotatedRect rect = minAreaRect(contour);
        // 规范化宽高
        float width = rect.size.width;
        float length = rect.size.height;
        if(width > length) swap(width, length);

        // 对比宽高比
        float ratio = 0.0f; // 【必做】变量显式初始化
        const float eps = 1e-6; // 浮点精度容错值，避免除0
        if (width > eps) {
            ratio = length / width;
        }
        float angle = rect.angle;
        // 宽高比
        if(ratio < c.preprocess.min_ratio || ratio > c.preprocess.max_ratio) continue;
        // 偏转角
        if(abs(angle) < 90 - c.preprocess.max_angle) continue;
        // 矩形度
        vector<Point> hull;
        convexHull(contour, hull);
        double hullArea = contourArea(hull);
        double rect_rate = area / hullArea;
        if(rect_rate < c.preprocess.min_rect_rate || rect_rate > c.preprocess.max_rect_rate) continue;

        //排列角点
        LightBar light;
        light.bar_center = rect.center;
        light.bar_width = width;
        light.bar_length = length;
        light.bar_angle = angle;
        
        //获取并排列角点
        light.bar_pts = sortCorners(rect);

        detected_bars.push_back(light);
    }
    return detected_bars;
}

vector<Point2f> PreProcess::calculateArmorCorners(const Armor& armor){
    vector<Point2f> corners(4);

    // 获取左右灯条四个角点
    const vector<Point2f>& left_corners = armor.left.bar_pts;
    const vector<Point2f>& right_corners = armor.right.bar_pts;

    // 装甲板角点顺序： 左上，右上，右下，左下
    corners[0] = left_corners[0];
    corners[1] = right_corners[1];
    corners[2] = right_corners[2];
    corners[3] = left_corners[3];

    return corners;
}

vector<Armor> PreProcess::detectArmors(const vector<LightBar>& detected_bars){
    Config& c = Config::get();
    vector<Armor> armors;
    if(detected_bars.size() < 2){
        cerr << "[预处理] 识别到的灯条不足2,共识别到：" << detected_bars.size() << "根" << endl;
        return armors;
    }

    // 遍历所有可能的灯条
    for(size_t i = 0; i < detected_bars.size(); i++){
        for(size_t j = i + 1; j < detected_bars.size(); j++){
            const LightBar& left = detected_bars[i];
            const LightBar& right = detected_bars[j];

            // 高度差过滤
            float height_diff = abs(left.bar_length - right.bar_length);
            float height_avg = (left.bar_length + right.bar_length) / 2.0f;
            if(height_diff / height_avg > c.armor.max_height_diff) continue;

            // 计算中心点距离和比例
            float center_dist = norm(left.bar_center - right.bar_center);
            float dist_ratio = center_dist / height_avg;
            if(dist_ratio < c.armor.min_dist_ratio || dist_ratio > c.armor.max_dist_ratio) continue;

            // 计算两灯条中心连线角度
            float connect_angle = atan2(
                right.bar_center.y - left.bar_center.y,
                right.bar_center.x - right.bar_center.x
            ) * 180 / CV_PI;

            // 连线角度与灯条角度过滤
            float left_angle_diff = abs(connect_angle - (left.bar_angle + 90));
            float right_angle_diff = abs(connect_angle - (right.bar_angle + 90));

            // 角度归一化
            left_angle_diff = min(left_angle_diff, 180 - left_angle_diff);
            right_angle_diff = min(right_angle_diff, 180 - right_angle_diff);

            if(left_angle_diff > c.armor.max_angle_diff || right_angle_diff > c.armor.max_angle_diff) continue;

            // 通过筛选构建装甲板
            Armor armor;
            armor.left = left;
            armor.right = right;

            // 计算中心
            armor.armor_center.x = (left.bar_center.x + right.bar_center.x) / 2.0f;
            armor.armor_center.y = (left.bar_center.y + right.bar_center.y) / 2.0f;

            armor.armor_width = center_dist;
            armor.armor_height = height_avg;

            // 计算角度
            armor.armor_angle = (left.bar_angle + right.bar_angle) / 2.0f;

            //计算角点
            armor.armor_pts = calculateArmorCorners(armor);

            armors.push_back(armor);
        }
    }

    return armors;
}