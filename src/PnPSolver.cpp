#include "PnPSolver.hpp"
#include <iostream>

PnPSolver::PnPSolver() {
    // 默认相机参数（备用）
    cameraMatrix = (Mat_<double>(3,3) << 800, 0, 320,
                                         0, 800, 240,
                                         0,   0,   1);
    distCoeffs = Mat::zeros(5, 1, CV_64F);

    // 小装甲板三维点（宽135mm，高125mm）
    float w = 135.0f / 2.0f;
    float h = 125.0f / 2.0f;
    armorPoints = {
        Point3f(-w, -h, 0),  // 左上
        Point3f( w, -h, 0),  // 右上
        Point3f( w,  h, 0),  // 右下
        Point3f(-w,  h, 0)   // 左下
    };
}

bool PnPSolver::loadCameraParams(const string& filename) {
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "【警告】无法打开相机参数文件: " << filename << "，将使用默认参数！" << endl;
        return false;
    }

    Mat tmpCam, tmpDist;
    vector<string> camKeys = {"camera_matrix", "cameraMatrix"};
    for (const auto& key : camKeys) {
        fs[key] >> tmpCam;
        if (!tmpCam.empty()) break;
    }
    if (tmpCam.empty()) {
        cerr << "【警告】文件中未找到相机内参矩阵，将使用默认参数！" << endl;
        fs.release();
        return false;
    }
    if (tmpCam.rows != 3 || tmpCam.cols != 3) {
        cerr << "【警告】相机内参矩阵应为3x3，实际为 "
             << tmpCam.rows << "x" << tmpCam.cols << "，将使用默认参数！" << endl;
        fs.release();
        return false;
    }

    vector<string> distKeys = {"dist_coeffs", "distortion_coeffs"};
    for (const auto& key : distKeys) {
        fs[key] >> tmpDist;
        if (!tmpDist.empty()) break;
    }
    if (tmpDist.empty()) {
        cerr << "【警告】文件中未找到畸变系数，将使用零向量！" << endl;
        tmpDist = Mat::zeros(5, 1, CV_64F);
    } else {
        if (tmpDist.rows == 1 && tmpDist.cols > 1) {
            tmpDist = tmpDist.reshape(1, tmpDist.cols);
        } else if (tmpDist.cols != 1 || tmpDist.rows < 4) {
            cerr << "【警告】畸变系数格式异常，将使用零向量！" << endl;
            tmpDist = Mat::zeros(5, 1, CV_64F);
        }
    }

    cameraMatrix = tmpCam.clone();
    distCoeffs = tmpDist.clone();
    fs.release();
    cout << "相机参数加载成功！" << endl;
    return true;
}

void PnPSolver::setCameraParams(const Mat& cameraMat, const Mat& distMat) {
    cameraMatrix = cameraMat.clone();
    distCoeffs = distMat.clone();
}

PnPResult PnPSolver::solveArmorPnP(const Armor& armor) {
    return solvePnP(armor.armor_pts);
}

PnPResult PnPSolver::solvePnP(const vector<Point2f>& imagePoints) {
    PnPResult result;
    result.isValid = false;

    if (imagePoints.size() != 4) {
        cerr << "【错误】需要4个图像点，实际只有 " << imagePoints.size() << " 个" << endl;
        return result;
    }

    // 第一次解算
    bool success = cv::solvePnP(armorPoints, imagePoints,
                                 cameraMatrix, distCoeffs,
                                 result.rotationVec, result.translationVec,
                                 false, SOLVEPNP_IPPE);
    if (!success) {
        cerr << "【警告】PnP解算失败！" << endl;
        return result;
    }

    // 迭代优化（提高精度）
    cv::solvePnP(armorPoints, imagePoints,
                 cameraMatrix, distCoeffs,
                 result.rotationVec, result.translationVec,
                 true, SOLVEPNP_ITERATIVE);

    // 计算旋转矩阵（用于欧拉角提取）
    Rodrigues(result.rotationVec, result.rotationMatrix);

    // 距离和平移向量
    result.distance = norm(result.translationVec);
    result.position = Point3f(result.translationVec.at<double>(0),
                               result.translationVec.at<double>(1),
                               result.translationVec.at<double>(2));

    // 计算欧拉角 仅用于调试
    calculateEulerAngles(result);

    // 用平移向量计算的 yaw 覆盖欧拉角 yaw 
    double x = result.translationVec.at<double>(0);
    double y = result.translationVec.at<double>(1);
    double z = result.translationVec.at<double>(2);
    double yaw_from_trans = atan2(x, z) * 180.0 / CV_PI;
    double pitch_from_trans = atan2(y, z) * 180.0 / CV_PI;
    result.yaw = yaw_from_trans;   //  result.yaw 是目标相对于相机的水平方位角
    result.pitch = pitch_from_trans;

    result.isValid = true;
    return result;
}

void PnPSolver::calculateEulerAngles(PnPResult& result) {
    Mat R = result.rotationMatrix;

    if (abs(R.at<double>(2,0)) < 0.999) {
        double yaw_rad   = atan2(R.at<double>(1,0), R.at<double>(0,0));
        double pitch_rad = asin(-R.at<double>(2,0));
        double roll_rad  = atan2(R.at<double>(2,1), R.at<double>(2,2));
        const double rad2deg = 180.0 / CV_PI;
        result.yaw   = yaw_rad * rad2deg;
        result.pitch = pitch_rad * rad2deg;
        result.roll  = roll_rad * rad2deg;
    } else {
        // 奇异情况（pitch = ±90°）
        result.yaw = 0;
        const double rad2deg = 180.0 / CV_PI;
        if (R.at<double>(2,0) > 0) {
            result.pitch = 90.0;
            result.roll  = atan2(R.at<double>(0,1), R.at<double>(0,2)) * rad2deg;
        } else {
            result.pitch = -90.0;
            result.roll  = atan2(-R.at<double>(0,1), -R.at<double>(0,2)) * rad2deg;
        }
    }
}


AngleSolver::AngleSolver() {
    const auto& config = Config::get();
    bulletSpeed = config.ballistic.bulletSpeed;
    gravity = config.ballistic.gravity;
    cameraOffset = Point3f(config.ballistic.cameraOffsetX,
                           config.ballistic.cameraOffsetY,
                           config.ballistic.cameraOffsetZ);
    cout << "偏移x：" << cameraOffset.x << "mm 偏移y：" << cameraOffset.y << "mm 偏移z：" << cameraOffset.z << "mm" << endl;
    cout << "G: " << gravity << " m/s^2" << endl;
    cout << "弹速: " << bulletSpeed << " m/s" << endl;
}

AimAngle AngleSolver::calculateAimAngle(const PnPResult& pnpResult) {
    AimAngle aim;
    aim.distance = static_cast<float>(pnpResult.distance / 1000.0); // mm -> m

    // 目标在相机坐标系下的位置（mm）
    Point3f targetCam = pnpResult.position;
    // 转换到枪口坐标系（考虑相机偏移）
    Point3f targetGimbal(
        targetCam.x - cameraOffset.x,
        targetCam.y - cameraOffset.y,
        targetCam.z - cameraOffset.z
    );

    float dz = targetGimbal.z / 1000.0f; // 前向距离（m）
    float dy = targetGimbal.y / 1000.0f; // 高度差（m），Y向下为正
    float dx = targetGimbal.x / 1000.0f; // 水平偏移（m）

    // 水平转角（rad）
    float yaw_rad = atan2(dx, dz);
    // 俯仰角（rad）
    float pitch_rad = solvePitch(dz, dy, bulletSpeed, gravity);

    const float rad2deg = 180.0f / static_cast<float>(CV_PI);
    aim.yaw   = yaw_rad * rad2deg;
    aim.pitch = pitch_rad * rad2deg;

    return aim;
}

float AngleSolver::solvePitch(float dz, float dy, float v, float g) {
    // 初始猜测：无重力时的 pitch
    float pitch = atan2(dy, dz);
    const int maxIter = 10;
    const float eps = 1e-4f;

    for (int i = 0; i < maxIter; i++) {
        float cosPitch = cos(pitch);
        float tanPitch = tan(pitch);
        float dyCalc = dz * tanPitch - (g * dz * dz) / (2 * v * v * cosPitch * cosPitch);
        float error = dy - dyCalc;
        if (fabs(error) < eps) break;

        // 数值求导
        float dp = 1e-4f;
        float pitch2 = pitch + dp;
        float cosPitch2 = cos(pitch2);
        float dyCalc2 = dz * tan(pitch2) - (g * dz * dz) / (2 * v * v * cosPitch2 * cosPitch2);
        float derivative = (dyCalc2 - dyCalc) / dp;
        if (fabs(derivative) < 1e-6f) break;
        pitch += error / derivative;
    }
    return pitch;
}
