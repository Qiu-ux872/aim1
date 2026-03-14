#pragma once

#include "Param.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "PreProcess.hpp"

using namespace std;
using namespace cv;

class ArmorModel{
public:
    static constexpr float ARMOR_HEIGHT = 125.0f;    // 装甲板高度（单位：mm）
    static constexpr float ARMOR_WIDTH = 135.0f;     // 装甲板宽度（单位：mm）

    static vector<Point3f> getArmorPoints(){
        vector<Point3f> points;
        float w = ARMOR_WIDTH  / 2.0f;
        float h = ARMOR_HEIGHT / 2.0f;

        //顺序：左上，右上，右下，左下
        points.emplace_back(-w,  h, 0);//左上
        points.emplace_back( w,  h, 0);//右上
        points.emplace_back( w, -h, 0);//右下
        points.emplace_back(-w, -h, 0);//左下

        return points;
    }
    // 为了兼容原有代码，保留这个接口但忽略armorType参数
    static std::vector<cv::Point3f> getPointsByType(int armorType = 0) {
        return getArmorPoints();
    }
};

class PnPSolver{
public:
    PnPSolver(){
        
    }
};