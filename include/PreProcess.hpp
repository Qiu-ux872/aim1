#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include "Param.hpp"

using namespace std;
using namespace cv;

class PreProcess{
public:
    static Mat process(const Mat& frame);
    static vector<LightBar> detectLightBars(const Mat& binary);
    static vector<Armor> detectArmors(const vector<LightBar>& detected_bars);
    static vector<Point2f> calculateArmorCorners(const Armor& armor);
};