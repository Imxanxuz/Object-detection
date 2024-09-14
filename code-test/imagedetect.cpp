#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace cv;
using namespace std;
using namespace dnn;

// Function to draw bounding box and label
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame, const vector<string>& classes) {
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 3);
    string label = format("%.2f", conf);
    if (!classes.empty()) {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ": " + label;
    }
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height), Point(left + labelSize.width, top + baseLine), Scalar(255, 255, 255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
}

int main() {
    // Paths to configuration and weight files
    string yolo_cfg = "C:/Users/chaii/OneDrive/Documents/SUN/projactNuttha/Yolov/yolov3-master/cfg/yolov4-tiny.cfg";
    string yolo_weights = "C:/Users/chaii/OneDrive/Documents/SUN/projactNuttha/Yolov/yolov3-master/cfg/yolov4-tiny.weights";
    string classes_file = "C:/Users/chaii/OneDrive/Documents/SUN/projactNuttha/Yolov/yolov3-master/cfg/coco.names";
    string image_file = "C:/Users/chaii/OneDrive/Documents/SUN/projactNuttha/picture/face picture/face13.jpg"; // Path to the input image

    // Load YOLO
    Net net = readNetFromDarknet(yolo_cfg, yolo_weights);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU); // Change to DNN_TARGET_CUDA if GPU is available

    // Load class names
    vector<string> classes;
    ifstream ifs(classes_file.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // Load input image
    Mat frame = imread(image_file);
    if (frame.empty()) {
        cerr << "Error loading image file" << endl;
        return -1;
    }

    // Resize the image to a smaller size for faster processing
    Size frameSize = Size(640, 360); // Resize the image to speed up processing
    Mat resizedFrame;
    resize(frame, resizedFrame, frameSize);

    // Create a blob and perform forward pass
    Mat blob;
    blobFromImage(resizedFrame, blob, 1.0 / 255.0, Size(416, 416), Scalar(0, 0, 0), true, false);
    net.setInput(blob);
    vector<Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    vector<Rect> boxes;
    vector<int> classIds;
    vector<float> confidences;

    for (const auto& out : outs) {
        for (int i = 0; i < out.rows; i++) {
            float confidence = out.at<float>(i, 4);
            if (confidence > 0.5) {
                for (int j = 5; j < out.cols; j++) {
                    float classConfidence = out.at<float>(i, j);
                    if (classConfidence > 0.5) {
                        classIds.push_back(j - 5);
                        confidences.push_back(confidence);
                        int centerX = static_cast<int>(out.at<float>(i, 0) * resizedFrame.cols);
                        int centerY = static_cast<int>(out.at<float>(i, 1) * resizedFrame.rows);
                        int width = static_cast<int>(out.at<float>(i, 2) * resizedFrame.cols);
                        int height = static_cast<int>(out.at<float>(i, 3) * resizedFrame.rows);
                        int left = centerX - width / 2;
                        int top = centerY - height / 2;
                        boxes.push_back(Rect(left, top, width, height));
                    }
                }
            }
        }
    }

    // Apply Non-Maximum Suppression to reduce overlapping bounding boxes
    vector<int> indices;
    NMSBoxes(boxes, confidences, 0.5, 0.4, indices);
    for (int idx : indices) {
        drawPred(classIds[idx], confidences[idx], boxes[idx].x, boxes[idx].y, boxes[idx].x + boxes[idx].width, boxes[idx].y + boxes[idx].height, resizedFrame, classes);
    }

    // Show the output image
    imshow("Detection", resizedFrame);
    waitKey(0); // Wait indefinitely until a key is pressed

    return 0;
}
