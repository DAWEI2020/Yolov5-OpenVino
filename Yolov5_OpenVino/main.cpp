#include "detector.h"

int main(int argc, char const *argv[])
{
    Detector* detector = new Detector;
    string xml_path = "yolov5s.xml";
    string bin_path = "yolov5s.bin";
    detector->init(xml_path, bin_path, 0.25, 0.45);  // init

    Mat src = imread("test.jpg");
    Mat src2 = src.clone();
    int width = src.cols;
    int height = src.rows;
    int channel = src.channels();
    double scale = min(640.0 / width, 640.0 / height);
    int w = round(width * scale);
    int h = round(height * scale);
    cout << "w: " << w << endl;
    cout << "h: " << h << endl;
    Mat src3;
    resize(src2, src3, Size(w, h));
    int top = 0, bottom = 0, left = 0, right = 0;
    if (w > h)
    {
        top = (w - h) / 2;
        bottom = (w - h) - top;
    }
    else if (h > w)
    {
        left = (h - w) / 2;
        right = (h - w) - left;
    }
    copyMakeBorder(src3, src3, top, bottom, left, right, BORDER_CONSTANT, Scalar(114,114,114));
    
    vector<Detector::Object> detected_objects;
    auto start = chrono::high_resolution_clock::now();
    detector->process_frame(src3, detected_objects);  // infer
    auto end = chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    cout<< "use " << diff.count() <<" s" << endl;

    for(int i=0; i<detected_objects.size(); ++i){
        int xmin = max(detected_objects[i].rect.x - left, 0);
        int ymin = max(detected_objects[i].rect.y - top, 0);
        int width = detected_objects[i].rect.width;
        int height = detected_objects[i].rect.height;
        Rect rect(int(xmin / scale), int(ymin / scale), int(width / scale), int(height / scale));
        cv::rectangle(src2, rect, Scalar(0, 0, 255), 1, LINE_8, 0);
    }
    imwrite("result.jpg", src2);

    cout << "Hello World!" << endl;
    waitKey(0);
    return 0;
}
