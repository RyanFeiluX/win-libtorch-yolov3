#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Darknet.h"

using namespace std; 
using namespace std::chrono; 

namespace fs = std::filesystem;

int main(int argc, const char* argv[])
{
    if (argc != 2) {
        std::cerr << "usage: yolo-app <image path>\n";
        return -1;
    }

    if (!fs::exists(argv[1]))
    {
        std::cerr << argv[1] << " does not exist";
        return -2;
    }

    fs::path root_path;
    std::vector<string> list_img;
    if (fs::is_directory(argv[1]))
    {
        root_path = fs::absolute(fs::path(argv[1])).parent_path();
        for (const auto & entry : fs::directory_iterator(fs::path(argv[1])))
        {
            if (entry.is_regular_file())
            {
                list_img.push_back(entry.path().string());
            }
        }
    }
    else
    {
        root_path = fs::absolute(fs::path(argv[1])).parent_path().parent_path();
        list_img.push_back(argv[1]);
    }
    
    torch::DeviceType device_type;

    if (torch::cuda::is_available() ) {        
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);

    // input image size for YOLO v3
    int input_image_size = 416;

    string cfg_path = root_path.string() + "/models/yolov3.cfg";
    Darknet net((const char *)cfg_path.c_str(), &device);

    map<string, string> *info = net.get_net_info();

    info->operator[]("height") = std::to_string(input_image_size);

    std::cout << "loading weight ..." << endl;
    string weight_path = root_path.string() + "/models/yolov3.weights";
    net.load_weights((const char *)weight_path.c_str());
    std::cout << "weight loaded ..." << endl;
    
    std::cout << "loading class-names ..." << endl;
    string class_path = root_path.string() + "/data/coco.names";
    ifstream infile(class_path);
    char buf[40];  // Assumption: class name not more than 40 characters.
    vector<string> list_class;
    while (!infile.eof()){
        infile.getline(buf, sizeof(buf));
        list_class.push_back(buf);
    }
    std::cout << "class-names loaded ..." << endl;
    
    net.to(device);

    torch::NoGradGuard no_grad;
    net.eval();

    std::cout << "start to inference ..." << endl;
    
    cv::Mat origin_image, resized_image;

    for (const auto & img : list_img)
    {
        // origin_image = cv::imread("../139.jpg");
        origin_image = cv::imread(img);
        
        cv::cvtColor(origin_image, resized_image,  cv::COLOR_BGR2RGB);
        cv::resize(resized_image, resized_image, cv::Size(input_image_size, input_image_size));

        cv::Mat img_float;
        resized_image.convertTo(img_float, CV_32F, 1.0/255);

        auto img_tensor = torch::from_blob(img_float.data, {1, input_image_size, input_image_size, 3}).to(device);
        img_tensor = img_tensor.permute({0,3,1,2});

        auto start = std::chrono::high_resolution_clock::now();
        
        auto output = net.forward(img_tensor);

        // prediction: batch size * bboxes of all anchors & all scales      * (bbox coord, confidence, classes)
        //             1          * 10647                                   * 85
        // detail:     batch size * anchors of 3 scales * bboxes per anchor * (bbox coord, confidence, classes)
        //             1          * (13*13+26*26+52*52) * 3                 * (4+1+80)
        
        // filter result by NMS 
        // class_num = 80
        // confidence = 0.6
        auto result = net.write_results(output, 80, 0.6, 0.4);

        auto end = std::chrono::high_resolution_clock::now();

        auto duration = duration_cast<milliseconds>(end - start); 

        // It should be known that it takes longer time at first time
        std::cout << fs::path(img).filename().string() << " => " << "inference taken : " << duration.count() << " ms" << endl; 

        if (result.dim() == 1)
        {
            std::cout << "no object found" << endl;
        }
        else
        {
            int obj_num = result.size(0);
            int vec_size = result.size(1);

            auto result_data = result.accessor<float, 2>();

            vector<string> list_label;
            list_label.push_back(list_class[(int)result_data[0][vec_size-1]]);
            string objects(list_label[0]);
            for (int i = 1; i < obj_num; i++)
            {
                list_label.push_back(list_class[(int)result_data[i][vec_size-1]]);
                objects.append(",").append(list_label[i]);
            }

            std::cout << obj_num << " objects found: " << objects << endl;

            float w_scale = float(origin_image.cols) / input_image_size;
            float h_scale = float(origin_image.rows) / input_image_size;

            result.select(1,1).mul_(w_scale);
            result.select(1,2).mul_(h_scale);
            result.select(1,3).mul_(w_scale);
            result.select(1,4).mul_(h_scale);

            cv::Size t_size, t_margin;
            cv::Point bboxp1, bboxp2, tboxp1, tboxp2;
            for (int i = 0; i < obj_num ; i++)
            {
                bboxp1 = cv::Point(result_data[i][1], result_data[i][2]);
                bboxp2 = cv::Point(result_data[i][3], result_data[i][4]);
                cv::rectangle(origin_image, bboxp1, bboxp2, cv::Scalar(0, 0, 255), 1, 1, 0);
                t_size = cv::getTextSize(list_label[i], cv::FONT_HERSHEY_PLAIN, 1, 1, 0);
                tboxp1 = bboxp1;
                t_margin = cv::Size(6, 4);
                tboxp2 = cv::Point(bboxp1.x + t_size.width + t_margin.width * 2, 
                                    bboxp1.y - t_size.height - t_margin.height * 2);
                cv::rectangle(origin_image, tboxp1, tboxp2, cv::Scalar(255, 0, 0), cv::FILLED);
                cv::putText(origin_image, list_label[i], 
                            cv::Point(tboxp1.x + t_margin.width, tboxp1.y - t_margin.height), 
                            cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(225, 255, 255), 1);
            }

            if (!fs::exists(root_path.string() + "/det"))
            {
                cout << "Create folder " << root_path.string() + "/det" << endl;
                fs::create_directory(root_path.string() + "/det");
            }
            cv::imwrite(root_path.string() + "/det/det_" + fs::path(img).filename().string(), origin_image);
        }
    }

    std::cout << "Done" << endl;
    
    return 0;
}
