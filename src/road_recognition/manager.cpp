#include "manager.hpp"
#include "constant.hpp"
#include <filesystem>
#include <fstream>

Manager::Manager(Reader &reader, Writer &writer)
    : reader_(reader), writer_(writer) {}

DistantRoadRecognitionManager::DistantRoadRecognitionManager(
    Reader &reader, Writer &writer, DistantRoadRecognition &marker)
    : Manager(reader, writer), marker_(marker) {
  cv::Mat sample = reader_.GetSample();
  marker_.SetRoi(sample);
}

DistantRoadRecognitionManager::~DistantRoadRecognitionManager() {
  pixels_vectors_.clear();
}

void DistantRoadRecognitionManager::Process() {
  for (size_t i = 0; i < reader_.GetSize(); i++) {
    cv::Mat img = reader_.Read();
    Points lane_pixels = *(marker_.MarkLaneAtDistance(img));
    writer_.Write(img);
    pixels_vectors_.push_back(lane_pixels);
  }
}
std::vector<Points> DistantRoadRecognitionManager::GetPixelsVectors() {
  return pixels_vectors_;
}

UpscaleManager::UpscaleManager(Reader &reader, Writer &writer,
                               Upscale &improver)
    : Manager(reader, writer), improver_(improver) {}

void UpscaleManager::Process() {
  for (size_t i = 0; i < reader_.GetSize(); i++) {
    cv::Mat img = reader_.Read();
    improver_.Execute(img);
    writer_.Write(img);
  }
}

RoiManager::RoiManager(Reader &reader, Writer &writer)
    : Manager(reader, writer), roi_() {}

void RoiManager::SetRoi(const cv::Mat &img) {
  cv::Rect roi;
  roi.x = (img.size().width * 7) / 18;
  roi.y = (img.size().height * 5) / 9;
  int remain_width = img.size().width - roi.x;
  int remain_height = img.size().height - roi.y;
  int correction = 1;
  if (remain_width < Constant::default_ROI_width ||
      remain_height < Constant::default_ROI_height) {
    int correction_width =
        (Constant::default_ROI_width + remain_width - 1) / remain_width;
    int correction_height =
        (Constant::default_ROI_height + remain_height - 1) / remain_height;
    correction = (correction_width > correction_height) ? correction_width
                                                        : correction_height;
  }
  roi.width = Constant::default_ROI_width / correction;
  roi.height = Constant::default_ROI_height / correction;
  roi_ = roi;
}

void RoiManager::Process() {
  if (roi_.empty()) {
    SetRoi(reader_.GetSample());
  }
  for (size_t i = 0; i < reader_.GetSize(); i++) {
    cv::Mat img = reader_.Read();
    img = img(roi_);
    writer_.Write(img);
  }
}

FPSManagerDistantRoadRecognition::FPSManagerDistantRoadRecognition(
    DistantRoadRecognition &marker, const std::string &path_to_txt)
    : marker_(marker), path_to_txt_(path_to_txt) {}

void FPSManagerDistantRoadRecognition::Process() {
  std::ofstream file(path_to_txt_);
  file << "FPS:";
  double fps_sum = 0;
  double fps_sq_sum = 0;
  int num_of_images = 0;
  std::vector<std::string> st_nums;
  for (int i = 1; i <= 7; i++) {
    std::string st_num = "";
    st_num += '0' + (i / 100);
    st_num += '0' + ((i % 100) / 10);
    st_num += '0' + (i % 10);
    st_nums.push_back(st_num);
  }
  for (int i = 15; i <= 48; i++) {
    if (i == 17 || i == 32) {
      continue;
    }
    std::string st_num = "";
    st_num += '0' + (i / 100);
    st_num += '0' + ((i % 100) / 10);
    st_num += '0' + (i % 10);
    st_nums.push_back(st_num);
  }
  std::string init_path = "/home/artem/Загрузки/ColorImage_road02/ColorImage/"
                          "Record001/Camera 5/170927_063811892_Camera_5.jpg";
  cv::Mat init_image = cv::imread(init_path);
  cv::Rect ROI = SetRoiRect(init_image);
  for (std::string str : st_nums) {
    std::string in_path_marked =
        "/home/artem/Загрузки/ColorImage_road02/ColorImage/Record" + str +
        "/Camera 5/";
    std::string in_path_gt = "/home/artem/Загрузки/Labels_road02/Label/Record" +
                             str + "/Camera 5 filtered/";
    std::string out_path_marked =
        "/home/artem/practice_4_sem/marked_images_DRR_Upscale/Record" + str +
        "/Camera 5/";
    std::string out_path_gt =
        "/home/artem/practice_4_sem/gt_images_DRR_Upscale/Record" + str +
        "/Camera 5/";
    FlowHandler handler(in_path_marked, PhotoExtension::jpg, 15, 4);
    FolderReader reader_gt(in_path_gt, PhotoExtension::png);
    cv::Mat img_gt = reader_gt.Read();
    FolderWriter writer_marked(out_path_marked, PhotoExtension::jpg);
    FolderWriter writer_gt(out_path_gt, PhotoExtension::jpg);
    handler.SetRoi(ROI);
    num_of_images += handler.GetSize();
    do {
      cv::Mat img = handler.GetCurrentFrame();
      cv::Rect ROI_rect = handler.GetRoi();
      cv::Mat ROI_img_marked = img(ROI_rect);
      cv::Mat ROI_img_gt = img_gt(ROI_rect);
      auto start = std::chrono::high_resolution_clock::now();
      marker_.MarkLaneAtDistance(ROI_img_marked);
      auto end = std::chrono::high_resolution_clock::now();
      writer_marked.Write(ROI_img_marked);
      writer_gt.Write(ROI_img_gt);
      auto duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      double fps = 1000.0 / static_cast<double>(duration.count());
      fps_sum += fps;
      fps_sq_sum += (fps * fps);
      file << " " << fps;
      img_gt = reader_gt.Read();
    } while (handler.Next());
  }
  file << std::endl;
  if (num_of_images > 0) {
    double expectation = fps_sum / static_cast<double>(num_of_images);
    double variance = (fps_sq_sum / static_cast<double>(num_of_images)) -
                      (expectation * expectation);
    file << "Num of images: " << num_of_images << std::endl;
    file << "Expectation: " << expectation << std::endl;
    file << "Variance: " << variance << std::endl;
  } else {
    file << "No images processed." << std::endl;
  }
  file.close();
}

MetricsManager::MetricsManager(std::string path_to_txt)
    : path_to_txt_(path_to_txt),
      res_IoU_(Constant::init_incorrect_metric_value),
      res_accuracy_(Constant::init_incorrect_metric_value) {}

double MetricsManager::EvaluateIoU(const cv::Mat &marking_res,
                                   const cv::Mat &ground_truth) {
  if (marking_res.rows != Constant::default_ROI_height ||
      ground_truth.rows != Constant::default_ROI_height ||
      marking_res.cols != Constant::default_ROI_width ||
      ground_truth.cols != Constant::default_ROI_width) {
    throw std::invalid_argument("Error: Incorrect image size");
  }
  int num_of_intersection = 0;
  int num_of_union = 0;
  for (int y = 0; y < marking_res.rows; y++) {
    for (int x = 0; x < marking_res.cols; x++) {
      bool is_marking_res_pixel_black =
          IsBlackPixel(marking_res.at<cv::Vec3b>(y, x));
      bool is_ground_truth_pixel_black =
          IsBlackPixel(ground_truth.at<cv::Vec3b>(y, x));
      if (!is_marking_res_pixel_black || !is_ground_truth_pixel_black) {
        num_of_union++;
      }
      if (!is_marking_res_pixel_black && !is_ground_truth_pixel_black) {
        num_of_intersection++;
      }
    }
  }
  if (num_of_union == 0) {
    return 1;
  }
  return static_cast<double>(num_of_intersection) /
         static_cast<double>(num_of_union);
}

double MetricsManager::EvaluateAccuracy(const cv::Mat &marking_res,
                                        const cv::Mat &ground_truth) {
  if (marking_res.rows != Constant::default_ROI_height ||
      ground_truth.rows != Constant::default_ROI_height ||
      marking_res.cols != Constant::default_ROI_width ||
      ground_truth.cols != Constant::default_ROI_width) {
    throw std::invalid_argument("Error: Incorrect image size");
  }
  int num_true_positive = 0;
  int num_true_negative = 0;
  int num_false_positive = 0;
  int num_false_negative = 0;
  for (int y = 0; y < marking_res.rows; y++) {
    for (int x = 0; x < marking_res.cols; x++) {
      bool is_marking_res_pixel_black =
          IsBlackPixel(marking_res.at<cv::Vec3b>(y, x));
      bool is_ground_truth_pixel_black =
          IsBlackPixel(ground_truth.at<cv::Vec3b>(y, x));
      if (is_marking_res_pixel_black) {
        if (is_ground_truth_pixel_black) {
          num_true_negative++;
        } else {
          num_false_negative++;
        }
      } else {
        if (!is_ground_truth_pixel_black) {
          num_true_positive++;
        } else {
          num_false_positive++;
        }
      }
    }
  }
  return static_cast<double>(num_true_positive + num_true_negative) /
         static_cast<double>(num_true_positive + num_true_negative +
                             num_false_positive + num_false_negative);
}

void MetricsManager::Process() {
  std::ofstream file(path_to_txt_);
  file << "(IoU, Accuracy):";

  double sum_IoU = 0;
  double sum_sq_IoU = 0;
  double sum_accuracy = 0;
  double sum_sq_accuracy = 0;
  int num_of_images = 0;

  std::vector<std::string> st_nums;
  for (int i = 1; i <= 7; i++) {
    std::string st_num = "";
    st_num += '0' + (i / 100);
    st_num += '0' + ((i % 100) / 10);
    st_num += '0' + (i % 10);
    st_nums.push_back(st_num);
  }
  for (int i = 15; i <= 48; i++) {
    if (i == 17 || i == 32) {
      continue;
    }
    std::string st_num = "";
    st_num += '0' + (i / 100);
    st_num += '0' + ((i % 100) / 10);
    st_num += '0' + (i % 10);
    st_nums.push_back(st_num);
  }
  for (std::string str : st_nums) {
    std::string marking_res_path =
        "/home/artem/practice_4_sem/marked_images_TLN/Record" + str +
        "/Camera 5/";
    std::string ground_truth_path =
        "/home/artem/practice_4_sem/gt_images_TLN/Record" + str + "/Camera 5/";
    FolderReader marking_res(marking_res_path, PhotoExtension::jpg);
    FolderReader ground_truth(ground_truth_path, PhotoExtension::jpg);
    int n_of_bad = 0;
    std::vector<double> iou_values;
    num_of_images += marking_res.GetSize();
    file << "Record" << str;
    for (int i = 0; i < marking_res.GetSize(); i++) {
      cv::Mat marking_res_img = marking_res.Read();
      cv::Mat ground_truth_img = ground_truth.Read();
      cv::imshow("Marked image", marking_res_img);
      cv::imshow("Ground truth image", ground_truth_img);
      cv::waitKey(15);
      double IoU = EvaluateIoU(marking_res_img, ground_truth_img) * 100;
      iou_values.push_back(IoU);
      if (IoU == 0) {
        n_of_bad++;
      }
      double accuracy =
          EvaluateAccuracy(marking_res_img, ground_truth_img) * 100;
      sum_IoU += IoU;
      sum_sq_IoU += (IoU * IoU);
      sum_accuracy += accuracy;
      sum_sq_accuracy += (accuracy * accuracy);
      file << " (" << IoU << ", " << accuracy << ")";
    }
    std::cout << "File " << str << std::endl;
    PrintHistogram(iou_values, 50);
    file << std::endl << "Num of bad images: " << n_of_bad;
    file << " share of bad: "
         << (static_cast<double>(n_of_bad) /
             static_cast<double>(marking_res.GetSize()))
         << std::endl;
  }
  file << std::endl;
  if (num_of_images > 0) {
    double expectation_IoU = sum_IoU / static_cast<double>(num_of_images);
    double variance_IoU = (sum_sq_IoU / static_cast<double>(num_of_images)) -
                          (expectation_IoU * expectation_IoU);

    double expectation_accuracy =
        sum_accuracy / static_cast<double>(num_of_images);
    double variance_accuracy =
        (sum_sq_accuracy / static_cast<double>(num_of_images)) -
        (expectation_accuracy * expectation_accuracy);

    file << "Num of images: " << num_of_images << std::endl;
    file << "IoU expectation: " << expectation_IoU << std::endl;
    file << "IoU variance: " << variance_IoU << std::endl;
    file << "Accuracy expectation: " << expectation_accuracy << std::endl;
    file << "Accuracy variance: " << variance_accuracy << std::endl;
  } else {
    file << "No images processed." << std::endl;
  }
  file.close();
}

void MetricsManager::PrintHistogram(const std::vector<double> &data,
                                    int num_bins) {
  int min_value = 0;
  int max_value = 100;
  std::vector<int> histogram(num_bins, 0);
  double step = static_cast<double>(max_value - min_value) /
                static_cast<double>(num_bins);
  for (double val : data) {
    int index = static_cast<int>((val - min_value) / step);
    if (index >= num_bins) {
      index = num_bins - 1;
    }
    histogram[index]++;
  }
  for (int i = 0; i < num_bins; i++) {
    int left = static_cast<int>(i * step);
    int right = static_cast<int>((i + 1) * step);
    std::cout << left / 100 << (left % 100) / 10 << left % 10 << "% - ";
    std::cout << right / 100 << (right % 100) / 10 << right % 10 << "% | ";
    for (int j = 0; j < histogram[i]; j++) {
      std::cout << "=";
    }
    std::cout << " " << histogram[i] << std::endl;
  }
}

FPSManagerTwinLiteNet::FPSManagerTwinLiteNet(const std::string &path_to_txt)
    : path_to_txt_(path_to_txt) {}

void FPSManagerTwinLiteNet::Process() {
  std::ofstream file(path_to_txt_);
  file << "FPS:";
  double fps_sum = 0;
  double fps_sq_sum = 0;
  int num_of_images = 0;
  std::vector<std::string> st_nums;
  for (int i = 1; i <= 7; i++) {
    std::string st_num = "";
    st_num += '0' + (i / 100);
    st_num += '0' + ((i % 100) / 10);
    st_num += '0' + (i % 10);
    st_nums.push_back(st_num);
  }
  for (int i = 15; i <= 48; i++) {
    if (i == 17 || i == 32) {
      continue;
    }
    std::string st_num = "";
    st_num += '0' + (i / 100);
    st_num += '0' + ((i % 100) / 10);
    st_num += '0' + (i % 10);
    st_nums.push_back(st_num);
  }
  std::string init_path = "/home/artem/Загрузки/ColorImage_road02/ColorImage/"
                          "Record001/Camera 5/170927_063811892_Camera_5.jpg";
  cv::Mat init_image = cv::imread(init_path);
  cv::Rect ROI = SetRoiRect(init_image);
  TwinLiteNet model(
      "/home/artem/practice_4_sem/RoadDirectionRecognition_AtDistance/"
      "TwinLiteNet-onnxruntime/models/best.onnx");
  for (std::string str : st_nums) {
    std::string in_path_marked =
        "/home/artem/Загрузки/ColorImage_road02/ColorImage/Record" + str +
        "/Camera 5/";
    std::string in_path_gt = "/home/artem/Загрузки/Labels_road02/Label/Record" +
                             str + "/Camera 5 filtered/";
    std::string out_path_marked =
        "/home/artem/practice_4_sem/marked_images_TLN/Record" + str +
        "/Camera 5/";
    std::string out_path_gt =
        "/home/artem/practice_4_sem/gt_images_TLN/Record" + str + "/Camera 5/";
    FlowHandler handler(in_path_marked, PhotoExtension::jpg, 15, 4);
    FolderReader reader_gt(in_path_gt, PhotoExtension::png);
    cv::Mat img_gt = reader_gt.Read();
    FolderWriter writer_marked(out_path_marked, PhotoExtension::jpg);
    FolderWriter writer_gt(out_path_gt, PhotoExtension::jpg);
    handler.SetRoi(ROI);
    num_of_images += handler.GetSize();
    do {
      cv::Mat img = handler.GetCurrentFrame();
      cv::resize(
          img, img,
          cv::Size(Constant::default_ROI_width, Constant::default_ROI_height));
      auto start = std::chrono::high_resolution_clock::now();
      cv::Mat da_out, ll_out;
      model.Infer(img, da_out, ll_out);
      img.setTo(cv::Scalar(0, 0, 0));
      img.setTo(cv::Scalar(180, 130, 70), ll_out);
      auto end = std::chrono::high_resolution_clock::now();
      cv::resize(
          img, img,
          cv::Size(Constant::appolo_img_width, Constant::appolo_img_height));
      cv::Rect ROI_rect = handler.GetRoi();
      cv::Mat ROI_img_marked = img(ROI_rect);
      cv::Mat ROI_img_gt = img_gt(ROI_rect);
      writer_marked.Write(ROI_img_marked);
      writer_gt.Write(ROI_img_gt);
      cv::imshow("ROI image marked", ROI_img_marked);
      cv::imshow("ROI image ground truth", ROI_img_gt);
      cv::waitKey(15);
      auto duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      double fps = 1000.0 / static_cast<double>(duration.count());
      fps_sum += fps;
      fps_sq_sum += (fps * fps);
      file << " " << fps;
      img_gt = reader_gt.Read();
    } while (handler.Next());
  }
  file << std::endl;
  if (num_of_images > 0) {
    double expectation = fps_sum / static_cast<double>(num_of_images);
    double variance = (fps_sq_sum / static_cast<double>(num_of_images)) -
                      (expectation * expectation);
    file << "Num of images: " << num_of_images << std::endl;
    file << "Expectation: " << expectation << std::endl;
    file << "Variance: " << variance << std::endl;
  } else {
    file << "No images processed." << std::endl;
  }
  file.close();
}

AutomatedRoiManagerDistantRoadRecognition::
    AutomatedRoiManagerDistantRoadRecognition(DistantRoadRecognition &marker)
    : marker_(marker) {}

double AutomatedRoiManagerDistantRoadRecognition::Process() {
  std::vector<std::string> st_nums;
  for (int i = 1; i <= 7; i++) {
    std::string st_num = "";
    st_num += '0' + (i / 100);
    st_num += '0' + ((i % 100) / 10);
    st_num += '0' + (i % 10);
    st_nums.push_back(st_num);
  }
  for (int i = 15; i <= 48; i++) {
    if (i == 17 || i == 32) {
      continue;
    }
    std::string st_num = "";
    st_num += '0' + (i / 100);
    st_num += '0' + ((i % 100) / 10);
    st_num += '0' + (i % 10);
    st_nums.push_back(st_num);
  }
  std::string init_path = "/home/artem/Загрузки/ColorImage_road02/ColorImage/"
                          "Record001/Camera 5/170927_063811892_Camera_5.jpg";
  cv::Mat init_image = cv::imread(init_path);
  cv::Rect ROI = SetRoiRect(init_image);
  size_t num_of_images = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for (std::string str : st_nums) {
    std::string in_path_marked =
        "/home/artem/Загрузки/ColorImage_road02/ColorImage/Record" + str +
        "/Camera 5/";
    FlowHandler handler(in_path_marked, PhotoExtension::jpg, 15, 4);
    handler.SetRoi(ROI);
    num_of_images += handler.GetSize();
    do {
      cv::Mat img = handler.GetCurrentFrame();
      cv::Rect ROI_rect = handler.GetRoi();
      cv::Mat ROI_img_marked = img(ROI_rect);
      marker_.MarkLaneAtDistance(ROI_img_marked);
    } while (handler.Next());
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  double fps = 1000.0 * static_cast<double>(num_of_images) /
               static_cast<double>(duration.count());
  return fps;
}