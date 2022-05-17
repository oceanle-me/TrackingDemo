#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/core/ocl.hpp>
#include <opencv2/tracking.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/model.h"
#include <cmath>

#include<thread>

#include <pigpio.h>

using namespace cv;
using namespace std;

const size_t width = 320;
const size_t height = 320;

Mat frame;
int m_x1, m_x2, m_y1, m_y2;

//bool tracking_is_running= false;
bool tracking_failure=false;
bool detect_success=false;
bool detect_stop_sign=false;
bool end_program=false;

std::vector<std::string> Labels;
std::unique_ptr<tflite::Interpreter> interpreter;
cv::VideoCapture cap("libcamerasrc ! video/x-raw,  width=(int)1280, height=(int)1280, framerate=(fraction)30/1 "
                     "! videoconvert ! videoscale ! video/x-raw, width=(int)320, height=(int)320 ! appsink", cv::CAP_GSTREAMER);

void InitPWM(void);
void control_motor(int x,int y);
int TrackingObj(void);
bool detect_from_video(Mat &src);
bool func_Detect_Stop_Sign(Mat &src);



void control_motor(int x,int y){ //input: x= (x1+x2)/2; y = (y1+y2)/2 -------300xx300
    //signal scaler in -+ 200:1000
    //cout <<x << "  ";
    if (x<160){
       unsigned int scaler = ((160-(float)x)/160)*185 + 70 ;
        //cout <<scaler << " \n ";
        gpioPWM(12,0);
        gpioPWM(13,scaler);
    } else {
       unsigned int scaler = (((float)x-160)/160)*185 + 70 ;
        //cout <<scaler << " \n ";
        gpioPWM(12,scaler);
        gpioPWM(13,0);
    }

}

static bool getFileContent(std::string fileName)
{

	// Open the File
	std::ifstream in(fileName.c_str());
	// Check if object is valid
	if(!in.is_open()) return false;

	std::string str;
	// Read the next line from File untill it reaches the end.
	while (std::getline(in, str))
	{
		// Line contains string of length > 0 then save it in vector
		if(str.size()>0) Labels.push_back(str);
	}
	// Close The File
	in.close();
	return true;
}

bool detect_from_video(Mat &src)
{
    Mat image;
    int cam_width =src.cols;
    int cam_height=src.rows;

    // copy image to input as input tensor
    cv::resize(src, image, Size(320,320));
    memcpy(interpreter->typed_input_tensor<uchar>(0), image.data, image.total() * image.elemSize());

    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(3);      //quad core
    interpreter->Invoke();      // run your model

    const float* detection_locations = interpreter->tensor(interpreter->outputs()[0])->data.f;
    const float* detection_classes=interpreter->tensor(interpreter->outputs()[1])->data.f;
    const float* detection_scores = interpreter->tensor(interpreter->outputs()[2])->data.f;
    const int    num_detections = *interpreter->tensor(interpreter->outputs()[3])->data.f;

    const float confidence_threshold = 0.6;

    float pre_x1 =160;
    float pre_x2 =160;
    float pre_y1 =160;
    float pre_y2 =160;
    float pre_score =0;

    for(int i = 0; i < num_detections; i++){
        if(detection_scores[i] > confidence_threshold){
            int  det_index = (int)detection_classes[i]+1;
            if (det_index==1){ //detecting person only
                float y1=detection_locations[4*i  ]*cam_height;
                float x1=detection_locations[4*i+1]*cam_width;
                float y2=detection_locations[4*i+2]*cam_height;
                float x2=detection_locations[4*i+3]*cam_width;

                //Rect:   x coordinate of the top-left corner, y coordinate of the top-left corner
                Rect rec((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));
                rectangle(src,rec, Scalar(0, 0, 255), 1, 8, 0);
                putText(src, format("%s", Labels[det_index].c_str()), Point(x1, y1-5) ,FONT_HERSHEY_SIMPLEX,0.5, Scalar(0, 0, 255), 1, 8, 0);

                if (detection_scores[i]>pre_score){
                    pre_score=detection_scores[i];
                    pre_x1 =x1;
                    pre_x2 =x2;
                    pre_y1 =y1;
                    pre_y2 =y2;
                }
            }
        }
    }
    if (pre_score>0){
        m_x1=pre_x1;
        m_x2=pre_x2;
        m_y1=pre_y1;
        m_y2=pre_y2;
        return true;
    } else return false;
}

int main(int argc,char ** argv)
{
    InitPWM();
    sleep(3);

    std::thread tracking_th(TrackingObj);
    cout <<"Creating tracking thread\n";

    //libcamerasrc ! video/x-raw,  width=(int)1280, height=(int)1280, framerate=(fraction)25/1 ! videoconvert ! videoscale ! video/x-raw, width=(int)300, height=(int)300 ! appsink
 //   gst-launch-1.0 libcamerasrc ! video/x-raw, width=1280, height=1280, framerate=30/1 ! videoconvert ! videoscale ! video/x-raw, width=300, height=300 ! clockoverlay time-format="%D %H:%M:%S" ! autovideosink

    std::cout << "Using pipeline: libcamerasrc ! video/x-raw,  width=(int)1280, height=(int)1280, framerate=(fraction)30/1 "
                     "! videoconvert ! videoscale ! video/x-raw, width=(int)320, height=(int)320 ! appsink\n\n";

    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("ssdlite_mobiledet_coco_qat_postprocess.tflite");

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
    interpreter->AllocateTensors();

	// Get the names
	bool result = getFileContent("COCO_labels.txt");
	if(!result){
        cout << "loading labels failed";
        exit(-1);
	}


    cout << "Start grabbing, press ESC on Live window to terminate" << endl;
	while(!end_program){
        cap >> frame;
        if (frame.empty()) {
            cerr << "ERROR: Unable to grab from the camera" << endl;
            break;
        }


        if (detect_from_video(frame)){
            cout <<"Detecting successfully \n";
            detect_success=true;
            tracking_failure=false;
            while(!tracking_failure){
                func_Detect_Stop_Sign(frame);
            }
        }

        imshow("Detector and Tracking", frame);
        char esc = waitKey(5);
        if(esc == 27)
            goto break_all;
    }

break_all:
    gpioPWM(12,0);
    gpioPWM(13,0);
    cout << "Closing the camera" << endl;
    destroyAllWindows();
    cout << "Bye!" << endl;
    gpioTerminate();
  return 0;
}

void InitPWM(void){
    gpioInitialise();

    gpioSetMode(12, PI_ALT0);
    gpioSetPullUpDown(12, PI_PUD_DOWN);
    gpioSetPWMfrequency(12,10000);
    gpioPWM(12,0);

    gpioSetMode(13, PI_ALT0);
    gpioSetPullUpDown(13, PI_PUD_DOWN);
    gpioSetPWMfrequency(13,10000);
    gpioPWM(13,0);
 //   gpioSetMode(21,PI_OUTPUT);
}

int TrackingObj(void){
    while(1){//always running until this function returning
        wait_detect:
        if(detect_success){
            detect_success=false;
            tracking_failure=false;
            cout <<"Tracking is running\n";

            // Define initial bounding box
            Rect bbox((int)m_x1, (int)m_y1, (int)(m_x2 - m_x1), (int)(m_y2 - m_y1));
            //Resizing box if it is too big
            /*if(bbox.width>70){
                bbox.x=bbox.x + bbox.width/2 - 35;
                bbox.width=70;
            }
            if(bbox.height>150){
                bbox.y=bbox.y+10;
                bbox.height=150;
            }*/
            if(bbox.width>70){
                bbox.x=bbox.x + bbox.width/2 - 35;
                bbox.width=70;
            }
            if(bbox.height>150){
                bbox.y=bbox.y+10;
                bbox.height=150;
            }
            Ptr<Tracker> tracker;
            tracker = TrackerKCF::create();
            //tracker = TrackerCSRT::create();
            tracker->init(frame, bbox);
            do
            {
                uint64_t timer = cv::getTickCount();
                // Update the tracking result
                bool ok = tracker->update(frame, bbox);

                if (ok){
                    // Tracking success : Draw the tracked object
                    rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
                } else { // Tracking failure detected.
                    tracking_failure=true;
                    cout << "Tracking failure detected \n";
                    gpioPWM(12,0);
                    gpioPWM(13,0);
                    goto wait_detect;
                }

                float fps = cv::getTickFrequency() / (float)(cv::getTickCount() - timer);
                putText(frame, format("FPS %0.2f",fps),Point(10,20),FONT_HERSHEY_SIMPLEX,0.6, Scalar(0, 0, 255));
                // Display frame.
                imshow("Detector and Tracking", frame);

                control_motor(bbox.x+bbox.width/2,0);

                // Exit if ESC pressed.
                int k = waitKey(5);
                if(k == 27)
                {
                    cout <<"ESC - out from tracking\n";
                    tracking_failure=true;
                    end_program=true;
                    gpioPWM(12,0);
                    gpioPWM(13,0);
                    return -1;
                }

            } while(cap.read(frame));
            tracking_failure=true;
            end_program=true;
            gpioPWM(12,0);
            gpioPWM(13,0);
            cout <<"Can't not grab the next frame, out from tracking\n";
            return 1;
        }
    }

}

bool func_Detect_Stop_Sign(Mat &src)
{
    uint64_t timer = cv::getTickCount();

    Mat image;
   /* int cam_width =src.cols;
    int cam_height=src.rows;*/

    // copy image to input as input tensor
    cv::resize(src, image, Size(320,320));
    memcpy(interpreter->typed_input_tensor<uchar>(0), image.data, image.total() * image.elemSize());

    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(2);      //quad core
    interpreter->Invoke();      // run your model

    //const float* detection_locations = interpreter->tensor(interpreter->outputs()[0])->data.f;
    const float* detection_classes=interpreter->tensor(interpreter->outputs()[1])->data.f;
    const float* detection_scores = interpreter->tensor(interpreter->outputs()[2])->data.f;
    const int    num_detections = *interpreter->tensor(interpreter->outputs()[3])->data.f;

    const float confidence_threshold = 0.63;

   /* float pre_x1 =150;
    float pre_x2 =150;
    float pre_y1 =150;
    float pre_y2 =150;
    float pre_score =0;*/

    for(int i = 0; i < num_detections; i++){
        if(detection_scores[i] > confidence_threshold){
            int  det_index = (int)detection_classes[i]+1;
            if (det_index==1){ //detecting person only
            /*    float y1=detection_locations[4*i  ]*cam_height;
                float x1=detection_locations[4*i+1]*cam_width;
                float y2=detection_locations[4*i+2]*cam_height;
                float x2=detection_locations[4*i+3]*cam_width;*/
                cout<< "detect a human\n";
                }
            }
    }
    float fps = cv::getTickFrequency() / (float)(cv::getTickCount() - timer);
    cout << "From detecting signs FPS="<<fps<< "\n";
    return 0;
}


