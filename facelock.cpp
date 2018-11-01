////// This example merges the below 4 dlib examples
////// 1) dnn_face_recognition_ex.cpp
////// 2) image_ex.cpp
////// 3) webcam_face_pose_ex.cpp
////// 4) face_landmark_detection_ex.cpp

//// dnn_face_recognition_ex.cpp
//// http://dlib.net/dnn_face_recognition_ex.cpp.html
// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the dlib C++
    Library.  In it, we will show how to do face recognition.  This example uses the
    pretrained dlib_face_recognition_resnet_model_v1 model which is freely available from
    the dlib web site.  This model has a 99.38% accuracy on the standard LFW face
    recognition benchmark, which is comparable to other state-of-the-art methods for face
    recognition as of February 2017.

    In this example, we will use dlib to do face clustering.  Included in the examples
    folder is an image, bald_guys.jpg, which contains a bunch of photos of action movie
    stars Vin Diesel, The Rock, Jason Statham, and Bruce Willis.   We will use dlib to
    automatically find their faces in the image and then to automatically determine how
    many people there are (4 in this case) as well as which faces belong to each person.

    Finally, this example uses a network with the loss_metric loss.  Therefore, if you want
    to learn how to train your own models, or to get a general introduction to this loss
    layer, you should read the dnn_metric_learning_ex.cpp and
    dnn_metric_learning_on_images_ex.cpp examples.
*/

//// image_ex.cpp
//// http://dlib.net/image_ex.cpp.html
// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the GUI API as well as some
    aspects of image manipulation from the dlib C++ Library.


    This is a pretty simple example.  It takes a BMP file on the command line
    and opens it up, runs a simple edge detection algorithm on it, and
    displays the results on the screen.
*/

//// webcam_face_pose_ex.cpp
//// http://dlib.net/webcam_face_pose_ex.cpp.html
// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.

    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use OpenCV's VideoCapture object to read from a camera instead
    of files.

    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.
*/

//// face_landmark_detection_ex.cpp
//// http://dlib.net/face_landmark_detection_ex.cpp.html
// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.

    The face detector we use is made using the classic Histogram of Oriented
    Gradients (HOG) feature combined with a linear classifier, an image pyramid,
    and sliding window detection scheme.  The pose estimator was created by
    using dlib's implementation of the paper:
       One Millisecond Face Alignment with an Ensemble of Regression Trees by
       Vahid Kazemi and Josephine Sullivan, CVPR 2014
    and was trained on the iBUG 300-W face landmark dataset (see
    https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
       C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
       300 faces In-the-wild challenge: Database and results.
       Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
    You can get the trained model file from:
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.
    Note that the license for the iBUG 300-W dataset excludes commercial use.
    So you should contact Imperial College London to find out if it's OK for
    you to use this model file in a commercial product.

    Also, note that you can train your own models using dlib's machine learning
    tools.  See train_shape_predictor_ex.cpp to see an example.

    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.
*/

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/gui_widgets.h>

#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_transforms.h>

#include <dlib/dnn.h>
#include <dlib/clustering.h>
#include <dlib/string.h>

#include <fstream>

//#include <opencv2/core.hpp>
//#include <opencv/cv.hpp>

using namespace dlib;
using namespace std;
//using namespace cv;

std::vector<cv::Point> righteye;
std::vector<cv::Point> lefteye;
std::vector<cv::Point> mouth;

cv::Point p;

// ----------------------------------------------------------------------------------------

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, and the training dataset consisted of about 3 million images instead of
// 55.  Also, the input layer was locked to images of size 150.
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
);

double compute_EAR(std::vector<cv::Point> vec) {
   try {
       double a = cv::norm(cv::Mat(vec[1]), cv::Mat(vec[5]));
       double b = cv::norm(cv::Mat(vec[2]), cv::Mat(vec[4]));
       double c = cv::norm(cv::Mat(vec[0]), cv::Mat(vec[3]));
       if (c > 0) {
           return MIN(a, b)/c;
       }

       return 0.0;
   } catch (...) {
       return 0.0;
   }
}

double compute_MAR(std::vector<cv::Point> vec) {
   try {
       double a = cv::norm(cv::Mat(vec[2]), cv::Mat(vec[10]));
       double b = cv::norm(cv::Mat(vec[4]), cv::Mat(vec[8]));
       double c = cv::norm(cv::Mat(vec[0]), cv::Mat(vec[6]));
       if (c > 0) {
           return MIN(a, b)/c;
       }

       return 0.0;
   } catch (...) {
       return 0.0;
   }
}

// ----------------------------------------------------------------------------------------

//int main(int argc, char** argv)
//int facelock(std::string& imagepath)
int facelock()
{
    try
    {
        //if (argc != 2)
        /*
        if (imagepath.empty())
        {
           cout << "Run this example by invoking it like this: " << endl;
           cout << "   ./videocapture_face_pose_edges_ex /dev/video0" << endl;
           cout << endl;
           cout << "with video device OR with single frame as below" << endl;
           cout << endl;
           cout << "   ./videocapture_face_pose_edges_ex faces/bald_guys.jpg" << endl;
           cout << endl;
           cout << "You will also need to get the face landmarking model file as well as " << endl;
           cout << "the face recognition model file.  Download and then decompress these files from: " << endl;
           cout << "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2" << endl;
           cout << "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2" << endl;
           cout << endl;
           return 1;
        }
        */

        // Create a VideoCapture object and open the input file
        // If the input is taken from the camera, pass 0 instead of the video file name
        //cv::VideoCapture cap(0);
        //cv::VideoCapture cap("/home/csn/examples/datasets/videos/openauto/video1.mov");
        cv::VideoCapture cap("/dev/video0");
        //cv::VideoCapture cap(imagepath);
        //cv::VideoCapture cap(argv[1]);

        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        image_window win;
        image_window win_fbl;
        image_window win_nei;
        image_window win_hot;
        image_window win_jet;
        image_window win_org;
        image_window win_faces_matrix;

        // The first thing we are going to do is load all our models.

        // First, since we need to
        // find faces in the image we will need a face detector:
        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();

        // We will also use a face landmarking model to align faces to a standard pose:
        // (see face_landmark_detection_ex.cpp for an introduction)
        shape_predictor pose_model;
        deserialize("/home/csn/ubuntu-examples/datasets/shape_predictor_68_face_landmarks.dat") >> pose_model;
        //deserialize("/home/csn/ubuntu-examples/datasets/sp.dat") >> pose_model;

        // And finally we load the DNN responsible for face recognition.
        anet_type net;
        deserialize("/home/csn/ubuntu-examples/datasets/dlib_face_recognition_resnet_model_v1.dat") >> net;
        //deserialize("/home/csn/ubuntu-examples/datasets/metric_network_renset.dat") >> net;

        // Grab and process frames until the main window is closed by the user.
        while(!win.is_closed())
        {
            // Grab a frame
            cv::Mat temp;
            if (!cap.read(temp))
            {
                break;
            }
            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            //cv::rotate(temp, temp, cv::ROTATE_90_CLOCKWISE);
            cv_image<bgr_pixel> cimg(temp);

            // Here we declare an image object that can store rgb_pixels.  Note that in
            // dlib there is no explicit image object, just a 2D array and
            // various pixel types.
            array2d<rgb_pixel> img;
            dlib::assign_image(img, cimg);

            // Run the face detector on the image, and
            // for each face extract a copy
            // that has been normalized to 150x150 pixels in size and
            // appropriately rotated and centered.
            std::vector<matrix<rgb_pixel>> faces_matrix;

    double retval = 0.0;
    //try {
       for (auto face_matrix : detector(img)) {
           try {
               auto shape_matrix = pose_model(img, face_matrix);
               matrix<rgb_pixel> face_chip;
               extract_image_chip(img, get_face_chip_details(shape_matrix,150,0.25), face_chip);
               faces_matrix.push_back(move(face_chip));
               // Also put some boxes on the faces so we can see that the detector is finding
               // them.
               win_faces_matrix.clear_overlay();
               win_faces_matrix.set_image(img);
               win_faces_matrix.add_overlay(face_matrix);
               //win_faces_matrix.add_overlay(face_matrix, rgb_pixel(255,0,0));
               win_faces_matrix.add_overlay(render_face_detections(shape_matrix));

               try {
                   for (int b = 36; b < 42; ++b) {
                       p.x = (int)shape_matrix.part(b).x();
                       p.y = (int)shape_matrix.part(b).y();
                       lefteye.push_back(p);
                   }
                   for (int b = 42; b < 48; ++b) {
                       p.x = (int)shape_matrix.part(b).x();
                       p.y = (int)shape_matrix.part(b).y();
                       righteye.push_back(p);
                   }
                   for (int b = 48; b < 68; ++b) {
                       p.x = (int)shape_matrix.part(b).x();
                       p.y = (int)shape_matrix.part(b).y();
                       mouth.push_back(p);
                   }
                   //Compute Eye aspect ration for eyes
                   double right_ear = compute_EAR(righteye);
                   double left_ear = compute_EAR(lefteye);
                   double mouth_score = compute_MAR(mouth);

                   cout << "right_ear:" << right_ear << endl;
                   cout << "left_ear:" << left_ear << endl;
                   cout << "mouth_score:" << mouth_score << endl;

                   if (right_ear == 0.0 || left_ear == 0.0 || mouth_score == 0.0) {
                       //return retval;
                       cout << "retval:" << retval << endl;

                   }

                   //if the avarage eye aspect ratio of lef and right eye less than 0.2, the stat$
                   retval = MAX(MAX(right_ear, left_ear), mouth_score);

                   righteye.clear();
                   lefteye.clear();
                   mouth.clear();
               } catch (...) {
                    //return retval;
               }

           } catch (...) {
                //return retval;
           }
       }
    //} catch (...) {
            //return retval;
    //}

    if (faces_matrix.size() == 0)
    {
        cout << "No faces found in image!" << endl;
        //return 1;
    }
    else
    {

    // This call asks the DNN to convert each face image in faces into a 128D vector.
    // In this 128D vector space, images from the same person will be close to each other
    // but vectors from different people will be far apart.  So we can use these vectors to
    // identify if a pair of images are from the same person or from different people.  
    std::vector<matrix<float,0,1>> face_descriptors = net(faces_matrix);


    // In particular, one simple thing we can do is face clustering.  This next bit of code
    // creates a graph of connected faces and then uses the Chinese whispers graph clustering
    // algorithm to identify how many people there are and which faces belong to whom.
    std::vector<sample_pair> edges;
    for (size_t i = 0; i < face_descriptors.size(); ++i)
    {
        for (size_t j = i; j < face_descriptors.size(); ++j)
        {
            // Faces are connected in the graph if they are close enough.  Here we check if
            // the distance between two face descriptors is less than 0.6, which is the
            // decision threshold the network was trained to use.  Although you can
            // certainly use any other threshold you find useful.
            if (length(face_descriptors[i]-face_descriptors[j]) < 0.6)
                edges.push_back(sample_pair(i,j));
        }
    }
    std::vector<unsigned long> labels;
    const auto num_clusters = chinese_whispers(edges, labels);
    // This will correctly indicate that there are x people in the image.
    cout << "number of people found in the image: "<< num_clusters << endl;

    // Now let's display the face clustering results on the screen.  You will see that it
    // correctly grouped all the faces.
    std::vector<image_window> win_clusters(num_clusters);
    for (size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
    {
        std::vector<matrix<rgb_pixel>> temp;
        for (size_t j = 0; j < labels.size(); ++j)
        {
            if (cluster_id == labels[j])
                temp.push_back(faces_matrix[j]);
        }
        win_clusters[cluster_id].set_title("face cluster " + cast_to_string(cluster_id));
        win_clusters[cluster_id].set_image(tile_images(temp));
    }

    // Finally, let's print one of the face descriptors to the screen.  
    cout << "face descriptor for one face: " << trans(face_descriptors[0]) << endl;

    // It should also be noted that face recognition accuracy can be improved if jittering
    // is used when creating face descriptors.  In particular, to get 99.38% on the LFW
    // benchmark you need to use the jitter_image() routine to compute the descriptors,
    // like so:
    matrix<float,0,1> face_descriptor = mean(mat(net(jitter_image(faces_matrix[0]))));
    cout << "jittered face descriptor for one face: " << trans(face_descriptor) << endl;
    // If you use the model without jittering, as we did when clustering the bald guys, it
    // gets an accuracy of 99.13% on the LFW benchmark.  So jittering makes the whole
    // procedure a little more accurate but makes face descriptor calculation slower.

    }

            // Detect faces
            std::vector<rectangle> faces_rectangle = detector(img);
            // Find the pose of each face.
            std::vector<full_object_detection> shapes_fod;
            for (unsigned long i = 0; i < faces_rectangle.size(); ++i)
                shapes_fod.push_back(pose_model(img, faces_rectangle[i]));

            // Find bright lines
            dlib::image_gradients dig;
            dlib::array2d<float> gradient_xx, gradient_xy, gradient_yy;
            dlib::array2d<float> fbl_horz_gradient, fbl_vert_gradient;
            dig.gradient_xx(img,gradient_xx);
            dig.gradient_xy(img,gradient_xy);
            dig.gradient_yy(img,gradient_yy);
            dlib::array2d<rgb_pixel> fbl_img;
            find_bright_lines(gradient_xx,gradient_xy,gradient_yy,fbl_horz_gradient, fbl_vert_gradient);
            suppress_non_maximum_edges(fbl_horz_gradient, fbl_vert_gradient, fbl_img);
            // show in window
            //image_window win_fbl(fbl_img, "Find bright lines");
            win_fbl.set_image(fbl_img);

            // Now let's use some image functions.  First let's blur the image a little.
            array2d<unsigned char> blurred_img;
            gaussian_blur(img, blurred_img);

            // Now find the horizontal and vertical gradient images.
            array2d<short> horz_gradient, vert_gradient;
            array2d<unsigned char> edge_image;
            sobel_edge_detector(blurred_img, horz_gradient, vert_gradient);

            // now we do the non-maximum edge suppression step so that our edges are nice and t$
            suppress_non_maximum_edges(horz_gradient, vert_gradient, edge_image);

            // Now we would like to see what our images look like.  So let's use a
            // window to display them on the screen.  (Note that you can zoom into
            // the window by holding CTRL and scrolling the mouse wheel)
            //image_window win_nei(edge_image, "Normal Edge Image");
            win_nei.set_image(edge_image);

	    // We can also easily display the edge_image as a heatmap or using the jet color
            // scheme like so.
            //image_window win_hot(heatmap(edge_image), "Heatmap Image");
            //image_window win_jet(jet(edge_image), "Jet Image");
            win_hot.set_image(heatmap(edge_image));
            win_jet.set_image(jet(edge_image));

            // also make a window to display the original image
            //image_window win_org(img, "Original Image");
            win_org.set_image(img);

            // Sometimes you want to get input from the user about which pixels are important
            // for some task.  You can do this easily by trapping user clicks as shown below.
            // This loop executes every time the user double clicks on some image pixel and it
            // will terminate once the user closes the window.

            //point p;
            //win_nei.get_next_double_click(p);
            //cout << "User double clicked on pixel:         " << p << endl;
            //cout << "edge pixel value at this location is: " << (int)edge_image[p.y()][p.x()] << endl;

            //while (win_nei.get_next_double_click(p))
            //{
            //   cout << "User double clicked on pixel:         " << p << endl;
            //   cout << "edge pixel value at this location is: " << (int)edge_image[p.y()][p.x()] << endl;
            //}

            // wait until the user closes the windows before we let the program 
            // terminate.
            //win_fbl.wait_until_closed();
            //win_nei.wait_until_closed();
            //win_hot.wait_until_closed();
            //win_jet.wait_until_closed();
            //win_org.wait_until_closed();

            // Finally, note that you can access the elements of an image using the normal [row$
            // operator like so:
            cout << horz_gradient[0][3] << endl;
            cout << "number of rows in image:    " << horz_gradient.nr() << endl;
            cout << "number of columns in image: " << horz_gradient.nc() << endl;

            // Display it all on the screen
            win.clear_overlay();
            //win.set_image(cimg);
            win.set_image(edge_image);
            win.add_overlay(render_face_detections(shapes_fod));
            //win.add_overlay(render_face_detections(shapes_fod),rgb_pixel(0,255,0));
            win.add_overlay(faces_rectangle);

            cout << "hit enter to continue" << endl;
            cin.get();

        }
        // wait until the user closes the windows
        // before we let the program terminate
        //win.wait_until_closed();

        cout << "hit enter to terminate" << endl;
        cin.get();

    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
)
{
    // All this function does is make 100 copies of img, all slightly jittered by being
    // zoomed, rotated, and translated a little bit differently. They are also randomly
    // mirrored left to right.
    thread_local dlib::rand rnd;

    std::vector<matrix<rgb_pixel>> crops; 
    for (int i = 0; i < 100; ++i)
        crops.push_back(jitter_image(img,rnd));

    return crops;
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    try
    {
        /*
        if (argc != 2)
        {
           cout << "Run this example by invoking it like this: " << endl;
           cout << "   ./videocapture_face_pose_edges_ex /dev/video0" << endl;
           cout << endl;
           cout << "with video device OR with single frame as below" << endl;
           cout << endl;
           cout << "   ./videocapture_face_pose_edges_ex faces/bald_guys.jpg" << endl;
           cout << endl;
           cout << "You will also need to get the face landmarking model file as well as " << endl;
           cout << "the face recognition model file.  Download and then decompress these files from: " << endl;
           cout << "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2" << endl;
           cout << "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2" << endl;
           cout << endl;
           return 1;
        }
        std::string imagepath = argv[1];
        facelock(imagepath);
        */
        facelock();

    } catch(exception& e)
    {
        cout << e.what() << endl;
    }
}

