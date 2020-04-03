//added Git Readme
/*synchro_cloud_data = synchro_cloud.makeShared();
pcl::toROSMsg(*synchro_cloud_data, msg_synchronised);
msg_synchronised.header.stamp = ros::Time::now();
pub_synchronised.publish(msg_synchronised);*/
#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/PCLHeader.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/pcl_base.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/convex_hull.h>

#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h> 
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/JointState.h>
#include <sensor_msgs/PointField.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf2/LinearMath/Quaternion.h>

#include <std_msgs/Header.h>
#include "std_msgs/String.h"
#include <std_srvs/Empty.h>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>
#include <Eigen/Dense>
#include <Eigen/Eigen>

#include <vector> // for vector 
#include <algorithm> // for copy() and assign() 
#include <iterator> // for back_inserter 
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int32MultiArray.h>

#define QUICKHULL_IMPLEMENTATION
#include "quickhull.h"

using namespace std;
using namespace cv;
using namespace sensor_msgs;
using namespace message_filters;
using namespace Eigen;

// Topics
static const string SCAN_TOPIC_1_3 = "/base_link/points_1_3";
static const string SCAN_TOPIC_1_4 = "/base_link/points_1_4";
static const string FILTERED_TOPIC = "/pcl_filtered";

// ROS Publisher
ros::Publisher pub_filtered, pub_clustered, pub_transformed, pub_merged, pub_merged_transformation, pub_octomap_centers, pub_centroid, pub_pred_centroid, pub_robot_removal;
ros::Publisher pub_jointXYZ;
ros::Publisher pub_vis_box,pub_vis_line,pub_vis_posePath, pub_vis_robot_bounding, pub_vis_text, pub_convex_hull;
ros::Publisher pub_cluster0, pub_cluster1, pub_cluster2, pub_cluster3, pub_cluster4, pub_cluster5, objID_pub;

bool first_msg = true;
sensor_msgs::PointCloud2 msg_clustered, msg_filtered,msg_transformed, msg_merged, msg_merged_transformed, msg_octomap_centers, msg_robot_removal, msg_convex_hull;
pcl::PointCloud<pcl::PointXYZ> cloud, filter, source_cloud, sensor_cloud_transformed, sensor_cloud, merged_LiDAR, merged_transformed_cloud, octomap_centers_cloud, octomap_centers_output, robot_removal_cloud, convex_hull_cloud;
pcl::PointCloud<pcl::PointXYZ>::Ptr filter_ptr, cloud_ptr, source_cloud_data, merged_LiDAR_data, merged_transformed_cloud_data, octomap_centers_cloud_data, octomap_centers_output_data, robot_removal_cloud_data, convex_hull_cloud_data;

//ROS Service
ros::ServiceClient clearClient;
std_srvs::Empty srv;
// Variables
int LEAF_SIZE = 40;
float RESOLUTION = 0.76f;
int MIN_FILTERED_CLOUD_SIZE = 30;
int MIN_CLUSTERED_CLOUD_SIZE = 150;
float CLUSTER_TOLERANCE = 0.7;
int MIN_CLUSTER_SIZE = 75;
int MAX_CLUSTER_SIZE = 25000;

ofstream indicesOutput;

struct PointView
  {
    float x;
    float y;
    float z;
  }PointView1;

#define PI 3.14159265
//Rotation Transformation manually put here, based on calibration between two Lidar sensor: Data can be read from QView Calibrate fn
// The angle of rotation (X,Y,Z respectively)  (in radians)
float thetaX = 0; 
float thetaY = 0; 
float thetaZ = -(90.0 * PI / 180.0); 


//RotationMatrix
float rotate11 = cos(thetaZ)*cos(thetaY);
float rotate12 = ((cos(thetaZ)*sin(thetaY)*sin(thetaX))-(sin(thetaZ)*cos(thetaX)));
float rotate13 = ((cos(thetaZ)*sin(thetaY)*sin(thetaX))-(sin(thetaZ)*sin(thetaX)));
float rotate21 = sin(thetaZ)*cos(thetaY);
float rotate22 = ((sin(thetaZ)*sin(thetaY)*sin(thetaX))+(cos(thetaZ)*cos(thetaX)));
float rotate23 = ((sin(thetaZ)*sin(thetaY)*cos(thetaX))-(cos(thetaZ)*sin(thetaX)));
float rotate31 = -sin(thetaY);
float rotate32 = cos(thetaY)*sin(thetaX);
float rotate33 = cos(thetaY)*cos(thetaX);

//palan


// KF init
int stateDim=4;// [x,y,v_x,v_y]//,w,h]
int measDim=2;// [z_x,z_y,z_w,z_h]
int ctrlDim=0;
cv::KalmanFilter KF0(stateDim,measDim,ctrlDim,CV_32F);
cv::KalmanFilter KF1(stateDim,measDim,ctrlDim,CV_32F);
cv::KalmanFilter KF2(stateDim,measDim,ctrlDim,CV_32F);
cv::KalmanFilter KF3(stateDim,measDim,ctrlDim,CV_32F);
cv::KalmanFilter KF4(stateDim,measDim,ctrlDim,CV_32F);
cv::KalmanFilter KF5(stateDim,measDim,ctrlDim,CV_32F);

std::vector<geometry_msgs::Point> prevClusterCenters;


cv::Mat state(stateDim,1,CV_32F);
cv::Mat_<float> measurement(2,1); 

std::vector<int> objID;// Output of the data association using KF
// measurement.setTo(Scalar(0));

bool firstFrame=true;

double euclidean_distance(geometry_msgs::Point& p1, geometry_msgs::Point& p2)
{
  return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}

//Visualisation Line
visualization_msgs::Marker visualisationLine(std::string ns, int id, float x_min, float y_min, float z_min, float x_max, float y_max, float z_max, float r, float g, float b)
{
  visualization_msgs::Marker line_list;
  line_list.header.frame_id = "base_link";
 
  line_list.ns = ns;
  line_list.id = id;

  line_list.type = visualization_msgs::Marker::LINE_LIST;
  line_list.action = visualization_msgs::Marker::ADD;
  
  //lines
  line_list.scale.x = 0.1;
  line_list.color.r = r;
  line_list.color.g = g;
  line_list.color.b = b;
  line_list.color.a = 1.0;

  geometry_msgs::Point LineStart;
  geometry_msgs::Point LineEnd;

  //LineStart points are currently Static (fed to Robot statically), 
  
  LineStart.x = x_max;
  LineStart.y = y_max;
  LineStart.z = z_max;
  
  LineEnd.x = x_min;
  LineEnd.y = y_min;
  LineEnd.z = z_min;

  line_list.points.push_back(LineStart);
  line_list.points.push_back(LineEnd);

  line_list.lifetime = ros::Duration(0.5);
  return line_list;
}
 
//Visualisation Marker-bounding box
visualization_msgs::Marker boundingBox(float c_x, float c_y, float c_z, int id, float r, float g, float b)
{
  visualization_msgs::Marker marker;
  marker.header.frame_id = "base_link";

  marker.id = id;

  marker.type = visualization_msgs::Marker::CUBE;
  marker.action = visualization_msgs::Marker::ADD;
 
  marker.pose.position.x = c_x;
  marker.pose.position.y = c_y;
  marker.pose.position.z = 1.5;

  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
 
  marker.scale.x = 0.5;
  marker.scale.y = 0.5;
  marker.scale.z = 1.0;
 
  if (marker.scale.x ==0)
      marker.scale.x=0.1;

  if (marker.scale.y ==0)
    marker.scale.y=0.1;

  if (marker.scale.z ==0)
    marker.scale.z=0.1;
   
  marker.color.r = r;
  marker.color.g = g;
  marker.color.b = b;
  marker.color.a = 0.5;

  marker.lifetime = ros::Duration(1.0);
  return marker;
}

//Visualisation Marker-bounding text
visualization_msgs::Marker boundingText(float c_x, float c_y, float c_z, int id, float r, float g, float b, string reg_id)
{
  visualization_msgs::Marker markerText;
  markerText.header.frame_id = "base_link";

  markerText.id = id;

  markerText.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  markerText.action = visualization_msgs::Marker::ADD;
  
  markerText.text = reg_id;
  markerText.pose.position.x = c_x;
  markerText.pose.position.y = c_y;
  markerText.pose.position.z = 1.5;

  markerText.scale.z = 1.0;
 
  markerText.color.r = 1;
  markerText.color.g = 1;
  markerText.color.b = 1;
  markerText.color.a = 1;

  markerText.lifetime = ros::Duration(1.0);
  return markerText;
}

//Robot Bounding box
visualization_msgs::Marker robotBoundingBox(string ns, float r, float g, float b)
{
  visualization_msgs::Marker robotBounding;
  robotBounding.header.frame_id = "base_link";

  robotBounding.ns = ns;
  robotBounding.id = 1;
  
  robotBounding.type = visualization_msgs::Marker::CUBE;
  robotBounding.action = visualization_msgs::Marker::ADD;
 
  robotBounding.pose.position.x = 0;
  robotBounding.pose.position.y = 0;
  robotBounding.pose.position.z = 1.5;

  robotBounding.pose.orientation.x = 0.0;
  robotBounding.pose.orientation.y = 0.0;
  robotBounding.pose.orientation.z = 0.0;
  robotBounding.pose.orientation.w = 1.0;
 
  robotBounding.scale.x = 6;
  robotBounding.scale.y = 6;
  robotBounding.scale.z = 6;
    
  robotBounding.color.r = r;
  robotBounding.color.g = g;
  robotBounding.color.b = b;
  robotBounding.color.a = 0.3;

  robotBounding.lifetime = ros::Duration(0.5);
  return robotBounding;
  
}

//Path
nav_msgs::Path navigationPath(float pre_c_x, float pre_c_y, float c_x, float c_y, float c_z)
{
  nav_msgs::Path HumanPath;
  
  geometry_msgs::PoseStamped centroidPosition;

  centroidPosition.header.frame_id = "base_link";

  float x_diff = c_x - pre_c_x;
  float y_diff = c_y - pre_c_y;

  /*tf2::Quaternion myQuaternion;
  myQuaternion.setEulerZYX(90,0,0);
  
 if(x_diff > 0 && y_diff > 0){centroidPosition.pose.orientation.y = 45;}
  else if(x_diff == 0 && y_diff > 0){centroidPosition.pose.orientation.y = 90;}
  else if(x_diff < 0 && y_diff > 0){centroidPosition.pose.orientation.y = 135;}
  else if(x_diff < 0 && y_diff == 0){centroidPosition.pose.orientation.y = 180;}
  else if(x_diff < 0 && y_diff < 0){centroidPosition.pose.orientation.y = 225;}
  else if(x_diff == 0 && y_diff < 0){centroidPosition.pose.orientation.y = 270;}
  else if(x_diff > 0 && y_diff < 0){centroidPosition.pose.orientation.y = 315;}
  else if(x_diff > 0 && y_diff == 0){centroidPosition.pose.orientation.y = 360;} */

  centroidPosition.pose.orientation.x = 0;
  centroidPosition.pose.orientation.y = 0;
  centroidPosition.pose.orientation.z = 0;
  centroidPosition.pose.orientation.w = 1.0;

  centroidPosition.pose.position.x = c_x;
  centroidPosition.pose.position.y = c_y;
  centroidPosition.pose.position.z = c_z;

  HumanPath.header.frame_id = "base_link";
  HumanPath.poses.push_back(centroidPosition);

  return HumanPath;
}

//Pose array

geometry_msgs::Pose posePath(float c_x, float c_y, float c_z, int id)
{
  geometry_msgs::Pose centroidPosition;
  /*tf2::Quaternion myQuaternion;
  myQuaternion.setEulerZYX(90,0,0);
  
 if(x_diff > 0 && y_diff > 0){centroidPosition.pose.orientation.y = 45;}
  else if(x_diff == 0 && y_diff > 0){centroidPosition.pose.orientation.y = 90;}
  else if(x_diff < 0 && y_diff > 0){centroidPosition.pose.orientation.y = 135;}
  else if(x_diff < 0 && y_diff == 0){centroidPosition.pose.orientation.y = 180;}
  else if(x_diff < 0 && y_diff < 0){centroidPosition.pose.orientation.y = 225;}
  else if(x_diff == 0 && y_diff < 0){centroidPosition.pose.orientation.y = 270;}
  else if(x_diff > 0 && y_diff < 0){centroidPosition.pose.orientation.y = 315;}
  else if(x_diff > 0 && y_diff == 0){centroidPosition.pose.orientation.y = 360;} */

  centroidPosition.position.x = c_x;
  centroidPosition.position.y = c_y;
  centroidPosition.position.z = c_z;

  return centroidPosition;
}


void visualisationHull(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster, int id)
{
  //LineStrip
  visualization_msgs::Marker line_strip;
  line_strip.header.frame_id= "/base_link";
  line_strip.header.stamp = ros::Time::now();
  line_strip.ns = "points_and_lines";
  line_strip.action = visualization_msgs::Marker::ADD;
  line_strip.pose.orientation.w  = 1.0;


  line_strip.id = id;


  line_strip.type = visualization_msgs::Marker::LINE_STRIP;

  // LINE_STRIP/LINE_LIST markers use only the x component of scale, for the line width
  line_strip.scale.x = 0.1;

  line_strip.color.b = 1.0;
  line_strip.color.a = 1.0;

  geometry_msgs::Point ConvexPoints;
  for (std::size_t i = 0; i < cloud_cluster->points.size (); ++i)
  {
    ConvexPoints.x =  cloud_cluster->points[i].x;
    ConvexPoints.y =  cloud_cluster->points[i].y;
    ConvexPoints.z =  cloud_cluster->points[i].z;
    line_strip.points.push_back(ConvexPoints);
  }
  pub_convex_hull.publish(line_strip);
}

//Quick HUll
  //Input:=  set of points in pcl
  //Output:= Hull in .obj file
void quickHullFunction(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster)
{

  int n= cloud_cluster->points.size ();
  const int nmeshes = n/2;
  qh_vertex_t vertices[n];
  
  cout << "\n BEFORE THE VERTICES-------------------\n";
  qh_mesh_t meshes[nmeshes];
  for (int j = 0; j < nmeshes; ++j) 
  {
    for (std::size_t i = 0; i < n; ++i) 
    {
      vertices[i].x = cloud_cluster->points[i].x;
      vertices[i].y = cloud_cluster->points[i].y;
      vertices[i].z = cloud_cluster->points[i].z;
    }
  cout << "\n AFTER THE VERTICES-------------------\n";
  meshes[j] = qh_quickhull3d(vertices, n);
  cout << "\n AFTER THE MESH-------------------\n";
  // ...
  }
  std::ofstream hullOutput("/home/manish/catkin_ws/src/lidar-human-classification/centroid_data/hull_mesh.obj");
    if (hullOutput.is_open()) {
        int vertexOffset = 0;
        int normalOffset = 0;
        for (int i = 0; i < nmeshes; ++i) {
            qh_mesh_t m = meshes[i];
            hullOutput << " " << "\n";
            hullOutput << "o " << std::to_string(i) << "\n";
            for (int i = 0; i < m.nvertices; ++i) {
                qh_vertex_t v = m.vertices[i];
                hullOutput << "v " << v.x << " " << v.y << " " << v.z << "\n";
            }
            for (int i = 0; i < m.nnormals; ++i) {
                qh_vec3_t n = m.normals[i];
                hullOutput << "vn " << n.x << " " << n.y << " " << n.z << "\n";
            }
            for (int i = 0, j = 0; i < m.nindices; i += 3, j++) {
                hullOutput << "f ";
                hullOutput << m.indices[i+0] + 1 + vertexOffset << "//";
                hullOutput << m.normalindices[j] + 1 + normalOffset << " ";
                hullOutput << m.indices[i+1] + 1 + vertexOffset << "//";
                hullOutput << m.normalindices[j] + 1 + normalOffset << " ";
                hullOutput << m.indices[i+2] + 1 + vertexOffset << "//";
                hullOutput << m.normalindices[j] + 1 + normalOffset << "\n";
            }
            vertexOffset += m.nvertices;
            normalOffset += m.nnormals;
        }
    }

    for (int i = 0; i < nmeshes; ++i) 
    {
        qh_free_mesh(meshes[i]);
    }
    cout << "\n AFTER THE FREE MESH-------------------\n";
}
void new_object(float person_x, float  person_y, float z_min, float robot_x, float robot_y, float robot_z , int id)
{
  pub_vis_line.publish(visualisationLine("cluster_line_green", id, person_x, person_y, z_min, robot_x, robot_y, robot_z, 0.0, 1.0, 0.0));

}
//Lines and Line Strips for Person
void new_person(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster, int id)
{
  Eigen::Vector4f centroid;
  Eigen::Vector4f body_min;
  Eigen::Vector4f body_max;

  pcl::PointXYZ cloud_min;
  pcl::PointXYZ cloud_max;
 
  pcl::compute3DCentroid (*cloud_cluster, centroid);
  pcl::getMinMax3D (*cloud_cluster, cloud_min, cloud_max);

  geometry_msgs::Point pose, velocity;
  pose.x = centroid[0];
  pose.y = centroid[1];
  pose.z = centroid[2];

  float x_min = cloud_min.x;
  float y_min = cloud_min.y;
  float x_max = cloud_max.x;
  float y_max = cloud_max.y;

  /*//Line
  if(x_min >=0 && y_min >=0)
  {
    pub_vis_line.publish(visualisationLine("cluster_line_green", id, x_min, y_min, cloud_min.z, 0.0, 1.0, 0.0));
  }
  else if(x_min <0 && y_min >=0)
  {
    pub_vis_line.publish(visualisationLine("cluster_line_green", id, x_max, y_min, cloud_min.z, 0.0, 1.0, 0.0));
  }
  else if(x_min <0 && y_min <0)
  {
    pub_vis_line.publish(visualisationLine("cluster_line_green", id, x_max, y_max, cloud_min.z, 0.0, 1.0, 0.0));
  }
  else if(x_min >=0 && y_min <0)
  {
    pub_vis_line.publish(visualisationLine("cluster_line_green", id, x_min, y_max, cloud_min.z, 0.0, 1.0, 0.0));
  }*/

  //LineStrip
  visualization_msgs::Marker line_strip;
  line_strip.header.frame_id= "/base_link";
  line_strip.header.stamp = ros::Time::now();
  line_strip.ns = "points_and_lines";
  line_strip.action = visualization_msgs::Marker::ADD;
  line_strip.pose.orientation.w  = 1.0;


  line_strip.id = id;


  line_strip.type = visualization_msgs::Marker::LINE_STRIP;

  // LINE_STRIP/LINE_LIST markers use only the x component of scale, for the line width
  line_strip.scale.x = 0.1;

  line_strip.color.b = 1.0;
  line_strip.color.a = 1.0;

  geometry_msgs::Point ConvexPoints;
  for (std::size_t i = 0; i < cloud_cluster->points.size (); ++i)
  {
    ConvexPoints.x =  cloud_cluster->points[i].x;
    ConvexPoints.y =  cloud_cluster->points[i].y;
    ConvexPoints.z =  cloud_cluster->points[i].z;
    line_strip.points.push_back(ConvexPoints);
  }
  
  pub_convex_hull.publish(line_strip);
}
		// Human edge (HumanXYZEnd2DPoit) facing towards the robot base is the piont of reference from human to robot//
		// Edge of the robot in the relevant quadrandt in 2D robot cell plane facing towards human edge HumanXYZEnd2DPoit 
void safety_distance(const sensor_msgs::PointCloud2ConstPtr& clustered_msg, const sensor_msgs::PointCloud2ConstPtr& robot_removal_point) // TODO: naming of clustere_msg human and robot point cloud; TODO input and output
{
  //clustered_objects
  pcl::PointCloud<pcl::PointXYZ> clustered_cloud;
  pcl::PCLPointCloud2 pcl_clustered_object;
  pcl_conversions::toPCL(*clustered_msg, pcl_clustered_object);  
  pcl::fromPCLPointCloud2(pcl_clustered_object, clustered_cloud); 
  pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cloud_data (new pcl::PointCloud<pcl::PointXYZ>);

  clustered_cloud_data = clustered_cloud.makeShared();

  //Robot_Removal_Cloud
  pcl::PointCloud<pcl::PointXYZ> robot_cloud;
  pcl::PCLPointCloud2 pcl_robot_cloud;
  pcl_conversions::toPCL(*robot_removal_point, pcl_robot_cloud);  
  pcl::fromPCLPointCloud2(pcl_robot_cloud, robot_cloud); 
  pcl::PointCloud<pcl::PointXYZ>::Ptr robot_cloud_data (new pcl::PointCloud<pcl::PointXYZ>);

  robot_cloud_data = robot_cloud.makeShared();

  Eigen::Vector4f centroid;
  pcl::PointXYZ cloud_min;
  pcl::PointXYZ cloud_max;

  pcl::compute3DCentroid (*clustered_cloud_data, centroid);
  pcl::getMinMax3D (*clustered_cloud_data, cloud_min, cloud_max);// gets human point cluod min max with respect to robot base, which is transformed

  geometry_msgs::Point pose, velocity;
  pose.x = centroid[0];
  pose.y = centroid[1];
  pose.z = centroid[2];

  float x_min = cloud_min.x;
  float y_min = cloud_min.y;
  float z_min = cloud_min.z;
  float x_max = cloud_max.x;
  float y_max = cloud_max.y;

  float person_x;
  float person_y;

  //KDTree Search // searching the relevant edge facing towards robot
  
  
  //	(- -)	|		x y (- +)
  //			|________________
  //
  //	(+ -)				(+ +)
  //
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud (robot_cloud_data);
  pcl::PointXYZ searchPoint;
  
  if(x_min >=0 && y_min >=0)
  {
    person_x = x_min;
    person_y = y_min;    
  }
  else if(x_min <0 && y_min >=0)
  {
    person_x = x_max;
    person_y = y_min;
  }
  else if(x_min <0 && y_min <0)
  {
    person_x = x_max;
    person_y = y_max;
  }
  else if(x_min >=0 && y_min <0)
  {
    person_x = x_min;
    person_y = y_max;
  }
  searchPoint.x = person_x; // TODO naming: edge of human facing robot.. 
  searchPoint.y = person_y;
  searchPoint.z = z_min;

  int K = 10;
  
  std::vector<int> pointIdxNKNSearch(K); // TODO: naming kdtre indexes facing towards human
  std::vector<float> pointNKNSquaredDistance(K);

  std::cout << "K nearest neighbor search at (" << searchPoint.x 
            << " " << searchPoint.y 
            << " " << searchPoint.z
            << ") with K=" << K << std::endl;

  sensor_msgs::PointCloud2 msg_count;
  msg_count.fields = clustered_msg->fields;

  int person_count = msg_count.fields[0].count;

  if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )// squareDistance: Euclidian distance
  {
    float robot_x = robot_cloud_data->points[ pointIdxNKNSearch[0] ].x;
    float robot_y = robot_cloud_data->points[ pointIdxNKNSearch[0] ].y;// shortest distance by 0th index
    float robot_z = robot_cloud_data->points[ pointIdxNKNSearch[0] ].z;

    cout << "\n Robot Minimal Point(Kd tREe): (" << robot_x << "," << robot_y << "," << robot_z << ")\n";
    cout << "\n Squared Distance" <<  pointNKNSquaredDistance[0];
    person_count++; 
    new_object(person_x, person_y, z_min, robot_x, robot_y, robot_z, person_count);//todo Naming convention; optimization
  }
  visualisationHull(robot_cloud_data,person_count);
}

//LiDAR Transformation (LiDAR2 to LiDAR1)
  //Input:=  Cloud to be transformed (Sensor_1_4)
  //Output:= Sensor_1_4 transformed to Sensor_1_3
void cloud_transformation(const sensor_msgs::PointCloud2ConstPtr& cloud_transformation_msg)
{
  // Convert to PCL data type
  pcl::PCLPointCloud2 pcl_transformation;
  pcl_conversions::toPCL(*cloud_transformation_msg, pcl_transformation);  
  pcl::fromPCLPointCloud2(pcl_transformation, source_cloud);  

  source_cloud.header = pcl_conversions::toPCL(cloud_transformation_msg->header);
  source_cloud_data = source_cloud.makeShared();

  Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
  
  //             (row, column)
  transformation (0,0) = rotate11;
  transformation (0,1) = rotate12;
  transformation (0,2) = rotate13;
  transformation (1,0) = rotate21;
  transformation (1,1) = rotate22;
  transformation (1,2) = rotate23;
  transformation (2,0) = rotate31;
  transformation (2,1) = rotate32;
  transformation (2,2) = rotate33;


  // Translation of 3.89 meters on the x axis, 5.85 meters on the y axis
  // Input data from Qview Calibration Tool, X, Y, Z Translation from Lidar2 to Lidar1(1.3)
  transformation (0,3) = 3.89;
  transformation (1,3) = 5.85;
  transformation (2,3) = 0.0;

  // Executing the transformation
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud;
  transformed_cloud = (pcl::PointCloud<pcl::PointXYZ>::Ptr) new pcl::PointCloud<pcl::PointXYZ>();
  (transformed_cloud)->header = pcl_conversions::toPCL(cloud_transformation_msg->header);

  pcl::transformPointCloud (*source_cloud_data, *transformed_cloud, transformation);
  
  pcl::toROSMsg(*transformed_cloud, msg_transformed);
  msg_transformed.header = cloud_transformation_msg->header;
  pub_transformed.publish(msg_transformed);
  //pub_vis_navPath.publish(navigationPath(0, 2, 2, 1, 1.5)); //For testing the Navigation arrow
}

//Merging/Concatenating the two LiDAR data
  //Input:=  Two Clouds for merging (Sensor_1_3 and Transformed Sensor_1_4)
  //Output:= Single Cloud (Merged: Sensor_1_3 + Sensor_1_4)
void cloud_concatenate(const sensor_msgs::PointCloud2ConstPtr& cloud_msg_sensor, const sensor_msgs::PointCloud2ConstPtr& cloud_msg_transformed)
{
  // Convert to PCL data type
  pcl::PCLPointCloud2 pcl_sensor_transformed;
  pcl::PCLPointCloud2 pcl_sensor;

  pcl_conversions::toPCL(*cloud_msg_transformed, pcl_sensor_transformed);
  pcl_conversions::toPCL(*cloud_msg_sensor, pcl_sensor);

  pcl::fromPCLPointCloud2(pcl_sensor_transformed, sensor_cloud_transformed);
  pcl::fromPCLPointCloud2(pcl_sensor, sensor_cloud);

  sensor_cloud_transformed.header = pcl_conversions::toPCL(cloud_msg_transformed->header);
  sensor_cloud.header = pcl_conversions::toPCL(cloud_msg_sensor->header);
 
  merged_LiDAR  = sensor_cloud;
  merged_LiDAR += sensor_cloud_transformed;
  //pcl::concatenateFields (sensor_cloud_data, sensor_cloud_transformed_data, merged_LiDAR);
  
  merged_LiDAR_data = merged_LiDAR.makeShared();
  pcl::toROSMsg(*merged_LiDAR_data, msg_merged);
  msg_merged.header = cloud_msg_sensor->header;
  msg_merged.header.stamp.now();
  pub_merged.publish(msg_merged); 
} 

//Transform to Robot Co-ordinates
  //Input:=  Merged Cloud (Sensor_1_3 + Sensor_1_4)
  //Output:= Merged sensor data Transformed to Robot co-ordinates
void merged_transform(const sensor_msgs::PointCloud2ConstPtr& cloud_msg_merged_transformed)
{
// Convert to PCL data type
  pcl::PCLPointCloud2 pcl_merged_transformation;
  pcl_conversions::toPCL(*cloud_msg_merged_transformed, pcl_merged_transformation);  
  pcl::fromPCLPointCloud2(pcl_merged_transformation, merged_transformed_cloud);  

  merged_transformed_cloud.header = pcl_conversions::toPCL(cloud_msg_merged_transformed->header);
  merged_transformed_cloud_data = merged_transformed_cloud.makeShared();

  Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
  
  //Shift the Complete data (-0.8m,-2.8m,-1.5m) on (x,y,z) Respectively (where (0,0,0) is Robot); Lidar to Robot Transformation

  //todo: Rotation matrix to be included after actual Lidar to Robot values are available 
  transformation (0,3) = -0.60;
  transformation (1,3) = -2.95;
  transformation (2,3) = 1.5;

  // Executing the translation
  pcl::PointCloud<pcl::PointXYZ>::Ptr merged_transformed_cloud;
  merged_transformed_cloud = (pcl::PointCloud<pcl::PointXYZ>::Ptr) new pcl::PointCloud<pcl::PointXYZ>();
  (merged_transformed_cloud)->header = pcl_conversions::toPCL(cloud_msg_merged_transformed->header);

  pcl::transformPointCloud (*merged_transformed_cloud_data, *merged_transformed_cloud, transformation);
  
  pcl::toROSMsg(*merged_transformed_cloud, msg_merged_transformed);
  msg_merged_transformed.header = cloud_msg_merged_transformed->header;
  msg_merged_transformed.header.stamp = ros::Time::now();
  pub_merged_transformation.publish(msg_merged_transformed);
}
// Code review 07.02.20


//DH Parameters
  //Input:=  Joint Angle, Link offset, Link Lenght and Link twist
  //Output:= DH matrix
Eigen::Matrix4f dhParam(float Theta,float d,float a,float alpha)
{
//  cout << "\nTheta: " << Theta << "\nd: " << d << "\na: " << a << "\nalpha: " << alpha;

  Eigen::Matrix4f getTransformMatrix = Eigen::Matrix4f::Identity();
  getTransformMatrix(0,0) = cos(Theta);
  getTransformMatrix(0,1) = -cos(alpha) * sin(Theta);
  getTransformMatrix(0,2) = sin(alpha) * sin(Theta);
  getTransformMatrix(0,3) = a * cos(Theta);

  getTransformMatrix(1,0) = sin(Theta);
  getTransformMatrix(1,1) = cos(alpha) * cos(Theta);
  getTransformMatrix(1,2) = -sin(alpha) * cos(Theta);
  getTransformMatrix(1,3) = a * sin(Theta);

  getTransformMatrix(2,0) = 0;
  getTransformMatrix(2,1) = sin(alpha);
  getTransformMatrix(2,2) = cos(alpha);
  getTransformMatrix(2,3) = d;

  return getTransformMatrix;
}

//Robot joints angles to Joint X,Y,Z
  //Input:=  Robot joint States (radians)
  //Output:= Robot Joint Positions (6 XYZ)
void robotJoints(const sensor_msgs::JointState& jointMsg)
{
  
  float a1 = jointMsg.position[0];
  float a2 = jointMsg.position[1];
  float a3 = jointMsg.position[2];
  float a4 = jointMsg.position[3];
  float a5 = jointMsg.position[4];
  float a6 = jointMsg.position[5];

  cout << "\na1 Position: " << a1;
  cout << "\na2 Position: " << a2;
  cout << "\na3 Position: " << a3;
  cout << "\na4 Position: " << a4;
  cout << "\na5 Position: " << a5;
  cout << "\na6 Position: " << a6;
  
  //Joint Angle
  float jointAngle_Theta0 = 0; 
  float jointAngle_Theta1 = a1; 
  float jointAngle_Theta2 = a2; 
  float jointAngle_Theta3 = a3; 
  float jointAngle_Theta4 = a4; 
  float jointAngle_Theta5 = a5; 
  float jointAngle_Theta6 = a6; 
  //Link Offset
  float linkOffset_d0 = 0.73; 
  float linkOffset_d1 = 0.4594; 
  float linkOffset_d2 = 0.178; 
  float linkOffset_d3 = -0.173; 
  float linkOffset_d4 = 0.0345;
  float linkOffset_d5 = -0.0345; 
  float linkOffset_d6 = 0;
  //Link Length
  float linkLength_a0 = 0;
  float linkLength_a1 = 0.35;     
  float linkLength_a2 = 1.35;
  float linkLength_a3 = 0.9445;
  float linkLength_a4 = 0.2555;
  float linkLength_a5 = 0.188;
  float linkLength_a6 = 0;
  //Link Twist
  float linkTwist_alpha0 = 0;
  float linkTwist_alpha1 = -(90.0 * PI / 180.0);
  float linkTwist_alpha2 = 0;
  float linkTwist_alpha3 = -(90.0 * PI / 180.0);
  float linkTwist_alpha4 = (90.0 * PI / 180.0);
  float linkTwist_alpha5 = -(90.0 * PI / 180.0);
  float linkTwist_alpha6 = 0;
  
  Eigen::Matrix4f TB0 = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f T01 = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f T12 = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f T23 = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f T34 = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f T45 = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f T56 = Eigen::Matrix4f::Identity();

  Eigen::Matrix4f Link1 = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f Link2 = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f Link3 = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f Link4 = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f Link5 = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f Link6 = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f Link_TCP = Eigen::Matrix4f::Identity();

  TB0 = dhParam(jointAngle_Theta0, linkOffset_d0, linkLength_a0, linkTwist_alpha0);
  T01 = dhParam(jointAngle_Theta1, linkOffset_d1, linkLength_a1, linkTwist_alpha1);
  T12 = dhParam(jointAngle_Theta2, linkOffset_d2, linkLength_a2, linkTwist_alpha2);
  T23 = dhParam(jointAngle_Theta3, linkOffset_d3, linkLength_a3, linkTwist_alpha3);
  T34 = dhParam(jointAngle_Theta4, linkOffset_d4, linkLength_a4, linkTwist_alpha4);
  T45 = dhParam(jointAngle_Theta5, linkOffset_d5, linkLength_a5, linkTwist_alpha5);
  T56 = dhParam(jointAngle_Theta6, linkOffset_d6, linkLength_a6, linkTwist_alpha6);
  
  //Transformation to base axis
  Eigen::Matrix4f transformationBaseAxis = Eigen::Matrix4f::Identity();
  //Robot ROtation in URDF data -80 deg;
  transformationBaseAxis (0,0) = cos(-80 * PI / 180.0);
  transformationBaseAxis (0,1) = -sin(-80 * PI / 180.0);
  transformationBaseAxis (1,0) = sin(-80 * PI / 180.0);
  transformationBaseAxis (1,1) = cos(-80 * PI / 180.0);
  
  Eigen::Matrix4f InverseTransformation = Eigen::Matrix4f::Identity();
  InverseTransformation (1,1) = -1;

  Link1 =  InverseTransformation * transformationBaseAxis * TB0;
  Link2 =  InverseTransformation * transformationBaseAxis * TB0 * T01;
  Link3 =  InverseTransformation * transformationBaseAxis * TB0 * T01 * T12;
  Link4 =  InverseTransformation * transformationBaseAxis * TB0 * T01 * T12 * T23;
  Link5 =  InverseTransformation * transformationBaseAxis * TB0 * T01 * T12 * T23 * T34;
  Link6 =  InverseTransformation * transformationBaseAxis * TB0 * T01 * T12 * T23 * T34 * T45;
  Link_TCP = InverseTransformation * transformationBaseAxis * TB0 * T01 * T12 * T23 * T34 * T45 * T56; 
  
  
  cout << "\n\n-----------------------------";
  cout << "\n   Links XYZ Information";
  cout << "\n-----------------------------";
  cout << "\n Link1 (XYZ): (" << Link1(0,3) << ", " << Link1(1,3) << ", " << Link1(2,3) << ")";
  cout << "\n Link2 (XYZ): (" << Link2(0,3) << ", " << Link2(1,3) << ", " << Link2(2,3) << ")";
  cout << "\n Link3 (XYZ): (" << Link3(0,3) << ", " << Link3(1,3) << ", " << Link3(2,3) << ")";
  cout << "\n Link4 (XYZ): (" << Link4(0,3) << ", " << Link4(1,3) << ", " << Link4(2,3) << ")";
  cout << "\n Link5 (XYZ): (" << Link5(0,3) << ", " << Link5(1,3) << ", " << Link5(2,3) << ")";
  cout << "\n Link6 (XYZ): (" << Link6(0,3) << ", " << Link6(1,3) << ", " << Link6(2,3) << ")";
  cout << "\n Link_TCP (XYZ): (" << Link_TCP(0,3) << ", " << Link_TCP(1,3) << ", " << Link_TCP(2,3) << ")";
  //Complete Matrix
  /*cout << "\n Link1: \n" << Link1;
  cout << "\n Link2: \n" << Link2;
  cout << "\n Link3: \n" << Link3;
  cout << "\n Link4: \n" << Link4;
  cout << "\n Link5: \n" << Link5;
  cout << "\n Link6: \n" << Link6;
  cout << "\n Link_TCP: \n" << Link_TCP;
  cout << "\n-----------------------------\n";*/
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz_data(new pcl::PointCloud<pcl::PointXYZ>);

  cloud_xyz_data->width = 8;
  cloud_xyz_data->height =1;
  cloud_xyz_data->points.resize (cloud_xyz_data->width * cloud_xyz_data->height);
  
  cloud_xyz_data->points[0].x = Link1(0,3);
  cloud_xyz_data->points[0].y = Link1(1,3);
  cloud_xyz_data->points[0].z = Link1(2,3);

  cloud_xyz_data->points[1].x = Link2(0,3);
  cloud_xyz_data->points[1].y = Link2(1,3);
  cloud_xyz_data->points[1].z = Link2(2,3);

  cloud_xyz_data->points[2].x = Link3(0,3);
  cloud_xyz_data->points[2].y = Link3(1,3);
  cloud_xyz_data->points[2].z = Link3(2,3);

  cloud_xyz_data->points[3].x = Link4(0,3);
  cloud_xyz_data->points[3].y = Link4(1,3);
  cloud_xyz_data->points[3].z = Link4(2,3);

  cloud_xyz_data->points[4].x = Link5(0,3);
  cloud_xyz_data->points[4].y = Link5(1,3);
  cloud_xyz_data->points[4].z = Link5(2,3);

  cloud_xyz_data->points[5].x = Link6(0,3);
  cloud_xyz_data->points[5].y = Link6(1,3);
  cloud_xyz_data->points[5].z = Link6(2,3);

  cloud_xyz_data->points[6].x = Link_TCP(0,3);
  cloud_xyz_data->points[6].y = Link_TCP(1,3);
  cloud_xyz_data->points[6].z = Link_TCP(2,3);
  
  sensor_msgs::PointCloud2 jointXYZ;
  pcl::toROSMsg(*cloud_xyz_data, jointXYZ);
  jointXYZ.header = jointMsg.header;
  jointXYZ.header.frame_id = "base_link";
  jointXYZ.header.stamp = ros::Time::now();
  pub_jointXYZ.publish(jointXYZ);
}

//Robot Removal
  //Input:=  Concatinated sensor data (Sensor_1_3 + Sensor_1_4) and Robot joint position(XYZ)
  //Output:= Removed robot in joined point cloud data
void robot_removal(const sensor_msgs::PointCloud2ConstPtr& cloud_msg_robot_removal, const sensor_msgs::PointCloud2ConstPtr& jointxyz)
{
  // Convert to PCL data type
  pcl::PCLPointCloud2 pcl_robot_removal;
  pcl_conversions::toPCL(*cloud_msg_robot_removal, pcl_robot_removal);  
  pcl::fromPCLPointCloud2(pcl_robot_removal, robot_removal_cloud);  

  //proxy
  pcl::PointCloud<pcl::PointXYZ> jointxyz_cloud;
  pcl::PCLPointCloud2 pcl_jointxyz;
  pcl_conversions::toPCL(*jointxyz, pcl_jointxyz);  
  pcl::fromPCLPointCloud2(pcl_jointxyz, jointxyz_cloud); 

  pcl::PointCloud<pcl::PointXYZ>::Ptr robot_removal_final_data (new pcl::PointCloud<pcl::PointXYZ>);

  robot_removal_cloud.header = pcl_conversions::toPCL(cloud_msg_robot_removal->header);
  robot_removal_cloud_data = robot_removal_cloud.makeShared();

  //Kdtree is generated to perform an efficient range search
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud (robot_removal_cloud_data);
  
  // Neighbors within radius search
  std::vector<int> pointIdxRadiusSearch; //to store index of surrounding points 
  std::vector<float> pointRadiusSquaredDistance; // to store distance to surrounding points
  
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
  pcl::PointCloud<pcl::PointXYZ>  robotPoint;
  
  cout << "\nStarted Removing the Robot";
  //Search Point for Links
  int person_count = 0;
  pcl::PointCloud<pcl::PointXYZ>::Ptr robot_removal_data (new pcl::PointCloud<pcl::PointXYZ>);

  float radius [7] = {0.8122059,0.405768383, 0.698290171, 0.140681105, 0.2113056, 0.088044, 0.05};


  for(size_t i=0;i<=6;i++)
  {
    //search point to start
    pcl::PointXYZ searchPoint(jointxyz_cloud.points[i].x,jointxyz_cloud.points[i].y,jointxyz_cloud.points[i].z);
    kdtree.radiusSearch (searchPoint, radius[i], pointIdxRadiusSearch, pointRadiusSquaredDistance);
    for (size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
    {
      /*robot_removal_cloud_data->points[pointIdxRadiusSearch[i]].x = 0;
      robot_removal_cloud_data->points[pointIdxRadiusSearch[i]].y = 0;
      robot_removal_cloud_data->points[pointIdxRadiusSearch[i]].z = 0;*/
      robot_removal_data->points.push_back(robot_removal_cloud_data->points[ pointIdxRadiusSearch[i] ]);
    }    
  }
  //Search Point for mid point of Links
  for(size_t i=0;i<=6;i++)
  {
    //search point to start
    pcl::PointXYZ searchPoint((jointxyz_cloud.points[i].x)/2,(jointxyz_cloud.points[i].y)/2,(jointxyz_cloud.points[i].z)/2);
    kdtree.radiusSearch (searchPoint, radius[i], pointIdxRadiusSearch, pointRadiusSquaredDistance);
    for (size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
    {
      /*robot_removal_cloud_data->points[pointIdxRadiusSearch[i]].x = 0;
      robot_removal_cloud_data->points[pointIdxRadiusSearch[i]].y = 0;
      robot_removal_cloud_data->points[pointIdxRadiusSearch[i]].z = 0;*/
      robot_removal_data->points.push_back(robot_removal_cloud_data->points[ pointIdxRadiusSearch[i] ]);
    }
  }
  //
  //person_count++;
  //new_person(robot_removal_data, person_count);
  //quickHullFunction(robot_removal_data);
  //robot_removal_cloud_data = robotPoint.makeShared();
  cout << "\nRobot Removal Done" ;
  
  pcl::PointXYZ robot_min;
  pcl::PointXYZ robot_max;
 
  pcl::getMinMax3D (*robot_removal_data, robot_min, robot_max);
  
  cout << "\n Robot Minimal and Maximal:: (" << robot_min << ", " << robot_max << ")\n";
 
  pcl::toROSMsg(*robot_removal_data, msg_robot_removal);
  
  msg_robot_removal.header = cloud_msg_robot_removal->header;
  msg_robot_removal.header.stamp = ros::Time::now();
  pub_robot_removal.publish(msg_robot_removal);

}

//RGB Image
  //Input:=  Detected objects in RGB
  //Output:= Centroid of the Detected objects
void rgb_image(const geometry_msgs::PoseStamped& dnn_objects)
{
  float rgb_c_x = dnn_objects.pose.position.x + 0.6; //Transformation from X-Axis :: 0.6mts
  float rgb_c_y = dnn_objects.pose.position.y - 2.95; //Transformation from Y-Axis :: 2.95mts

  cout << "\nCentroid From RGB (X,Y): (" << rgb_c_x << "," << rgb_c_y << ")"; 
}

//Finding the minimum index of ID allocation
  //Input:= Euclidean Distance
  //Output:= Minimum Index value
std::pair<int,int> findIndexOfMin(std::vector<std::vector<float> > distMat)
{
    std::pair<int,int>minIndex;
    float minEl=std::numeric_limits<float>::max();
    for (int i=0; i<distMat.size();i++)
        for(int j=0;j<distMat.at(0).size();j++)
        {
            if( distMat[i][j]<minEl)
            {
                minEl=distMat[i][j];
                minIndex=std::make_pair(i,j);
            }
        }
    return minIndex;
}

//Kalman Prediction
  //Input:= Centroid of Object 
  //Output:= Measurement Matrix for next Update phase
void kalmanPredictionTracking(const std_msgs::Float32MultiArray objectCentroidKFT)
{
  // First predict, to update the internal statePre variable
  std::vector<cv::Mat> pred{KF0.predict(),KF1.predict(),KF2.predict(),KF3.predict(),KF4.predict(),KF5.predict()};
  //cout<<"Pred successfull\n";

  //cv::Point predictPt(prediction.at<float>(0),prediction.at<float>(1));
  // cout<<"Prediction 1 ="<<prediction.at<float>(0)<<","<<prediction.at<float>(1)<<"\n";

  // Get measurements
  // Extract the position of the clusters from the multiArray. To check if the data
  // coming in, check the .z (every third) coordinate and that will be 0.0
  std::vector<geometry_msgs::Point> actualCentroid;//clusterCenters

  int i=0;
  //Actual Centroid
  for (std::vector<float>::const_iterator it=objectCentroidKFT.data.begin();it!=objectCentroidKFT.data.end();it+=3)
  {
    geometry_msgs::Point pt;
    pt.x=*it;
    pt.y=*(it+1);
    pt.z=*(it+2);

    actualCentroid.push_back(pt);
  }
  
  std::vector<geometry_msgs::Point> predictedCentroid;
  i=0;
  //Predicted Centroid
  for (auto it=pred.begin();it!=pred.end();it++)
  {
      geometry_msgs::Point pt;
      pt.x=(*it).at<float>(0);
      pt.y=(*it).at<float>(1);
      pt.z=(*it).at<float>(2);

      predictedCentroid.push_back(pt);
  }
  
  // Find the cluster that is more probable to be belonging to a given KF.
  objID.clear();//Clear the objID vector
  objID.resize(6);//Allocate default elements so that [i] doesnt segfault. Should be done better
  // Copy clusterCentres for modifying it and preventing multiple assignments of the same ID
  std::vector<geometry_msgs::Point> copyOfActualCentroid(actualCentroid);
  std::vector<std::vector<float> > distMat;

  for(int filterN=0; filterN<6; filterN++)
  {
    std::vector<float> distVec;
    for(int n=0;n<6;n++)
    { 
      //Calculating the Euclidean distance for Actual and Precited centroids
      distVec.push_back(euclidean_distance(predictedCentroid[filterN],copyOfActualCentroid[n]));
    }

    distMat.push_back(distVec);
    //cout<<"filterN="<<filterN<<"\n";
  }

  //cout<<"distMat.size()"<<distMat.size()<<"\n";
  //cout<<"distMat[0].size()"<<distMat.at(0).size()<<"\n";
  
  for ( const auto &row : distMat )
  {
    for ( const auto &s : row ) std::cout << s << ' ';
    std::cout << std::endl;
  }

  for(int clusterCount=0;clusterCount<6;clusterCount++)
  {
    // 1. Find min(distMax)==> (i,j);
    std::pair<int,int> minIndex(findIndexOfMin(distMat));
    //cout << "\nReceived minIndex= " << minIndex.first << ", " << minIndex.second << "\n";

    // 2. objID[i]=actualCentroid[j]; counter++
    objID[minIndex.first] = minIndex.second;

    // 3. distMat[i,:]=10000; distMat[:,j]=10000
    distMat[minIndex.first]=std::vector<float>(6,10000.0);// Set the row to a high number.
    for(int row=0;row<distMat.size();row++)//set the column to a high number
    {
        distMat[row][minIndex.second]=10000.0;
    }
    // 4. if(counter<6) got to 1.
    cout<<"clusterCount=" << clusterCount<<"\n";
  }

  // cout<<"Got object IDs"<<"\n";
  //countIDs(objID);// for verif/corner cases

  //display objIDs
  
  cout<<"objID= ";
  for(auto it=objID.begin();it!=objID.end();it++)
  cout<<*it<<" ,";
  cout<<"\n";
  
  visualization_msgs::MarkerArray boundingBox_Array;
  visualization_msgs::MarkerArray boundingText_Array;
  geometry_msgs::PoseArray posePath_Array;
  

  for (int i=0;i<6;i++)
  {
    float r = i%2?1:0;
    float g = i%3?1:0;
    float b = i%4?1:0;
  
    string registry_id = to_string(i);
    //geometry_msgs::Point clusterC(actualCentroid.at(objID[i]));
    geometry_msgs::Point clusterC(predictedCentroid[i]);
    if(clusterC.x != 0 && clusterC.z!=0)
    {
      cout << "\nPredicted Centroid Obtained: " << predictedCentroid[i] << "\n";
      //cout << "\n[Marker] Actual Centroid Obtained (XYZ): (" << clusterC.x << "," << clusterC.y << "," << clusterC.z << ")\n";
      boundingBox_Array.markers.push_back(boundingBox(clusterC.x, clusterC.y, clusterC.z, i, r, g, b));
      boundingText_Array.markers.push_back(boundingText(clusterC.x, clusterC.y, clusterC.z, i, r, g, b, registry_id));
      posePath_Array.poses.push_back(posePath(clusterC.x, clusterC.y, clusterC.z, i));
    }
    
    posePath_Array.header.frame_id = "base_link";
  }

  prevClusterCenters = actualCentroid;

  pub_vis_box.publish(boundingBox_Array);
  pub_vis_text.publish(boundingText_Array);
  pub_vis_posePath.publish(posePath_Array);

  std_msgs::Int32MultiArray obj_id;
  for(auto it=objID.begin();it!=objID.end();it++)
      obj_id.data.push_back(*it);
  // Publish the object IDs
  objID_pub.publish(obj_id);
    // convert actualCentroid from geometry_msgs::Point to floats
  std::vector<std::vector<float> > objectCentroidUpdate;
  for (int i=0;i<6;i++)
  {
    vector<float> pt;
    pt.push_back(actualCentroid[objID[i]].x);
    pt.push_back(actualCentroid[objID[i]].y);
    pt.push_back(actualCentroid[objID[i]].z);
    
    objectCentroidUpdate.push_back(pt);
  }
  //cout<<"objectCentroidUpdate[5][0]="<<objectCentroidUpdate[5].at(0)<<"objectCentroidUpdate[5][1]="<<objectCentroidUpdate[5].at(1)<<"objectCentroidUpdate[5][2]="<<objectCentroidUpdate[5].at(2)<<"\n";
  
  //Assigning Centroid to Measurement Matrix for Update
  float meas0[2]={objectCentroidUpdate[0].at(0),objectCentroidUpdate[0].at(1)};
  float meas1[2]={objectCentroidUpdate[1].at(0),objectCentroidUpdate[1].at(1)};
  float meas2[2]={objectCentroidUpdate[2].at(0),objectCentroidUpdate[2].at(1)};
  float meas3[2]={objectCentroidUpdate[3].at(0),objectCentroidUpdate[3].at(1)};
  float meas4[2]={objectCentroidUpdate[4].at(0),objectCentroidUpdate[4].at(1)};
  float meas5[2]={objectCentroidUpdate[5].at(0),objectCentroidUpdate[5].at(1)};


  // The update phase 
  cv::Mat meas0Mat=cv::Mat(2,1,CV_32F,meas0);
  cv::Mat meas1Mat=cv::Mat(2,1,CV_32F,meas1);
  cv::Mat meas2Mat=cv::Mat(2,1,CV_32F,meas2);
  cv::Mat meas3Mat=cv::Mat(2,1,CV_32F,meas3);
  cv::Mat meas4Mat=cv::Mat(2,1,CV_32F,meas4);
  cv::Mat meas5Mat=cv::Mat(2,1,CV_32F,meas5);

  //cout<<"meas0Mat"<<meas0Mat<<"\n";
  if (!(meas0Mat.at<float>(0,0)==0.0f || meas0Mat.at<float>(1,0)==0.0f))
      Mat estimated0 = KF0.correct(meas0Mat);
  if (!(meas1[0]==0.0f || meas1[1]==0.0f))
      Mat estimated1 = KF1.correct(meas1Mat);
  if (!(meas2[0]==0.0f || meas2[1]==0.0f))
      Mat estimated2 = KF2.correct(meas2Mat);
  if (!(meas3[0]==0.0f || meas3[1]==0.0f))
      Mat estimated3 = KF3.correct(meas3Mat);
  if (!(meas4[0]==0.0f || meas4[1]==0.0f))
      Mat estimated4 = KF4.correct(meas4Mat);
  if (!(meas5[0]==0.0f || meas5[1]==0.0f))
      Mat estimated5 = KF5.correct(meas5Mat);
 
  // Publish the point clouds belonging to each clusters


   // cout<<"estimate="<<estimated.at<float>(0)<<","<<estimated.at<float>(1)<<"\n";
   // Point statePt(estimated.at<float>(0),estimated.at<float>(1));
   //cout<<"DONE KF_TRACKER\n";
}

//Publishing Each Object
  //Input:= Individual Object
  //Output:= Published Object
void publish_cloud(ros::Publisher& pub, pcl::PointCloud<pcl::PointXYZ>::Ptr cluster)
{
  sensor_msgs::PointCloud2::Ptr clustermsg (new sensor_msgs::PointCloud2);
  pcl::toROSMsg (*cluster , *clustermsg);
  clustermsg->header.frame_id = "base_link";
  clustermsg->header.stamp = ros::Time::now();
  pub.publish (*clustermsg);
}

//Cloud Segmentation
  //Input:=  Concatinated Cloud with no Robot
  //Output:= Dynamic Objects (Cloud clkusters)
void cloud_segmentation(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
  // Convert to PCL data type
  pcl::PCLPointCloud2 pcl_pc;
  //Sensor Messages to PointCloud2
  pcl_conversions::toPCL(*cloud_msg, pcl_pc);
  //PointCloud2 to ROS (pcl::PointCloud<pcl::PointXYZ> ) messages for data processing, pcl::PointCloud<pcl::PointXYZ> cloud
  pcl::fromPCLPointCloud2(pcl_pc, cloud);

  cloud.header = pcl_conversions::toPCL(cloud_msg->header);

  //Resolution -> Length of the smallest voxels at lowest octree level /  side length of octree voxels
  pcl::octree::OctreePointCloudChangeDetector<pcl::PointXYZ> octree (RESOLUTION);
  
  cloud_ptr = cloud.makeShared();

  if (first_msg){
    first_msg = false;
    filter_ptr = cloud.makeShared();
  }
  //octree
  octree.setInputCloud(filter_ptr);
  octree.addPointsFromInputCloud();

  // Switch octree buffers: This resets octree but keeps previous tree structure in memory.
  
  octree.switchBuffers();

  octree.setInputCloud(cloud_ptr);
  octree.addPointsFromInputCloud();

  std::vector<int> voxelChange;
  // Leaf Size -> minimum amount of points required within leaf node.
  octree.getPointIndicesFromNewVoxels(voxelChange, LEAF_SIZE);
  //TODO::  Debug messages to be commented out
  ROS_INFO("\nindicies %zd", voxelChange.size());

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud;
  filtered_cloud = (pcl::PointCloud<pcl::PointXYZ>::Ptr) new pcl::PointCloud<pcl::PointXYZ>();

  (filtered_cloud)->header = pcl_conversions::toPCL(cloud_msg->header);

  filtered_cloud->points.reserve(voxelChange.size());

  for (std::vector<int>::iterator it = voxelChange.begin (); it != voxelChange.end (); it++)
    filtered_cloud->points.push_back(cloud_ptr->points[*it]);

  std::vector<pcl::PointIndices> clustered_segments;
  if(filtered_cloud->size() > MIN_FILTERED_CLOUD_SIZE)
  {
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (filtered_cloud);

    // ClusterExtraction: KD tree with Euclidian distance of 4cm
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (CLUSTER_TOLERANCE); // 4cm
    ec.setMinClusterSize (MIN_CLUSTER_SIZE);
    ec.setMaxClusterSize (MAX_CLUSTER_SIZE);
    ec.setSearchMethod (tree);
    ec.setInputCloud (filtered_cloud);
    ec.extract (clustered_segments);
    ROS_INFO("cluster size %zd", clustered_segments.size());
  }

  //Kalman filter Initializing
  if (firstFrame)
  {   
  // Initialize 6 Kalman Filters; Assuming 6 max objects in the dataset. 
  // Could be made generic by creating a Kalman Filter only when a new object is detected  

    float dvx = 0.01f; //1.0
    float dvy = 0.01f;//1.0
    float dx = 1.0f;
    float dy = 1.0f;
    KF0.transitionMatrix = (Mat_<float>(4, 4) << dx,0,1,0,   0,dy,0,1,  0,0,dvx,0,  0,0,0,dvy);
    KF1.transitionMatrix = (Mat_<float>(4, 4) << dx,0,1,0,   0,dy,0,1,  0,0,dvx,0,  0,0,0,dvy);
    KF2.transitionMatrix = (Mat_<float>(4, 4) << dx,0,1,0,   0,dy,0,1,  0,0,dvx,0,  0,0,0,dvy);
    KF3.transitionMatrix = (Mat_<float>(4, 4) << dx,0,1,0,   0,dy,0,1,  0,0,dvx,0,  0,0,0,dvy);
    KF4.transitionMatrix = (Mat_<float>(4, 4) << dx,0,1,0,   0,dy,0,1,  0,0,dvx,0,  0,0,0,dvy);
    KF5.transitionMatrix = (Mat_<float>(4, 4) << dx,0,1,0,   0,dy,0,1,  0,0,dvx,0,  0,0,0,dvy);

    cv::setIdentity(KF0.measurementMatrix);
    cv::setIdentity(KF1.measurementMatrix);
    cv::setIdentity(KF2.measurementMatrix);
    cv::setIdentity(KF3.measurementMatrix);
    cv::setIdentity(KF4.measurementMatrix);
    cv::setIdentity(KF5.measurementMatrix);
    // Process Noise Covariance Matrix Q
    // [ Ex 0  0    0 0    0 ]
    // [ 0  Ey 0    0 0    0 ]
    // [ 0  0  Ev_x 0 0    0 ]
    // [ 0  0  0    1 Ev_y 0 ]
    //// [ 0  0  0    0 1    Ew ]
    //// [ 0  0  0    0 0    Eh ]
    float sigmaP = 0.01;
    float sigmaQ = 0.1;
    setIdentity(KF0.processNoiseCov, Scalar::all(sigmaP));
    setIdentity(KF1.processNoiseCov, Scalar::all(sigmaP));
    setIdentity(KF2.processNoiseCov, Scalar::all(sigmaP));
    setIdentity(KF3.processNoiseCov, Scalar::all(sigmaP));
    setIdentity(KF4.processNoiseCov, Scalar::all(sigmaP));
    setIdentity(KF5.processNoiseCov, Scalar::all(sigmaP));
    // Meas noise cov matrix R
    cv::setIdentity(KF0.measurementNoiseCov, cv::Scalar(sigmaQ));//1e-1
    cv::setIdentity(KF1.measurementNoiseCov, cv::Scalar(sigmaQ));
    cv::setIdentity(KF2.measurementNoiseCov, cv::Scalar(sigmaQ));
    cv::setIdentity(KF3.measurementNoiseCov, cv::Scalar(sigmaQ));
    cv::setIdentity(KF4.measurementNoiseCov, cv::Scalar(sigmaQ));
    cv::setIdentity(KF5.measurementNoiseCov, cv::Scalar(sigmaQ));

    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr > cluster_vec;
      // Cluster centroids
    std::vector<pcl::PointXYZ> clusterCentroids;
    // Iterative segmentation for more than 1 dynamic objects
    // input as segmented point clouds and filterout segments smaller than 150
    int person_count = 0;
    int hullCount = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = clustered_segments.begin (); it !=  clustered_segments.end (); ++it) 
    {
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
      float x = 0.0; float y = 0.0;
      int numPts = 0;
      for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++) 
      {
        cloud_cluster->points.push_back (filtered_cloud->points[*pit]);
        x+= filtered_cloud->points[*pit].x;
        y+= filtered_cloud->points[*pit].y;
        numPts++;

        pcl::toROSMsg(*cloud_cluster, msg_clustered);
          msg_clustered.header = cloud_msg->header;
          msg_clustered.fields[0].count = person_count;
          msg_clustered.header.stamp = ros::Time::now(); 
          pub_clustered.publish(msg_clustered);
          person_count++;
      }
      pcl::PointXYZ centroid;
      centroid.x = x/numPts;
      centroid.y = y/numPts;
      centroid.z = 0.0;
      
      if(cloud_cluster->points.size() > MIN_CLUSTERED_CLOUD_SIZE)
      {
          
          //new_person(cloud_cluster, person_count);
          hullCount++;
          visualisationHull(cloud_cluster,hullCount);

      }

      
      cluster_vec.push_back(cloud_cluster);
      //Get the centroid of the cluster
      clusterCentroids.push_back(centroid);
    }
    while (cluster_vec.size() < 6)
    {
      pcl::PointCloud<pcl::PointXYZ>::Ptr empty_cluster (new pcl::PointCloud<pcl::PointXYZ>);
      empty_cluster->points.push_back(pcl::PointXYZ(0,0,0));
      cluster_vec.push_back(empty_cluster);
    }
    while (clusterCentroids.size()<6)
    {
      pcl::PointXYZ centroid;
      centroid.x = 0.0;
      centroid.y = 0.0;
      centroid.z = 0.0;
      clusterCentroids.push_back(centroid);
    }
      
    // Set initial state
    KF0.statePre.at<float>(0) = clusterCentroids.at(0).x;
    KF0.statePre.at<float>(1) = clusterCentroids.at(0).y;
    KF0.statePre.at<float>(2) = 0;// initial v_x
    KF0.statePre.at<float>(3) = 0;//initial v_y

    // Set initial state
    KF1.statePre.at<float>(0) = clusterCentroids.at(1).x;
    KF1.statePre.at<float>(1) = clusterCentroids.at(1).y;
    KF1.statePre.at<float>(2) = 0;// initial v_x
    KF1.statePre.at<float>(3) = 0;//initial v_y

    // Set initial state
    KF2.statePre.at<float>(0) = clusterCentroids.at(2).x;
    KF2.statePre.at<float>(1) = clusterCentroids.at(2).y;
    KF2.statePre.at<float>(2) = 0;// initial v_x
    KF2.statePre.at<float>(3) = 0;//initial v_y


    // Set initial state
    KF3.statePre.at<float>(0) = clusterCentroids.at(3).x;
    KF3.statePre.at<float>(1) = clusterCentroids.at(3).y;
    KF3.statePre.at<float>(2) = 0;// initial v_x
    KF3.statePre.at<float>(3) = 0;//initial v_y

    // Set initial state
    KF4.statePre.at<float>(0) = clusterCentroids.at(4).x;
    KF4.statePre.at<float>(1) = clusterCentroids.at(4).y;
    KF4.statePre.at<float>(2) = 0;// initial v_x
    KF4.statePre.at<float>(3) = 0;//initial v_y

    // Set initial state
    KF5.statePre.at<float>(0) = clusterCentroids.at(5).x;
    KF5.statePre.at<float>(1) = clusterCentroids.at(5).y;
    KF5.statePre.at<float>(2) = 0;// initial v_x
    KF5.statePre.at<float>(3) = 0;//initial v_y

    firstFrame = false;

    for (int i=0;i<6;i++)
    {
      geometry_msgs::Point pt;
      pt.x = clusterCentroids.at(i).x;
      pt.y = clusterCentroids.at(i).y;
      prevClusterCenters.push_back(pt);
    }
  }
  else
  {
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr > cluster_vec;
      // Cluster centroids
    std::vector<pcl::PointXYZ> clusterCentroids;
    // Iterative segmentation for more than 1 dynamic objects
    // input as segmented point clouds and filterout segments smaller than 150
    int person_count = 0;
    int hullCount = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = clustered_segments.begin (); it != clustered_segments.end (); ++it) 
    {
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
      float x = 0.0; float y = 0.0;
      int numPts = 0;
      for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++) 
      {
          cloud_cluster->points.push_back (filtered_cloud->points[*pit]);
          x+= filtered_cloud->points[*pit].x;
          y+= filtered_cloud->points[*pit].y;
          numPts++;
      }
      pcl::PointXYZ centroid;
      centroid.x = x/numPts;
      centroid.y = y/numPts;
      centroid.z = 0.0;

      //person_count++;
      //new_person(cloud_cluster, person_count);
      hullCount++;
      visualisationHull(cloud_cluster,hullCount);

      pcl::toROSMsg(*cloud_cluster, msg_clustered);
      msg_clustered.header = cloud_msg->header;
      msg_clustered.fields[0].count = person_count;
      msg_clustered.header.stamp = ros::Time::now(); 
      pub_clustered.publish(msg_clustered);
      person_count++;
     
      cluster_vec.push_back(cloud_cluster);

      //Get the centroid of the cluster
      clusterCentroids.push_back(centroid);
    }
    while (cluster_vec.size() < 6)
    {
      pcl::PointCloud<pcl::PointXYZ>::Ptr empty_cluster (new pcl::PointCloud<pcl::PointXYZ>);
      empty_cluster->points.push_back(pcl::PointXYZ(0,0,0));
      cluster_vec.push_back(empty_cluster);
    }
    while (clusterCentroids.size()<6)
    {
      pcl::PointXYZ centroid;
      centroid.x = 0.0;
      centroid.y = 0.0;
      centroid.z = 0.0;
      clusterCentroids.push_back(centroid);
    }
    std_msgs::Float32MultiArray objectCentroid;
    
    for(int i=0;i<6;i++)
    {
      objectCentroid.data.push_back(clusterCentroids.at(i).x);
      objectCentroid.data.push_back(clusterCentroids.at(i).y);
      objectCentroid.data.push_back(clusterCentroids.at(i).z);    
    }
    // cout<<"6 clusters initialized\n";
    //cc_pos.publish(objectCentroid);// Publish cluster mid-points.
    //Kalman Prediction
    kalmanPredictionTracking(objectCentroid);
    int i=0;
    bool publishedCluster[6];
    for(auto it=objID.begin();it!=objID.end();it++)
    { //cout<<"Inside the for loop\n";
      //Publishing Each Object
      switch(i)
      {
          cout<<"Inside the switch case\n";
          case 0: { 
                    publish_cloud(pub_cluster0,cluster_vec[*it]);
                    publishedCluster[i] = true;//Use this flag to publish only once for a given obj ID
                    i++;
                    break;
                  }
          case 1: {
                    publish_cloud(pub_cluster1,cluster_vec[*it]);
                    publishedCluster[i] = true;//Use this flag to publish only once for a given obj ID
                    i++;
                    break;
                  }
          case 2: {
                    publish_cloud(pub_cluster2,cluster_vec[*it]);
                    publishedCluster[i] = true;//Use this flag to publish only once for a given obj ID
                    i++;
                    break;
                  }
          case 3: {
                    publish_cloud(pub_cluster3,cluster_vec[*it]);
                    publishedCluster[i] = true;//Use this flag to publish only once for a given obj ID
                    i++;
                    break;
                  }
          case 4: {
                    publish_cloud(pub_cluster4,cluster_vec[*it]);
                    publishedCluster[i] = true;//Use this flag to publish only once for a given obj ID
                    i++;
                    break;
                  }

          case 5: {
                    publish_cloud(pub_cluster5,cluster_vec[*it]);
                    publishedCluster[i] = true;//Use this flag to publish only once for a given obj ID
                    i++;
                    break;
                  }
          default: break;
      }
    }
  }
    
  //cout << "\nCentroid (from LiDAR 1): ("<< out_x << "," << out_y << ","  <<  out_z << ")\n";
  pcl::toROSMsg(*filtered_cloud, msg_filtered);
  msg_filtered.header = cloud_msg->header;
  pub_filtered.publish(msg_filtered);
}

// Filtering 3x3 Bounding box from the Octomap center points for boundry condition.
  //Input:=  Octomap Centers
  //Output:= Robot Zone (Visualisation:Red, Green and Yellow) 
//ToDo :Sensitivity and Voxel Size to be worked on, 
void octomap_centers(const sensor_msgs::PointCloud2ConstPtr& cloud_msg_octomap_centers)
{
// Convert to PCL data type
  pcl::PCLPointCloud2 pcl_octomap_centers;
  pcl_conversions::toPCL(*cloud_msg_octomap_centers, pcl_octomap_centers);  
  pcl::fromPCLPointCloud2(pcl_octomap_centers, octomap_centers_cloud);  

  octomap_centers_cloud.header = pcl_conversions::toPCL(cloud_msg_octomap_centers->header);
  octomap_centers_cloud_data = octomap_centers_cloud.makeShared();
  
  //Focusing on 3X3 area occupied octomap
  pcl::CropBox<pcl::PointXYZ> cropBoxFilter;
  cropBoxFilter.setInputCloud (octomap_centers_cloud_data);

  //3.0m bounding box
  Eigen::Vector4f min_pt (-3.0f, -3.0f, -3.0f, 1.0f);
  Eigen::Vector4f max_pt (3.0f, 3.0f, 3.0f, 1.0f);
  
  cropBoxFilter.setMin (min_pt);
  cropBoxFilter.setMax (max_pt);

  cropBoxFilter.filter (octomap_centers_output);

  octomap_centers_output_data = octomap_centers_output.makeShared();
  pcl::PointXYZ cloud_min_octo;
  pcl::PointXYZ cloud_max_octo;

  if(octomap_centers_output.size ()>0)
  {
    pcl::getMinMax3D (*octomap_centers_cloud_data, cloud_min_octo, cloud_max_octo);
    ROS_INFO("Number of Points(In the Robot Area)  %zd", octomap_centers_output.size());
    //Robot at low Speed;: 1-3 meter in all directions from robot center point
    if( (((cloud_min_octo.x < 3.0f) && (cloud_min_octo.x >= 1.0f)) || ((cloud_min_octo.x <= -1.0f) && (cloud_min_octo.x > -3.0f))) || (((cloud_min_octo.y < 3.0f) && (cloud_min_octo.y >= 1.0f)) || ((cloud_min_octo.y <= -1.0f) && (cloud_min_octo.y > -3.0f))) )
    {
      pub_vis_robot_bounding.publish(robotBoundingBox("Robot_box_yellow", 1.0, 1.0, 0.0));
    }
    //Robot at NO speed
    else
    {
      pub_vis_robot_bounding.publish(robotBoundingBox("Robot_box_red", 1.0, 0.0, 0.0));
      
    }
  }
  //Robot at Normal Speed
  else
  {
    pub_vis_robot_bounding.publish(robotBoundingBox("Robot_box_green", 0.0, 1.0, 0.0));
  }
  // Todo: Reset not optimal., ...
  if(clearClient.call(srv))
  {
    ROS_INFO("Reset is Done");
  }
  
  pcl::toROSMsg(*octomap_centers_output_data, msg_octomap_centers);
  msg_octomap_centers.header = cloud_msg_octomap_centers->header;
  pub_octomap_centers.publish(msg_octomap_centers);
}

int main (int argc, char** argv)
{
  ros::init (argc, argv, "lidar_body_extraction");
  ros::NodeHandle node;
  
  //reset the octomap
  clearClient = node.serviceClient<std_srvs::Empty>("/octomap_node/reset");
  
  indicesOutput.open ("/home/manish/catkin_ws/src/lidar-human-classification/centroid_data/centroid_data.csv");
  indicesOutput  << "Actual Centroid (X) " << ";" << "Predicted Centroid(X)" << ";" << "Actual Centroid (Y)" << ";" << "Predicted Centroid (Y)" << ";"<< "Actual Centroid(Z)" << "\n";
  
  string topic_scan_1_3,topic_scan_1_4, topic_filtered, topic_clustered, topic_transformed ;
  node.param<string>("scan_topic_1_3", topic_scan_1_3, SCAN_TOPIC_1_3);
  node.param<string>("scan_topic_1_4", topic_scan_1_4, SCAN_TOPIC_1_4);
  node.param<string>("filtered_topic", topic_filtered, FILTERED_TOPIC);
  
  
  pub_filtered = node.advertise<sensor_msgs::PointCloud2>(topic_filtered, 1);
  pub_clustered = node.advertise<sensor_msgs::PointCloud2>("pcl_clustered", 1);
  pub_transformed = node.advertise<sensor_msgs::PointCloud2>("pcl_transformed", 1);
  pub_merged = node.advertise<sensor_msgs::PointCloud2>("pcl_merged", 1);
  pub_merged_transformation = node.advertise<sensor_msgs::PointCloud2>("pcl_merged_transformed", 1);
  
  
  pub_octomap_centers = node.advertise<sensor_msgs::PointCloud2>( "octomap_point_cloud_centers_output", 1 );
  pub_centroid = node.advertise<geometry_msgs::PoseStamped>( "centroid", 1, true);
  pub_pred_centroid = node.advertise<geometry_msgs::PoseStamped>( "predicted_centroid", 1, true );
  pub_robot_removal = node.advertise<sensor_msgs::PointCloud2>("pcl_robot_removal", 1, true);
  pub_jointXYZ = node.advertise<sensor_msgs::PointCloud2>("joint_xyz",1,true);
  // Publisher for the Each object
  pub_cluster0 = node.advertise<sensor_msgs::PointCloud2> ("cluster_0", 1);
  pub_cluster1 = node.advertise<sensor_msgs::PointCloud2> ("cluster_1", 1);
  pub_cluster2 = node.advertise<sensor_msgs::PointCloud2> ("cluster_2", 1);
  pub_cluster3 = node.advertise<sensor_msgs::PointCloud2> ("cluster_3", 1);
  pub_cluster4 = node.advertise<sensor_msgs::PointCloud2> ("cluster_4", 1);
  pub_cluster5 = node.advertise<sensor_msgs::PointCloud2> ("cluster_5", 1);
  objID_pub = node.advertise<std_msgs::Int32MultiArray>("obj_id", 1);
    
  //cc_pos=node.advertise<std_msgs::Float32MultiArray>("objectCentroidKFT",100);//clusterCenter1
  pub_vis_box= node.advertise<visualization_msgs::MarkerArray> ("visualisation_box",1);
  pub_vis_text = node.advertise<visualization_msgs::MarkerArray> ("visualisation_text",1);
  pub_vis_line= node.advertise<visualization_msgs::Marker> ("visualisation_line",1);
  pub_vis_posePath= node.advertise<geometry_msgs::PoseArray> ("visualisation_posePath",1);
  pub_vis_robot_bounding = node.advertise<visualization_msgs::Marker>( "visualisation_robot_bounding_box", 1 );
  pub_convex_hull = node.advertise<visualization_msgs::Marker> ("visualisation_hull",1);
  
  //Transforming Sensor_1_4 (to Sensor_1_3)
  ros::Time start_time_transform = ros::Time::now();
    ros::Subscriber sub_1_4 = node.subscribe(topic_scan_1_4, 1, cloud_transformation);
  ros::Time end_time_transform = ros::Time::now();
  //Syncing Sensor_1_3, Transformed Sensor_1_4
  //Subscribed Synced Data ------- Callback: Concatination (Two LiDAR sensors)
  ros::Time start_time_merged = ros::Time::now();
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_1_3(node, topic_scan_1_3, 1);
    message_filters::Subscriber<sensor_msgs::PointCloud2> sub_1_4_transformed(node, "pcl_transformed", 1);
    // todo Aquib: check ApproxmiateTime Resolution: 1ms or lower or higher
    typedef sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> MySyncPolicy;
    Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), sub_1_3, sub_1_4_transformed);
    sync.registerCallback(boost::bind(cloud_concatenate, _1, _2));
  ros::Time end_time_merged = ros::Time::now();

  //Subscribed: concatinated LiDAR Data ----- Callback: Transformation to Robot Co-ordinates
  ros::Subscriber sub_merged = node.subscribe("pcl_merged", 1, merged_transform);

  //Subscribed: robot Joints ---------- Callback: DH matrix caluculation
  ros::Subscriber sub_robot_joints = node.subscribe("/joint_states", 1, robotJoints);

  //Syncing Transformed LiDAR data, Robot Actual XYZ
  //Subscribed: synced Data ---------- Callback: Robot removal in pcl
  message_filters::Subscriber<sensor_msgs::PointCloud2> sub_merged_transformed(node, "pcl_merged_transformed", 1);
  message_filters::Subscriber<sensor_msgs::PointCloud2> sub_robotJointXYZ(node, "joint_xyz", 1);
  typedef sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> MySyncPolicy2;
  Synchronizer<MySyncPolicy2> sync2(MySyncPolicy2(10), sub_merged_transformed, sub_robotJointXYZ);
  sync2.registerCallback(boost::bind(robot_removal, _1, _2));
  //ros::Subscriber temp_sub = node.subscribe("pcl_merged_transformed", 1, robot_removal);
  
  ros::Subscriber sub_merged_segmentation = node.subscribe("pcl_merged_transformed", 1, cloud_segmentation);
  
  //Syncing: removed robot pcl, clustered object pcl
  //Subscribed: synced data ----------- Callback: Lines and LineStrips
  message_filters::Subscriber<sensor_msgs::PointCloud2> sub_cluster(node, "pcl_clustered", 1);
  message_filters::Subscriber<sensor_msgs::PointCloud2> sub_robot_removal(node, "pcl_robot_removal", 1);
  typedef sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::PointCloud2> MySyncPolicy3;
  Synchronizer<MySyncPolicy3> sync3(MySyncPolicy3(10), sub_cluster, sub_robot_removal);
  sync3.registerCallback(boost::bind(safety_distance, _1, _2));
  
  //Subscribed: Octomap centers ------------ Callback: Robot Zone Visualisation
  ros::Subscriber sub_octomap_centers = node.subscribe("/octomap_point_cloud_centers", 1, octomap_centers);
  
  //ros::Subscriber sub_dnn_detected_centers = node.subscribe("dnn_detect/rgb_centers", 1, rgb_image);
 
  //Testing purposes
  //ros::Subscriber sub_temp = node.subscribe(topic_scan_1_3, 1,robot_removal);
 


  double execution_time_transform = (end_time_transform - start_time_transform).toNSec() * 1e-6;
  double execution_time_merged = (end_time_merged - start_time_merged).toNSec() * 1e-6;
  ROS_INFO_STREAM("Execution time for Transformation (ms): " << execution_time_transform);
  ROS_INFO_STREAM("Execution time for Concatination (ms): " << execution_time_merged);
  
  ROS_INFO_STREAM("Spinning");
  ros::spin();
  indicesOutput.close();

  return 0;
}


