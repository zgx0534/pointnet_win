# -*- coding: UTF-8 -*-
from pclpy import pcl
import tensorflow as tf

def preHandle(cloud_batch):
    batchSize=cloud_batch.shape[0]
    pointNum=cloud_batch.shape[1]
    POINTNUM_OUT=1024

    # 定义一个batch个pcl对象数组
    pclObj_arr=[]
    for i in range(batchSize):
        obj = pcl.PointCloud.PointXYZ()
        for j in range(pointNum):
            point = pcl.point_types.PointXYZ()
            point.x = cloud_batch[i][j][0]
            point.y = cloud_batch[i][j][1]
            point.z = cloud_batch[i][j][2]
            obj.push_back(point)
        pclObj_arr.append(obj)

    # 将对象去噪
    for i in range(batchSize):
        obj=pclObj_arr[i]
        obj_ed = pcl.PointCloud.PointXYZ()
        sor = pcl.filters.StatisticalOutlierRemoval.PointXYZ()
        sor.setInputCloud(obj)
        sor.setMeanK(50)
        sor.setStddevMulThresh(1.0)
        sor.filter(obj_ed)
        pclObj_arr[i]=obj_ed

    # 将对象下采样
    for i in range(batchSize):
        obj=pclObj_arr[i]
        obj_ed = pcl.PointCloud.PointXYZ()
        randomSample=pcl.filters.RandomSample.PointXYZ()
        randomSample.setInputCloud(obj)
        randomSample.setSample(POINTNUM_OUT)
        randomSample.filter(obj_ed)
        pclObj_arr[i]=obj_ed

    # 将对象返回到tensor
    data=[]
    for pointcloud in pclObj_arr:
        for index in range(POINTNUM_OUT):
            data.append(pointcloud.at(index).x)
            data.append(pointcloud.at(index).y)
            data.append(pointcloud.at(index).z)
    tsr=tf.convert_to_tensor(data)
    tsr_res=tf.reshape(tsr,[batchSize,POINTNUM_OUT,3])
    return tsr_res


