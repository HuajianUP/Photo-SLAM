/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ATLAS_H
#define ATLAS_H

#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "GeometricCamera.h"
#include "Pinhole.h"
#include "KannalaBrandt8.h"

#include <set>
#include <unordered_set>
#include <queue>
#include <mutex>
#include <tuple>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>


namespace ORB_SLAM3
{
class Viewer;
class Map;
class MapPoint;
class KeyFrame;
class KeyFrameDatabase;
class Frame;
class KannalaBrandt8;
class Pinhole;

//BOOST_CLASS_EXPORT_GUID(Pinhole, "Pinhole")
//BOOST_CLASS_EXPORT_GUID(KannalaBrandt8, "KannalaBrandt8")

class MappingOperation
{
public:
    enum OprType{
        LocalMappingBA = 1,
        LoopClosingBA = 2,
        ScaleRefinement = 3
    };

private:
    MappingOperation(
        const MappingOperation &opr,
        const std::lock_guard<std::mutex> &,
        const std::lock_guard<std::mutex> &)
        : mvAssociatedKeyFrames(std::move(opr.mvAssociatedKeyFrames)),
          mvAssociatedMapPoints(std::move(opr.mvAssociatedMapPoints)),
          meOperationType(opr.meOperationType),
          mfScale(opr.mfScale),
          mT(opr.mT)
    {}

public:
    MappingOperation(
        OprType type,
        const float scale = 1.0f,
        const Sophus::SE3f T = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()),
        const std::size_t nKFs = 0UL,
        const std::size_t nMPs = 0UL)
        : meOperationType(type),
          mfScale(scale),
          mT(T)
    {
        mvAssociatedKeyFrames.reserve(nKFs);
        int length = nMPs * 3;
        std::get<0>(mvAssociatedMapPoints).reserve(length);
        std::get<1>(mvAssociatedMapPoints).reserve(length);
    }

    MappingOperation(const MappingOperation &opr)
        : MappingOperation(
            opr,
            std::lock_guard<std::mutex>(opr.mMutexKeyFrames),
            std::lock_guard<std::mutex>(opr.mMutexMapPoints))
    {}

public:
    void reserveKeyFrames(const std::size_t nKFs)
    {
        mvAssociatedKeyFrames.reserve(nKFs);
    }

    void addKeyFrame(KeyFrame* pKF, bool isLoopClosureKF = false)
    {
        std::unique_lock<std::mutex> lock(mMutexKeyFrames);
        std::vector<float> pixels;
        std::vector<float> pointsLocal;
        pKF->GetKeypointInfo(pixels, pointsLocal);
        mvAssociatedKeyFrames.emplace_back(
            std::make_tuple(
                pKF->mnId,
                pKF->mpCamera->GetId(),
                pKF->GetPose(),
                pKF->imgLeftRGB.clone(),
                isLoopClosureKF,
                pKF->imgAuxiliary,
                pixels,
                pointsLocal,
                pKF->mNameFile));
    }

    std::vector<std::tuple<
        unsigned long,
        unsigned long,
        Sophus::SE3f,
        cv::Mat,
        bool,
        cv::Mat,
        std::vector<float>,
        std::vector<float>,
        std::string>>&
    associatedKeyFrames() { return mvAssociatedKeyFrames; }

    void reserveMapPoints(const std::size_t nMPs)
    {
        int length = nMPs * 3;
        std::get<0>(mvAssociatedMapPoints).reserve(length);
        std::get<1>(mvAssociatedMapPoints).reserve(length);
    }

    void addMapPoint(MapPoint* pMP)
    {
        std::unique_lock<std::mutex> lock(mMutexMapPoints);
        auto pt = pMP->GetWorldPos();
        std::get<0>(mvAssociatedMapPoints).emplace_back(pt.x());
        std::get<0>(mvAssociatedMapPoints).emplace_back(pt.y());
        std::get<0>(mvAssociatedMapPoints).emplace_back(pt.z());
        auto color = pMP->GetColorRGB();
        std::get<1>(mvAssociatedMapPoints).emplace_back(color.x());
        std::get<1>(mvAssociatedMapPoints).emplace_back(color.y());
        std::get<1>(mvAssociatedMapPoints).emplace_back(color.z());
    }

    std::tuple<std::vector<float/*pos*/>, std::vector<float/*color*/>>&
    associatedMapPoints() { return mvAssociatedMapPoints; }

public:
    // Type
    OprType meOperationType;

    // Data
    float mfScale; ///<  ScaleRefinement: global; LoopClosingBA: only for visible; LocalMappingBA: meaningless
    Sophus::SE3f mT;

protected:
    // Data
    std::tuple<std::vector<float/*pos*/>,
               std::vector<float/*color*/>> mvAssociatedMapPoints;

    std::vector<std::tuple<
        unsigned long/*Id*/,
        unsigned long/*CameraId*/,
        Sophus::SE3f/*pose*/,
        cv::Mat/*image*/,
        bool/*isLoopClosure*/,
        cv::Mat/*auxiliaryImage*/,
        std::vector<float>/*keypoints pixel*/,
        std::vector<float>/*keypoints local 3D*/,
        std::string/*main image file name*/>> mvAssociatedKeyFrames;

    // Mutex
    mutable std::mutex mMutexMapPoints;
    mutable std::mutex mMutexKeyFrames;
};

class Atlas
{
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar.template register_type<Pinhole>();
        ar.template register_type<KannalaBrandt8>();

        // Save/load a set structure, the set structure is broken in libboost 1.58 for ubuntu 16.04, a vector is serializated
        //ar & mspMaps;
        ar & mvpBackupMaps;
        ar & mvpCameras;
        // Need to save/load the static Id from Frame, KeyFrame, MapPoint and Map
        ar & Map::nNextId;
        ar & Frame::nNextId;
        ar & KeyFrame::nNextId;
        ar & MapPoint::nNextId;
        ar & GeometricCamera::nNextId;
        ar & mnLastInitKFidMap;
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Atlas();
    Atlas(int initKFid); // When its initialization the first map is created
    ~Atlas();

    void CreateNewMap();
    void ChangeMap(Map* pMap);

    unsigned long int GetLastInitKFid();

    void SetViewer(Viewer* pViewer);

    // Method for change components in the current map
    void AddKeyFrame(KeyFrame* pKF);
    void AddMapPoint(MapPoint* pMP);
    //void EraseMapPoint(MapPoint* pMP);
    //void EraseKeyFrame(KeyFrame* pKF);

    GeometricCamera* AddCamera(GeometricCamera* pCam);
    std::vector<GeometricCamera*> GetAllCameras();

    /* All methods without Map pointer work on current map */
    void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);
    void InformNewBigChange();
    int GetLastBigChangeIdx();

    long unsigned int MapPointsInMap();
    long unsigned KeyFramesInMap();

    // Method for get data in current map
    std::vector<KeyFrame*> GetAllKeyFrames();
    std::vector<MapPoint*> GetAllMapPoints();
    std::vector<MapPoint*> GetReferenceMapPoints();
    std::unordered_set<unsigned long> GetCurrentKeyFrameIds();

    vector<Map*> GetAllMaps();

    int CountMaps();

    void clearMap();

    void clearAtlas();

    Map* GetCurrentMap();

    void SetMapBad(Map* pMap);
    void RemoveBadMaps();

    bool isInertial();
    void SetInertialSensor();
    void SetImuInitialized();
    bool isImuInitialized();

    // Function for garantee the correction of serialization of this object
    void PreSave();
    void PostLoad();

    map<long unsigned int, KeyFrame*> GetAtlasKeyframes();

    void SetKeyFrameDababase(KeyFrameDatabase* pKFDB);
    KeyFrameDatabase* GetKeyFrameDatabase();

    void SetORBVocabulary(ORBVocabulary* pORBVoc);
    ORBVocabulary* GetORBVocabulary();

    long unsigned int GetNumLivedKF();

    long unsigned int GetNumLivedMP();

    void pushMappingOperation(MappingOperation opr);
    MappingOperation getAndPopMappingOperation();
    bool hasMappingOperation();
    void clearMappingOperation();

protected:
    std::queue<MappingOperation> mqMappingOperations;

    std::set<Map*> mspMaps;
    std::set<Map*> mspBadMaps;
    // Its necessary change the container from set to vector because libboost 1.58 and Ubuntu 16.04 have an error with this cointainer
    std::vector<Map*> mvpBackupMaps;

    Map* mpCurrentMap;

    std::vector<GeometricCamera*> mvpCameras;

    unsigned long int mnLastInitKFidMap;

    Viewer* mpViewer;
    bool mHasViewer;

    // Class references for the map reconstruction from the save file
    KeyFrameDatabase* mpKeyFrameDB;
    ORBVocabulary* mpORBVocabulary;

    // Mutex
    std::mutex mMutexAtlas;
    std::mutex mMutexMappingOperations;

}; // class Atlas

} // namespace ORB_SLAM3

#endif // ATLAS_H
