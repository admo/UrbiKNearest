// UKNearest is a module for Urbi to handle K-Nearest algorithm
// implemented in OpenCV
// Copyright (C) 2012 Adam Oleksy

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

// Usage:
// var u = UKNearest.new();
// u.train([0, 0, 0], "black");

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <utility>
#include <limits>
#include <fstream>

#include <urbi/uobject.hh>

#include <boost/scoped_ptr.hpp>
#include <boost/bimap.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/serialization/scoped_ptr.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

#include <ml.h>
#include <cv.h>

using namespace std;
using namespace cv;
using namespace boost;
using namespace boost::bimaps;

class UKNearest: public urbi::UObject {

	typedef bimap<int, string> ClusterMap;
	typedef ClusterMap::value_type Cluster;
	typedef multimap<int, vector<double> > TrainData;

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int /* version */) {
		ar & boost::serialization::make_nvp("maxK", mMaxK);
		ar & boost::serialization::make_nvp("clusterMap", mClusterMap);
		ar & boost::serialization::make_nvp("trainData", mTrainData);
	}

public:
	UKNearest(const string&);

	void init(const int maxK);

	// Load and save data
	bool loadData(const string&);
	bool saveData(const string&) const;

	// Train
	bool train(const vector<double>, const string&);

	// Find
	string find(const vector<double>, const int k) const;

private:
	scoped_ptr<KNearest> mKnn;
	scoped_ptr<ClusterMap> mClusterMap;
	scoped_ptr<TrainData> mTrainData;
	int mMaxK;
};

UKNearest::UKNearest(const string &s) :
		urbi::UObject(s), mKnn(NULL), mClusterMap(NULL), mTrainData(NULL) {
	UBindFunction(UKNearest, init);
}

void UKNearest::init(const int maxK) {
	mKnn.reset(new KNearest);
	mClusterMap.reset(new ClusterMap);
	mTrainData.reset(new TrainData);

	mMaxK = maxK;

	UBindFunctions(UKNearest, loadData, saveData, train, find);
}

bool UKNearest::loadData(const string& filename) {
	ifstream ifs(filename.c_str());

	boost::archive::xml_iarchive ia(ifs);

	ia >> boost::serialization::make_nvp("UKNearest", *this);

	return true;
}

bool UKNearest::saveData(const string& filename) const {

	ofstream ofs(filename.c_str());

	boost::archive::xml_oarchive oa(ofs);
	oa << boost::serialization::make_nvp("UKNearest", *this);

	return true;
}

bool UKNearest::train(const vector<double> data, const string& label) {
	// Sprawdz czy dany klaster istnieje, jesli nie to wstaw z kolejnym ID
	if (mClusterMap->right.count(label) == 0)
		mClusterMap->insert(Cluster(mClusterMap->size(), label));

	// Musi być conajmniej jeden element mClusters
	assert(!mClusterMap->empty());

	int response = mClusterMap->right.at(label);

	mTrainData->insert(make_pair(response, data));

	return mKnn->train(
			Mat(vector<float>(data.begin(), data.end())).t(), // dane - double->float może wyjść inf
			Mat(Size(1, 1), CV_32FC1, Scalar(response)), // odpowiedź
			Mat(),
			false,
			mMaxK,
			mKnn->get_sample_count() != 0); // Czy update
}

string UKNearest::find(const vector<double> data, const int k) const {
	// numeric_cast??
	int response = static_cast<int>(mKnn->find_nearest(
			Mat(vector<float>(data.begin(), data.end())).t(), k));

	// Wynik klasyfikacji musi być zawarty w mapie
	assert(mClusterMap->left.count(response));

	return mClusterMap->left.at(response);
}

UStart(UKNearest);
