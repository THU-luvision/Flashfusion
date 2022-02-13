
#ifndef SPARSE_MATCH_H
#define SPARSE_MATCH_H


#include "mild.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <immintrin.h>
#include <list>
#include <bitset>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <malloc.h>
namespace MILD
{
	typedef gstd::lightweight_vector<unsigned short> sparse_match_entry;

	// fast frame match based on binary features, using MILD
	class SparseMatcher
	{
	public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        SparseMatcher(int feature_type = FEATURE_TYPE_ORB, int input_hash_table_num = 32, int input_depth_level = 0, float input_distance_threshold = 50)
		{
            switch (feature_type)
            {
                case FEATURE_TYPE_ORB:
                {
                     descriptor_length = ORB_DESCRIPTOR_LEN;
                     break;
                }
                case FEATURE_TYPE_BRISK:
                {
                     descriptor_length = BRISK_DESCRIPTOR_LEN;
                     break;
                }
                default:
                {
                     std::cout << "unknown descriptor" << std::endl;
                     return;
                }
            }
            descriptor_type = feature_type;
            depth_level = input_depth_level;
            bits_per_substring = (int)(descriptor_length / input_hash_table_num);
            if (bits_per_substring > sizeof(size_t)* 8)
            {
                std::cout << "substring too large !, invalied" << std::endl;
                return;
            }
            hash_table_num = input_hash_table_num;
            entry_num_per_hash_table = pow(float(2), float(bits_per_substring));
            buckets_num = entry_num_per_hash_table * hash_table_num;
            distance_threshold = input_distance_threshold;

            features_buffer = std::vector<sparse_match_entry>(entry_num_per_hash_table * hash_table_num);
            for (int i = 0; i < entry_num_per_hash_table * hash_table_num; i++)
            {
                features_buffer[i].clear();
            }
            statistics_num_distance_calculation = 0;
		}
        ~SparseMatcher()
        {

        }

        void clear_buffer()
        {
            for (int i = 0; i < entry_num_per_hash_table * hash_table_num; i++)
            {
                features_buffer[i].clear();
            }
        }

		void displayParameters()
        {
                std::cout << "parameters: " << std::endl
                  << "unit length :	" << descriptor_length << std::endl
                  << "chunk_num_per_unit :	" << depth_level << std::endl
                  << "bits_per_substring :	" << bits_per_substring << std::endl
                  << "hash_table_num :	" << hash_table_num << std::endl
                  << "entry_num_per_hash_table :	" << entry_num_per_hash_table << std::endl
                  << "buckets_num :	" << buckets_num << std::endl;
        }
        void displayStatistics()
        {
            std::cout << "num of distance calculation : " << statistics_num_distance_calculation << std::endl;
        }


    void train(cv::Mat desc)
    {
        clear_buffer();
        features_descriptor = (uint64_t *)desc.data;
		int feature_num = desc.rows;
		if (descriptor_type == FEATURE_TYPE_ORB)
		{
			int descriptor_length = desc.cols * 8;
			if (descriptor_length != descriptor_length)
			{
                std::cout << "error ! feature descriptor length doesn't match" << std::endl;
			}
		}
		std::vector<unsigned long> hash_entry_index = std::vector<unsigned long>(hash_table_num);
		for (unsigned short feature_idx = 0; feature_idx < feature_num; feature_idx++)
		{
			unsigned int *data = desc.ptr<unsigned int>(feature_idx);
			multi_index_hashing(hash_entry_index, data, hash_table_num, bits_per_substring);
			for (int hash_table_id = 0; hash_table_id < hash_table_num; hash_table_id++)
			{
				int entry_pos = hash_table_id*entry_num_per_hash_table + hash_entry_index[hash_table_id];
				features_buffer[entry_pos].push_back(feature_idx);
			}
		}
	}			
    void search(cv::Mat desc, std::vector<cv::DMatch> &matches)
    {
		int feature_num = desc.rows;
		matches.clear();
		matches.reserve(feature_num);
		std::vector<unsigned long> hash_entry_index = std::vector<unsigned long>(hash_table_num);
		for (unsigned short feature_idx = 0; feature_idx < feature_num; feature_idx++)
		{
			unsigned int *data = desc.ptr<unsigned int>(feature_idx);
			unsigned short min_distance = 256;
			unsigned short best_corr_fid = 0;
            uint64_t * f1 = desc.ptr<uint64_t>(feature_idx);
			multi_index_hashing(hash_entry_index, data, hash_table_num, bits_per_substring);
			for (int hash_table_id = 0; hash_table_id < hash_table_num; hash_table_id++)
			{
				unsigned long entry_idx = hash_table_id*entry_num_per_hash_table + hash_entry_index[hash_table_id];
				int conflict_num = features_buffer[entry_idx].size();
				for (int i = 0; i < conflict_num; i++)
				{
					unsigned int feature_index = features_buffer[entry_idx][i];
                    uint64_t * f2 = features_descriptor + feature_index * 4;
					int hamming_distance = calculate_hamming_distance_256bit(f1, f2);
					if (hamming_distance < min_distance)
					{
						min_distance = hamming_distance;
						best_corr_fid = feature_index;
					}
				}
			}
            cv::DMatch m;
            m.queryIdx = feature_idx;
            m.trainIdx = best_corr_fid;
            m.distance = min_distance;
            matches.push_back(m);
		}
	}
		// special case, when hash table num equal to 32
    void train_8(cv::Mat desc)
    {
        clear_buffer();
        features_descriptor = (uint64_t *)desc.data;
		int feature_num = desc.rows;
		if (descriptor_type == FEATURE_TYPE_ORB)
		{
			int descriptor_length = desc.cols * 8;
			if (descriptor_length != descriptor_length)
			{
                std::cout << "error ! feature descriptor length doesn't match" << std::endl;
			}
		}
		for (unsigned short feature_idx = 0; feature_idx < feature_num; feature_idx++)
		{
			unsigned char *data = (unsigned char *)features_descriptor + 32 * feature_idx;
			for (int hash_table_id = 0; hash_table_id < hash_table_num; hash_table_id++)
			{
				int entry_pos = hash_table_id*entry_num_per_hash_table + data[hash_table_id];
				features_buffer[entry_pos].push_back(feature_idx);
			}
		}
	}
    void search_8(cv::Mat desc, std::vector<cv::DMatch> &matches, int hamming_threshold)
    {
		int feature_num = desc.rows;
		matches.clear();
		matches.reserve(feature_num);
		for (unsigned short feature_idx = 0; feature_idx < feature_num; feature_idx++)
		{
			unsigned char *data = (unsigned char *)desc.data + feature_idx * 32;
            uint64_t * f1 = (uint64_t *)desc.data + feature_idx * 4;
			unsigned short min_distance = 256;
			unsigned short best_corr_fid = 0;
			for (int hash_table_id = 0; hash_table_id < hash_table_num; hash_table_id++)
			{
				unsigned long entry_idx = hash_table_id*entry_num_per_hash_table + data[hash_table_id];
				int conflict_num = features_buffer[entry_idx].size();
				for (int i = 0; i < conflict_num; i++)
				{
					unsigned int feature_index = features_buffer[entry_idx][i];
                    uint64_t * f2 = features_descriptor + feature_index * 4;
					int hamming_distance = calculate_hamming_distance_256bit(f1, f2);
					if (hamming_distance < min_distance)
					{
						min_distance = hamming_distance;
						best_corr_fid = feature_index;
					}
				}
			}
              if(min_distance < hamming_threshold)
              {

                    cv::DMatch m;
                    m.queryIdx = feature_idx;
                    m.trainIdx = best_corr_fid;
                    m.distance = min_distance;
                    matches.push_back(m);
              }
        }

	}

		// search feature within a range
    void search_8_with_range(cv::Mat desc,
                             std::vector<cv::DMatch> &matches,
                             std::vector<cv::KeyPoint> &train_features,
                             std::vector<cv::KeyPoint> &query_features,
                             float range,
                             int hamming_distance_threshold)
    {
		int feature_num = desc.rows;
    float euclidean_distance_threshold = range*range;
		matches.clear();
		for (unsigned short feature_idx = 0; feature_idx < feature_num; feature_idx++)
		{
			unsigned char *data = (unsigned char *)desc.data + feature_idx * 32;
      uint64_t * f1 = (uint64_t *)desc.data + feature_idx * 4;
      cv::Point2f query_feature = query_features[feature_idx].pt;
			unsigned short min_distance = 256;
			unsigned short best_corr_fid = 0;
			for (int hash_table_id = 0; hash_table_id < hash_table_num; hash_table_id++)
			{
				unsigned long entry_idx = hash_table_id*entry_num_per_hash_table + data[hash_table_id];
				int conflict_num = features_buffer[entry_idx].size();
				for (int i = 0; i < conflict_num; i++)
				{
					unsigned int train_feature_index = features_buffer[entry_idx][i];
          uint64_t * f2 = features_descriptor + train_feature_index * 4;
          cv::Point2f train_feature = train_features[train_feature_index].pt;
					if ((train_feature.x - query_feature.x)*(train_feature.x - query_feature.x) +
						(train_feature.y - query_feature.y)*(train_feature.y - query_feature.y)
            < euclidean_distance_threshold)
					{
						int hamming_distance = calculate_hamming_distance_256bit(f1, f2);
						if (hamming_distance < min_distance)
						{
							min_distance = hamming_distance;
							best_corr_fid = train_feature_index;
						}

					}
					
				}
			}
      if (min_distance <= hamming_distance_threshold)
			{
        cv::DMatch m;
				m.queryIdx = feature_idx;
				m.trainIdx = best_corr_fid;
				m.distance = min_distance;
				matches.push_back(m);
			}

		}

	}
        void BFMatch(cv::Mat d1, cv::Mat d2, std::vector<cv::DMatch> &matches)
        {
            int feature_f1_num = d1.rows;
            int feature_f2_num = d2.rows;
            std::cout << "f1 " << feature_f1_num << std::endl << "f2 " << feature_f2_num << std::endl;
            unsigned short * delta_distribution = new unsigned short[feature_f1_num * feature_f2_num];
            uint64_t current_descriptor[4];
            for (int f1 = 0; f1 < feature_f1_num; f1++)
            {
                uint64_t *feature1_ptr = (d1.ptr<uint64_t>(f1));
                int best_corr_fid = 0;
                int min_distance = 256;
                for (int f2 = 0; f2 < feature_f2_num; f2++)
                {
                    int hamming_distance = calculate_hamming_distance_256bit(feature1_ptr, d2.ptr<uint64_t>(f2));
                    delta_distribution[f1 * feature_f2_num + f2] = hamming_distance;
                    if (hamming_distance < min_distance)
                    {
                        min_distance = hamming_distance;
                        best_corr_fid = f2;
                    }
                }

                if (min_distance <= distance_threshold)
                {
                    cv::DMatch m;
                    m.queryIdx = f1;
                    m.trainIdx = best_corr_fid;
                    m.distance = min_distance;
                    matches.push_back(m);
                }
            }

            FILE * fp = fopen("data_file.bin", "wb+");
            fwrite(delta_distribution, sizeof(unsigned short), feature_f1_num * feature_f2_num, fp);
            fclose(fp);
        }
        int calculate_hamming_distance_256bit(uint64_t * f_1, uint64_t * f_2)
        {
            unsigned long long * f1 = (unsigned long long *)f_1;
            unsigned long long * f2 = (unsigned long long *)f_2;
            long long int hamming_distance = (_mm_popcnt_u64(*(f1) ^ *(f2)) +
                _mm_popcnt_u64(*(f1 + 1) ^ *(f2 + 1)) +
                _mm_popcnt_u64(*(f1 + 2) ^ *(f2 + 2)) +
                _mm_popcnt_u64(*(f1 + 3) ^ *(f2 + 3)));
            #if DEBUG_MODE_MILD
                statistics_num_distance_calculation++;
            #endif
            return (int)hamming_distance;
        }
        uint64_t *features_descriptor;
        std::vector<sparse_match_entry>				features_buffer;
        int statistics_num_distance_calculation;	// for statistic information

    private:
		int	descriptor_type;						// ORB feature or not
		unsigned int descriptor_length;				// feature descriptor length
		unsigned int bits_per_substring;			// substring length
		unsigned int hash_table_num;				// basic parameters of MIH, hash table num
		unsigned int depth_level;					// depth of MIH
		unsigned int entry_num_per_hash_table;		// entry num per hash table
		unsigned int buckets_num;					// total entry num
		float distance_threshold;
	};

}

#endif
