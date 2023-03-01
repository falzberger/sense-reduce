#!/bin/bash

# Download data from ZAMG API, cf. https://dataset.api.hub.zamg.ac.at/v1/docs/index.html
# Due to server limitations, we download the data in three parts for each station

curl -X 'GET' \
  -o 'vienna_20100101_20131231.csv' \
  'https://dataset.api.hub.zamg.ac.at/v1/station/historical/klima-v1-10min?parameters=DD%2CDD_FLAG%2CFFAM%2CFFAM_FLAG%2CP%2CP0%2CP0_FLAG%2CP_FLAG%2CRF%2CRF_FLAG%2CRR%2CRR_FLAG%2CSO%2CSO_FLAG%2CTL%2CTL_FLAG%2CTP%2CTP_FLAG&start=2010-01-01T00%3A00&end=2013-12-31T23%3A59&station_ids=5925&output_format=csv' \
  -H 'accept: text/csv'

curl -X 'GET' \
  -o 'vienna_20140101_20171231.csv' \
  'https://dataset.api.hub.zamg.ac.at/v1/station/historical/klima-v1-10min?parameters=DD%2CDD_FLAG%2CFFAM%2CFFAM_FLAG%2CP%2CP0%2CP0_FLAG%2CP_FLAG%2CRF%2CRF_FLAG%2CRR%2CRR_FLAG%2CSO%2CSO_FLAG%2CTL%2CTL_FLAG%2CTP%2CTP_FLAG&start=2014-01-01T00%3A00&end=2017-12-31T23%3A59&station_ids=5925&output_format=csv' \
  -H 'accept: text/csv'

curl -X 'GET' \
  -o 'vienna_20180101_20211231.csv' \
  'https://dataset.api.hub.zamg.ac.at/v1/station/historical/klima-v1-10min?parameters=DD%2CDD_FLAG%2CFFAM%2CFFAM_FLAG%2CP%2CP0%2CP0_FLAG%2CP_FLAG%2CRF%2CRF_FLAG%2CRR%2CRR_FLAG%2CSO%2CSO_FLAG%2CTL%2CTL_FLAG%2CTP%2CTP_FLAG&start=2018-01-01T00%3A00&end=2021-12-31T23%3A59&station_ids=5925&output_format=csv' \
  -H 'accept: text/csv'
  
curl -X 'GET' \
  -o 'linz_20100101_20131231.csv' \
  'https://dataset.api.hub.zamg.ac.at/v1/station/historical/klima-v1-10min?parameters=DD%2CDD_FLAG%2CFFAM%2CFFAM_FLAG%2CP%2CP0%2CP0_FLAG%2CP_FLAG%2CRF%2CRF_FLAG%2CRR%2CRR_FLAG%2CSO%2CSO_FLAG%2CTL%2CTL_FLAG%2CTP%2CTP_FLAG&start=2010-01-01T00%3A00&end=2013-12-31T23%3A59&station_ids=3202&output_format=csv' \
  -H 'accept: text/csv'

curl -X 'GET' \
  -o 'linz_20140101_20171231.csv' \
  'https://dataset.api.hub.zamg.ac.at/v1/station/historical/klima-v1-10min?parameters=DD%2CDD_FLAG%2CFFAM%2CFFAM_FLAG%2CP%2CP0%2CP0_FLAG%2CP_FLAG%2CRF%2CRF_FLAG%2CRR%2CRR_FLAG%2CSO%2CSO_FLAG%2CTL%2CTL_FLAG%2CTP%2CTP_FLAG&start=2014-01-01T00%3A00&end=2017-12-31T23%3A59&station_ids=3202&output_format=csv' \
  -H 'accept: text/csv'

curl -X 'GET' \
  -o 'linz_20180101_20211231.csv' \
  'https://dataset.api.hub.zamg.ac.at/v1/station/historical/klima-v1-10min?parameters=DD%2CDD_FLAG%2CFFAM%2CFFAM_FLAG%2CP%2CP0%2CP0_FLAG%2CP_FLAG%2CRF%2CRF_FLAG%2CRR%2CRR_FLAG%2CSO%2CSO_FLAG%2CTL%2CTL_FLAG%2CTP%2CTP_FLAG&start=2018-01-01T00%3A00&end=2021-12-31T23%3A59&station_ids=3202&output_format=csv' \
  -H 'accept: text/csv'