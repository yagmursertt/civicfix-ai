#!/usr/bin/env bash
# Downloads the 2025 NYC 311 infrastructure complaints (~50 MB) into this folder
set -e; cd "$(dirname "$0")"
W="created_date between '2025-01-01T00:00:00' and '2025-12-31T23:59:59' AND complaint_type in('Street Condition','Sidewalk Condition','Sewer','Root/Sewer/Sidewalk Condition')"
S="unique_key,created_date,closed_date,agency,complaint_type,descriptor,status,resolution_description,borough,incident_zip,latitude,longitude,location_type,community_board"
for off in 0 50000 100000; do
  curl -s -G "https://data.cityofnewyork.us/resource/erm2-nwe9.csv" --data-urlencode "\$select=$S" --data-urlencode "\$where=$W" \
    --data-urlencode "\$order=unique_key" --data-urlencode "\$limit=50000" --data-urlencode "\$offset=$off" -o part_$off.csv
done
python3 -c "import pandas as pd; pd.concat([pd.read_csv(f'part_{o}.csv') for o in [0,50000,100000]]).drop_duplicates('unique_key').to_csv('nyc311_infra_2025.csv', index=False)"
rm part_*.csv; echo "saved nyc311_infra_2025.csv"
