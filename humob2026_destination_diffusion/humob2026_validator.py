# Submission validator for HuMob 2026
#
# Usage: python humob2026_validator.py <submission_file_path>

import sys
import ast
import re

LAT_MIN = 1
LAT_MAX = 70
LON_MIN = 1
LON_MAX = 100

feb_set = {"202402{:02d}".format(i) for i in range(1, 29 + 1)}
mar_set = {"202403{:02d}".format(i) for i in range(1, 31 + 1)}
feb_set.remove("20240202")
mar_set.remove("20240305")
date_str_set = feb_set.union(mar_set)

def error(message):
    print(message)
    sys.exit(1)

def in_valid_area(lat, lon):
    if lat >= LAT_MIN and lat <= LAT_MAX and lon >= LON_MIN and lon <= LON_MAX:
        return True
    if lat == -1 and lon == -1:
        return True
    return False

def parse_od_matrix(line_num, od_matrix_str):
    try:
        od_matrix = ast.literal_eval(od_matrix_str)
    except:
        error("Error at line {}: Could not parse the OD matrix in the second column.".format(line_num))

    if type(od_matrix) != dict:
        error("Error at line {}: The OD matrix in the second column must be a dict.".format(line_num))

    if len(od_matrix) == 0:
        error("Error at line {}: The OD matrix is empty.".format(line_num))

    for orig_key, v in od_matrix.items():
        if not re.match(r"^-?\d+_-?\d+$", orig_key):
            error("Error at line {}: Origin key `{}` has an invalid format.".format(line_num, orig_key))

        lat_str, lon_str = orig_key.split("_")
        lat = int(lat_str)
        lon = int(lon_str)
        if not in_valid_area(lat, lon):
            error("Error at line {}: The origin {} is outside the valid area.".format(line_num, orig_key))

        if type(v) != dict:
            error("Error at line {}: The value for key `{}` must be a dict.".format(line_num, orig_key))

        if len(v) == 0:
            error("Error at line {}: No destinations for origin {}.".format(line_num, orig_key))

        for dest_key, weight_str in v.items():
            if not re.match(r"^-?\d+_-?\d+$", dest_key):
                error("Error at line {}: Destination key `{}` under origin `{}` has an invalid format.".format(line_num, dest_key, orig_key))

            lat_str, lon_str = dest_key.split("_")
            lat = int(lat_str)
            lon = int(lon_str)
            if not in_valid_area(lat, lon):
                error("Error at line {}: Destination {} under origin {} is outside the valid area.".format(line_num, dest_key, orig_key))
            
            try:
                weight = float(weight_str)
            except:
                error("Error at line {}: The value {} for edge {} -> {} is not numeric.".format(line_num, weight_str, orig_key, dest_key))

            if weight < 0:
                error("Error at line {}: The value {} for edge {} -> {} is negative.".format(line_num, weight_str, orig_key, dest_key))

    return od_matrix

def main():
    argv = sys.argv
    if len(argv) != 2:
        error("Usage: python humob2026_validator.py <submission_file_path>")

    fpath = argv[1]
    lines = list()
    try:
        with open(fpath) as f:
            for l in f:
                l = l.rstrip()
                lines.append(l)
    except:
        error("File could not be read.")

    if len(lines) == 0 or (len(lines) == 1 and len(lines[0]) == 0):
        error("The file {} is empty.".format(fpath))

    subm_dict = dict()
    seen_date_set = set()
    for line_num, l in enumerate(lines):
        tpl = l.split("\t")
        if len(tpl) != 2:
            error("Error at line {}: Expected 2 tab-delimited columns.".format(line_num))

        date_str, od_matrix_str = tpl

        if not re.match(r"^\d{8}$", date_str):
            error("Error at line {}: The date must be in the format YYYYMMDD.".format(line_num))

        if not (date_str >= "20240201" and date_str <= "20240331"):
            error("Error at line {}: The date must be within the evaluation period 20240201 - 20240331.".format(line_num))

        if date_str in seen_date_set:
            error("Error at line {}: The date is duplicated.".format(line_num))
        seen_date_set.add(date_str)

        if date_str not in date_str_set:
            continue

        od_matrix = parse_od_matrix(line_num, od_matrix_str)
        subm_dict[date_str] = od_matrix

    subm_date_str_set = set(subm_dict.keys())
    date_str_set_diff = date_str_set - subm_date_str_set
    if len(date_str_set_diff) > 0:
        diff_list = sorted(list(date_str_set_diff))
        error("The submission is missing required dates: {}".format(diff_list))

    print("Validation passed!")

if __name__ == "__main__":
    main()
