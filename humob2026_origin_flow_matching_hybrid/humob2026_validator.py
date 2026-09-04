import sys
import re

def validate_submission(file_path):
    print(f"Checking file: {file_path}")
    
    date_pattern = re.compile(r"^2024(02|03)\d{2}$")
    grid_pattern = re.compile(r"^(\d+_\d+|-1_-1)$")
    
    target_dates = set()
    errors = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split("\t")
            if len(parts) != 2:
                errors.append(f"Line {line_idx}: Format error. Expected exactly 2 columns separated by tab.")
                continue
            
            date_str, dict_str = parts[0].strip(), parts[1].strip()
            
            if not date_pattern.match(date_str):
                errors.append(f"Line {line_idx}: Invalid date '{date_str}'. Must be in 20240201-20240331.")
            else:
                target_dates.add(date_str)
                
            try:
                data = eval(dict_str)
                if not isinstance(data, dict):
                    errors.append(f"Line {line_idx}: Column 2 is not a dictionary.")
                    continue
                
                if len(data) == 0:
                    errors.append(f"Line {line_idx}: Dictionary is empty for date {date_str}.")
                    
                for o_grid, d_dict in data.items():
                    if not grid_pattern.match(o_grid):
                        errors.append(f"Line {line_idx}: Invalid origin grid '{o_grid}'.")
                    if not isinstance(d_dict, dict) or len(d_dict) == 0:
                        errors.append(f"Line {line_idx}: Value for origin '{o_grid}' must be a non-empty dictionary.")
                    else:
                        for d_grid, weight in d_dict.items():
                            if not grid_pattern.match(d_grid):
                                errors.append(f"Line {line_idx}: Invalid destination grid '{d_grid}'.")
                            if not isinstance(weight, (int, float)) or weight < 0:
                                errors.append(f"Line {line_idx}: Invalid weight '{weight}' for {o_grid}->{d_grid}.")
            except Exception as e:
                errors.append(f"Line {line_idx}: Failed to parse dictionary. Error: {str(e)}")

            if len(errors) > 20:
                errors.append("...Too many errors, stopping check.")
                break

    if "20240202" in target_dates or "20240305" in target_dates:
        errors.append("Evaluation dates should not contain NA days (20240202, 20240305).")

    if errors:
        print(f"Validation FAILED with {len(errors)} errors:")
        for err in errors[:20]:
            print(f" - {err}")
        return False
    else:
        print("Validation passed!")
        print(f"Found {len(target_dates)} valid evaluation dates.")
        return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python humob2026_validator.py <path_to_submission.tsv>")
        sys.exit(1)
    
    file_to_check = sys.argv[1]
    success = validate_submission(file_to_check)
    if not success:
        sys.exit(1)
