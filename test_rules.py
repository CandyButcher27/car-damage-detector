from parts_rules import lookup as parts_lookup, get_image_region
from repair_rules import lookup as repair_lookup, get_severity

# Test get_severity
print("=== SEVERITY ===")
print(get_severity(100, 10000))   # 1% → minor
print(get_severity(800, 10000))   # 8% → moderate
print(get_severity(2000, 10000))  # 20% → severe

# Test get_image_region
print("\n=== REGIONS ===")
car_bbox = [0.5, 0.5, 0.8, 0.6]   # center, large car
print(get_image_region([0.3, 0.3, 0.1, 0.1], car_bbox))  # top_left
print(get_image_region([0.7, 0.3, 0.1, 0.1], car_bbox))  # top_right
print(get_image_region([0.5, 0.5, 0.05, 0.05], car_bbox)) # center

# Test parts lookup
print("\n=== PARTS ===")
print("scratches + center:", parts_lookup("scratches", "center"))
print("glass-crack + top_left:", parts_lookup("glass-crack", "top_left"))
print("flat-tire + bottom_left:", parts_lookup("flat-tire", "bottom_left"))
print("unknown class:", parts_lookup("unknown", "center"))

# Test repair lookup
print("\n=== REPAIR ===")
print("scratches + minor:", repair_lookup("scratches", "minor"))
print("severe-deformation + severe:", repair_lookup("severe-deformation", "severe"))
print("glass-crack + moderate:", repair_lookup("glass-crack", "moderate"))
print("unknown:", repair_lookup("unknown", "minor"))