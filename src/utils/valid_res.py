import os

def check_filename(filename):
    """Check if filename ends with 768.jpg, 364.jpg, or 1023.jpg"""
    return filename.endswith(("768.jpg", "384.jpg", "1024.jpg"))

def check_files_in_path(directory_path):
    """Check all files in a directory for the target endings"""
    matching_files = []
    
    if not os.path.exists(directory_path):
        print(f"Path '{directory_path}' does not exist!")
        return matching_files
    
    for filename in os.listdir(directory_path):
        if not check_filename(filename):
            full_path = os.path.join(directory_path, filename)
            matching_files.append(full_path)
            #print(f"Found: {filename}")
    
    return matching_files

# Example usage:
directory_path = "data/mix/images/conversations"  # Replace with your actual path

print(f"Checking files in: {directory_path}")
matches = check_files_in_path(directory_path)
print(f"\nFound {len(matches)} matching files:")


# Alternative: Check if ANY file exists with those endings
def has_matching_files(directory_path):
    """Returns True if directory contains any files with target endings"""
    if not os.path.exists(directory_path):
        return False
    
    return any(check_filename(f) for f in os.listdir(directory_path))