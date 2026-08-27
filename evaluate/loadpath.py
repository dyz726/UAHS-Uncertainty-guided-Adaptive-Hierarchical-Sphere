import os
import cv2


def load_saliency_map_from_png(path):
                    
    sal_map = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                                        
    if sal_map is None:
                                  
        raise ValueError(f"Saliency map in {path} could not be loaded.")
                               
    return sal_map.T              

def load_ground_map_from_png(path):
    """
    定义函数 `load_ground_truth_from_png`，用于加载真值图像而不进行额外处理。
    """
    gt_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                                               
                           
    if gt_image is None:
                                  
        raise ValueError(f"Ground truth image in {path} could not be loaded.")
                               
    return gt_image.T
                
def load_ground_fix_from_png(path):
    """
    Load the ground truth image without any processing.
    """
    gt_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                                                 
    if gt_image is None:
        raise ValueError(f"Ground fix image in {path} could not be loaded.")
    return gt_image.T

def get_ground_map_path(saliency_folder, saliency_file, ground_truth_root,DatasetName):
    """
    定义函数 `get_ground_truth_path`，用于根据显著性文件夹生成对应的真值图路径。
    """
    folder_name = os.path.basename(saliency_folder)             
                                                                  
                                           
                                                              
           
                                                             
    if DatasetName == "SVGC_AVA":
        ground_truth_path = os.path.join(ground_truth_root, folder_name, 'maps', f'{saliency_file}.png')
        return ground_truth_path

    if DatasetName == "Sports-360":
                      
        ground_truth_path = os.path.join(ground_truth_root, folder_name, 'maps', '{:04d}.png'.format(int(saliency_file)))
                                                        
        return ground_truth_path
                     
    if  DatasetName == "AVS-ODV":
        folder_name = os.path.basename(saliency_folder)                                        
        ground_truth_path = os.path.join(ground_truth_root, 'maps',folder_name, f'{saliency_file}.jpg')
        return ground_truth_path

    if DatasetName == "VR-EyeTracking":
        ground_truth_path = os.path.join(ground_truth_root, folder_name, 'maps','{:04d}.png'.format(int(saliency_file)))
        return ground_truth_path

def get_ground_fix_path(saliency_folder, saliency_namefile, ground_truth_root,Dataname):
    """
    Generate the corresponding ground truth path based on the saliency file folder name.
    """
    global ground_fix_path

    if Dataname=='Sports-360':
        folder_name = os.path.basename(saliency_folder)                                
        ground_fix_path = os.path.join(ground_truth_root, folder_name, 'fixation', f'{saliency_namefile}.png')
    if Dataname=='AVS-ODV':
        folder_name = os.path.basename(saliency_folder)
        ground_fix_path = os.path.join(ground_truth_root, 'fixation',folder_name, f'{saliency_namefile}fix.png')
    if Dataname=='SVGC_AVA':
        folder_name=os.path.basename(saliency_folder)
        ground_fix_path=os.path.join(ground_truth_root,folder_name,'fixation',f'{saliency_namefile}.png')

    if Dataname == "VR-EyeTracking":
        folder_name = os.path.basename(saliency_folder)
        ground_fix_path = os.path.join(ground_truth_root, folder_name, 'fixation', '{:04d}.png'.format(int(saliency_namefile)))

    return ground_fix_path



def get_all_ground_fix(ground_fix_root, Dataname):
    all_ground_truths = []

                  
    for folder_name in os.listdir(ground_fix_root):
        folder_path = os.path.join(ground_fix_root, folder_name)

                   
        if os.path.isdir(folder_path):
                                    
            maps_folder = os.path.join(folder_path, 'fixation')
            if os.path.isdir(maps_folder):
                for filename in os.listdir(maps_folder):
                    if filename.endswith('.png'):               
                        ground_truth_path = os.path.join(maps_folder, filename)

                        if Dataname == 'Sports-360':
                            name_file = os.path.splitext(filename)[0]
                        if Dataname == 'SVGC_AVA' or Dataname == "VR-EyeTracking":
                            name_without_ext = os.path.splitext(filename)[0]
                            name_file = str(int(name_without_ext))

                                                     
                        try:
                            ground_truth = load_ground_fix_from_png(ground_truth_path)
                            all_ground_truths.append(ground_truth)
                            print(f"Loaded ground fix for {name_file}...")
                        except Exception as e:
                            print(f"Error loading ground fix from {ground_truth_path}: {e}")

    return all_ground_truths