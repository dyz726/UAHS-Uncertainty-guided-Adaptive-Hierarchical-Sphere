import os
import numpy as np


from loadpath import *
                             
from utils_metrics import AUC_Judd, AUC_shuffled, CC, NSS, SIM,KLD

def evaluate_saliency_maps_in_folder(saliency_folder, ground_truth_root,DatasetName):
    """
    在指定的文件夹中评估所有显著性图
    :param saliency_folder: 包含显著性图的文件夹路径
    :param ground_truth_root: 对应的真值文件的根路径(fixation,map)
    """
                  
    auc_j_list = []                  
    nss_list = []             
    kl_div_list = []            
    sim_list = []             
    cc_list = []            
    auc_s_list = []                     
                                                                         

                       
    for folder_name in os.listdir(saliency_folder):
        folder_path = os.path.join(saliency_folder, folder_name)         

                
        if os.path.isdir(folder_path):
            print(folder_path)
            for filename in os.listdir(folder_path):                  
                if filename.endswith('.png'):             
                    saliency_path = os.path.join(folder_path, filename)           
                    if DatasetName in {"Sports-360", "AVS-ODV"}:
                        name_file = os.path.splitext(filename)[0]           
                    if DatasetName == "SVGC_AVA":
                        name_file = os.path.splitext(filename)[0]
                        name_file = str(int(name_file))
                    if DatasetName == "VR-EyeTracking":
                        name_file = os.path.splitext(filename)[0]
                        name_file = str(int(name_file) - 1)

                    try:
                                
                        sal_map = load_saliency_map_from_png(saliency_path)
                        if sal_map.any() == 0:
                            continue
                    except Exception as e:                    
                        print(f"Error loading saliency map from {saliency_path}: {e}")
                        continue

                               
                    ground_map_path = get_ground_map_path(folder_path, name_file, ground_truth_root,DatasetName)
                    ground_fix_path = get_ground_fix_path(folder_path, name_file, ground_truth_root, DatasetName)

           
                    if os.path.exists(ground_map_path) and os.path.exists(ground_fix_path):
                                                                      
                        try:
                            ground_map = load_ground_map_from_png(ground_map_path)         
                            ground_fix = load_ground_fix_from_png(ground_fix_path)                  
                        except Exception as e:                   
                            print(f"Error loading ground truth from {ground_map_path}: {e}")
                            continue

                              
                                                               
                                
                        auc_j = AUC_Judd(sal_map,ground_fix)
                        nss=NSS(sal_map,ground_fix)
                        kl_div=KLD(sal_map,ground_map)
                        sim=SIM(sal_map,ground_map)
                        cc=CC(sal_map,ground_map)
                                                                               
                                                                             
                                     
                                                                                                                                        
                                                
                                                                                                
                                      
                                            
                        auc_j_list.append(auc_j)
                        nss_list.append(nss)
                        kl_div_list.append(kl_div)
                        sim_list.append(sim)
                        cc_list.append(cc)
                                                 
                    else:
                        print(f"Ground truth for {filename} not found, skipping...",ground_map_path,ground_fix_path)

    if auc_j_list and nss_list and kl_div_list and sim_list and cc_list:
        print("\nAverage AUC-J: ", np.mean(auc_j_list))
        print("Average NSS: ", np.mean(nss_list))
        print("Average KL Divergence: ", np.mean(kl_div_list))
        print("Average SIM: ", np.mean(sim_list))
        print("Average CC: ", np.mean(cc_list))
    else:
        print("No valid results found.")


if __name__ == '__main__':

                              
    DatasetName = "AVS-ODV"
                                   

    model = "SphereUformer-split-2"
                             
    saliency_folder = '/home/dyz/PythonProject/DataSet_Output/'+DatasetName+'/Results/Results_Oth/Saliency/'+model +'/saliency_png'
                                                                                                             
    ground_truth_folder = '/home/dyz/PythonProject/Dataset/' + DatasetName 
                                                          
                            
    evaluate_saliency_maps_in_folder(saliency_folder, ground_truth_folder,DatasetName)

                                                                                          